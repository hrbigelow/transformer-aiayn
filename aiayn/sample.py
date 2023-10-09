import tensorflow as tf
import gc
import fire
import sys
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from aiayn import model, data, hparams
from tokenizers import decoders
import pdb

jnp.set_printoptions(precision=3, threshold=100000, edgeitems=100, linewidth=180)

def load_model(hps, bos_id, eos_id, n_vocab):
    mod = model.make_model(hps, bos_id, eos_id, n_vocab, do_batch=False, do_train=False) 
    mngr = orbax.CheckpointManager(
        hps.ckpt_dir, orbax.Checkpointer(orbax.PyTreeCheckpointHandler()))
    state = mngr.restore(hps.resume_ckpt)
    params = state['params']
    return mod, params

def predict_interactive(mod, params, tokenizer, special_toks, hps): 
    # print('Restored model from checkpoint')
    rng_key = jax.random.PRNGKey(hps.random_seed)

    print('Enter text (Ctrl-D to exit)')

    try:
        while True:
            query = input('> ')
            if len(query) == 0:
                continue
            queries = [query] * hps.batch_dim0
            encodings = tokenizer.encode_batch(queries)
            query_toks = np.array([e.ids for e in encodings])
            pred_toks, pred_scores, _, _ = mod.apply(params, rng_key, query_toks,
                    special_toks.pad_id, hps.beam_search_alpha, hps.beam_search_beta,
                    hps.beam_size, hps.max_source_len, hps.max_target_len)
            pred_scores0 = pred_scores[0].tolist()
            # print('Inference for batch element 0:')
            pred_sentences = tokenizer.decode_batch(pred_toks[0])
            # pdb.set_trace()
            for score, sentence in zip(pred_scores0, pred_sentences):
                print(f'{score:0.2f} {sentence}')
            print('\n')
    except EOFError:
        print('Bye')
        sys.exit(0)

def batch_gen(it, batch_size):
    def gen():
        try:
            while True:
                batch = []
                for _ in range(batch_size):
                    batch.append(next(it))
                yield batch
        except StopIteration:
            if len(batch) > 0:
                yield batch
    return gen()

def predict_batch(mod, params, tokenizer, special_toks, batch_file, out_file, hps):
    """
    Generate translations for each input sentence in batch_file.
    tokenizer:  must have pad_token_id and eos_token_id set
    batch_file has lines of the form:
    sentence
    """
    tokenizer.enable_padding(length=hps.max_source_len, pad_id=special_toks.pad_id,
            pad_token='[PAD]')
    n_vocab = tokenizer.get_vocab_size() + 2

    out_fh = tf.io.gfile.GFile(out_file, 'w')
    fh = tf.io.gfile.GFile(batch_file, 'r')
    bit = batch_gen(fh, hps.batch_dim0)
    print(f'Working on {out_file}: ', end='', flush=True)

    for chunk, lines in enumerate(bit):
        lines = [ l.strip() for l in lines ]
        inputs = tokenizer.encode_batch(lines)
        inputs = jnp.array([item.ids for item in inputs])
        # print('inputs: ', inputs)
        # pdb.set_trace()
        args = (special_toks.pad_id, hps.beam_search_alpha, hps.beam_search_beta,
                hps.beam_size, hps.max_source_len, hps.max_target_len)

        pred_toks, pred_scores = mod.apply(params, inputs, *args)
        # print(pred_toks.shape)
        top_toks = pred_toks[:,0]
        # pdb.set_trace()
        top_scores = pred_scores[:,0].tolist()
        top_seqs = tokenizer.decode_batch(top_toks)
        for i in range(len(top_seqs)):
            # _id = ids[i]
            seq = top_seqs[i]
            score = top_scores[i]
            # out_fh.write(f'{score}\t{seq}\n')
            out_fh.write(f'{seq}\n')
            # print(f'{_id}\t{score:2.3f}\t{seq}')
        out_fh.flush()
        print(f'.', end='', flush=True)

    fh.close()
    out_fh.close()
    print(f'  done.')

def evaluate_bleu(ref_file, hyp_file):
    """
    Evaluate the Bleu score for the hypothesis file `hyp_file` against the reference
    translation file `ref_file`
    """
    print(f'Computing Bleu score for hypothesis file {hyp_file} against '
            f'reference file {ref_file}')
    bleu = bleu_tools.bleu_wrapper(ref_file, hyp_file)
    print(f'Bleu Score: {bleu:2.3f}')

def sample(ckpt_dir, resume_ckpt, tokenizer_file, batch_file=None, out_template=None,
        hps_keys: str = 'arch,reg,data,sample', **hps_overrides):
    hps = hparams.setup_hparams(hps_keys,
            dict(
                ckpt_dir=ckpt_dir, 
                resume_ckpt=resume_ckpt,
                **hps_overrides)
            )
    out_file = out_template.format(resume_ckpt)
    special_toks = data.get_special_tokens(tokenizer_file) 
    tokenizer = data.get_tokenizer(tokenizer_file)
    tokenizer.enable_padding(pad_id=special_toks.pad_id, pad_token='[PAD]')
    tokenizer.decoder = decoders.ByteLevel()
    n_vocab = tokenizer.get_vocab_size() + 2 # 
    mod, params = load_model(hps, special_toks.bos_id, special_toks.eos_id, n_vocab)
    print(f'Loaded model from {ckpt_dir}/{resume_ckpt}')
    if batch_file is None:
        predict_interactive(mod, params, tokenizer, special_toks, hps)
    else:
        predict_batch(mod, params, tokenizer, special_toks, batch_file, out_file, hps)

def all(ckpt_dir, resume_ckpts, tokenizer_file, batch_file, out_template,
        hps_keys: str = 'arch,reg,data,sample', **hps_overrides):
    for resume_ckpt in resume_ckpts:
        out_file = out_template.format(resume_ckpt)
        main(ckpt_dir, resume_ckpt, tokenizer_file, batch_file, out_file, hps_keys,
                **hps_overrides)


if __name__ == '__main__':
    cmds = dict(sample=sample, all=all, evaluate=evaluate_bleu)
    fire.Fire(cmds)

