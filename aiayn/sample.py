import fire
import sys
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from aiayn import model, data, hparams
import pdb


def load_model(hps, bos_id, eos_id, n_vocab):
    mod = model.make_model(hps, bos_id, eos_id, n_vocab, False) 
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
            pred_toks, pred_scores = mod.apply(params, rng_key, query_toks,
                    special_toks.pad_id, hps.beam_search_alpha,
                    hps.beam_search_beta, hps.beam_size, hps.max_target_len)
            pred_scores0 = pred_scores[0].tolist()
            # print('Inference for batch element 0:')
            pred_sentences = data.de_tokenize(tokenizer, pred_toks[0], special_toks.eos_id)
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

def predict_batch(mod, params, tokenizer, special_toks, batch_file, hps):
    """
    Generate translations for each input sentence in batch_file.
    tokenizer:  must have pad_token_id and eos_token_id set
    batch_file has lines of the form:
    sentence
    """
    rng_key = jax.random.PRNGKey(hps.random_seed)
    n_vocab = tokenizer.get_vocab_size() + 2

    fh = open(batch_file, 'r')
    bit = batch_gen(fh, hps.batch_dim0)
    for lines in bit:
        lines = [ l.strip() for l in lines ]
        # lines = [ l.split('\t') for l in lines if l is not None ]
        # ids = [ int(item) for item, _, _ in lines ]
        # pairs = [ (a.strip(), b.strip()) for _, a, b in lines ]
        # first = [ f for f,_ in pairs ]
        # second = [ s for _,s in pairs ]
        inputs = tokenizer.encode(lines, special_toks.pad_value)
        # pdb.set_trace()

        pred_toks, pred_scores = mod.apply(params, rng_key, inputs,
                pad_value, hps.beam_search_alpha, hps.beam_search_beta,
                hps.beam_size, hps.max_target_len)
        # print(pred_toks.shape)
        top_toks = pred_toks[:,0]
        # pdb.set_trace()
        top_scores = pred_scores[:,0].tolist()
        top_seqs = data.de_tokenize(tokenizer, top_toks, special_toks.eos_id)
        for i in range(len(top_seqs)):
            # _id = ids[i]
            seq = top_seqs[i]
            score = top_scores[i]
            print(seq)
            # print(f'{_id}\t{score:2.3f}\t{seq}')
    fh.close()

def main(ckpt_dir, resume_ckpt, tokenizer_file, batch_file = None,
        hps_keys: str = 'arch,reg,data,sample', **hps_overrides):
    hps = hparams.setup_hparams(hps_keys,
            dict(
                ckpt_dir=ckpt_dir, 
                resume_ckpt=resume_ckpt,
                **hps_overrides)
            )
    special_toks = data.get_special_tokens(tokenizer_file) 
    tokenizer = data.get_tokenizer(tokenizer_file)
    tokenizer.enable_padding(pad_id=special_toks.pad_id, pad_token='[PAD]')
    n_vocab = tokenizer.get_vocab_size() + 2 # 
    mod, params = load_model(hps, special_toks.bos_id, special_toks.eos_id, n_vocab)
    if batch_file is None:
        return predict_interactive(mod, params, tokenizer, special_toks, hps)
    return predict_batch(mod, params, tokenizer, special_toks, batch_file, hps)

if __name__ == '__main__':
    fire.Fire(main)

