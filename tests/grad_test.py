import fire
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from aiayn import model, hparams, data

def main(token_info_file, data_path, hps_keys: str = 'arch,tiny,train,data'):
    jnp.set_printoptions(precision=2, threshold=100000, edgeitems=100, linewidth=180)

    defaults = dict(batch_dim0=2, accum_steps=1, max_sentence_length=15,
            token_info_file=token_info_file)
    hps = hparams.setup_hparams(hps_keys, defaults)
    print('Hyperparams:\n', hps)
    token_info = np.load(hps.token_info_file)
    mask_id = token_info['mask']
    mod = model.make_model(hps, True, token_info)
    objective = model.make_objective(hps, token_info) 

    # this call must be before data.main_dataset, or you get:
    # Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
    rng_key = jax.random.PRNGKey(42)

    dataset = data.main_dataset(data_path, token_info, hps.max_sentence_length,
            hps.batch_dim0, hps.swap_source_target, shuffle_size=100)
    enc_input, dec_input, _, _ = next(iter(dataset))
    enc_input = enc_input.astype(jnp.float32)
    dec_input = dec_input.astype(jnp.float32)

    def loss_fn(params, rng_key, enc_input, dec_input):
        dec_output = mod.apply(params, rng_key, enc_input, dec_input)
        loss, _, _ = objective.apply(None, None, dec_input, dec_output)
        return loss

    key1, key2 = jax.random.split(rng_key)
    grad_fn = jax.grad(loss_fn, argnums=(2,3))
    params = mod.init(key1, enc_input, dec_input)
    loss_fn(params, key2, enc_input, dec_input)

    enc_grad, dec_grad = grad_fn(params, key2, enc_input, dec_input)
    # tree = jax.tree_util.tree_structure(grad_pars)
    print(enc_grad)
    print(dec_grad)
    # print(emb_grad)
    # jax.debug.print('emb_grad: {}', emb_grad)
    # jnp.equal(emb_grad, 0.0)
    # for k, v in grad_pars.items():
        # print(f'{k}: {v.keys()}')
    # jax.debug.print('grad_pars: {}', grad_pars)


if __name__ == '__main__':
    fire.Fire(main)

    

    
