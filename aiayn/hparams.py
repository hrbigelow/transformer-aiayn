# All keys that appear in each entry of HPARAMS_REGISTRY must also appear in
# some entry of DEFAULTS
HPARAMS_REGISTRY = {}
DEFAULTS = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f'attribute {attr} undefined')

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __repr__(self):
        return '\n'.join(f'{k}: {v}' for k, v in self.items())


def setup_hparams(hparam_set_names, kwargs):
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(",")
    hparam_sets = [HPARAMS_REGISTRY[x.strip()] for x in hparam_set_names if x] + [kwargs]
    for k, v in DEFAULTS.items():
        H.update(v)
    for hps in hparam_sets:
        for k in hps:
            if k not in H:
                raise ValueError(f"{k} not in default args")
        H.update(**hps)
    H.update(**kwargs)
    return H


arch = Hyperparams(
    H = 8, # heads
    K = 64, # key (d_k in paper)
    V = 64, # value (d_v in paper)
    M = 512, # model (d_model in paper)
    F = 2048, # feed-forward dimension (d_ff in paper)
    num_layers = 6
)

tiny = Hyperparams(
    H = 2, # heads
    K = 8, # key (d_k in paper)
    V = 8, # value (d_v in paper)
    M = 13, # model (d_model in paper)
    F = 19, # feed-forward dimension (d_ff in paper)
    num_layers = 2
)

reg = Hyperparams(
    # Section 5.4: Regularization (P_drop = 0.1)
    dropout_rate = 0.1,

    # mixture coefficient for positional encoding
    pos_encoding_factor = 1.0 
)

train = Hyperparams(
    batch_dim0 = 64,
    val_loop_elem = 16,
    accum_steps = 8,
    adam_beta1 = 0.9,
    adam_beta2 = 0.98,
    adam_eps = 1e-9,
    label_smooth_eps = 0.1, # see page 8
    with_attn_entropy = False,
    attn_loss_weight = 0.0,
    warmup_steps = 4000,
    random_seed = 982349820,
    ckpt_every = 5000,
    ckpt_dir = None,
    ckpt_max_keep = 20,
    resume_ckpt = None,
    )

logging = Hyperparams(
    report_every = 100,
    eval_every = 100,
    with_metrics = False,
    streamvis_run_name = None,
    streamvis_path = None,
    streamvis_buffer_items = None
    )

data = Hyperparams(
    dataset_glob = None, # either filesystem or gs:// locator glob pattern
    val_dataset_glob = None,
    tokenizer_file = None,
    shuffle_size = 1000, 
    swap_source_target = True,
    max_source_len = 80,
    max_target_len = 128,
    ckpt_has_last_batch = False
    )

sample = Hyperparams(
    temperature = 1.0,
    random_seed = 42,
    ckpt_dir = None,
    resume_ckpt = None,
    force_save_ckpt = False,
    num_sample = 10,
    beam_size = 4, # From section 6.1
    beam_search_alpha = 0.6, # From section 6.1 
    beam_search_beta = 0.4, # see https://arxiv.org/pdf/1609.08144.pdf Table 2 
    beam_search_maxlen = 100 
    )


HPARAMS_REGISTRY['tiny'] = tiny
HPARAMS_REGISTRY['arch'] = arch
HPARAMS_REGISTRY['reg'] = reg
HPARAMS_REGISTRY['train'] = train
HPARAMS_REGISTRY['data'] = data
HPARAMS_REGISTRY['logging'] = logging 
HPARAMS_REGISTRY['sample'] = sample

DEFAULTS['arch'] = arch
DEFAULTS['reg'] = reg
DEFAULTS['train'] = train
DEFAULTS['data'] = data
DEFAULTS['logging'] = logging
DEFAULTS['sample'] = sample

