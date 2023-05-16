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
    num_layers = 6,
    T = None # number of tokens
)

reg = Hyperparams(
    # Section 5.4: Regularization (P_drop = 0.1)
    dropout_rate = 0.1,

    # mixture coefficient for positional encoding
    pos_encoding_factor = 0.01
)

train = Hyperparams(
    batch_size = 128,
    sub_batch_size = 8,
    adam_beta1 = 0.9,
    adam_beta2 = 0.98,
    adam_eps = 1e-9,
    warmup_steps = 4000,
    random_seed = 982349820,
    ckpt_every = 5000,
    report_every = 100,
    resume_ckpt = None,
    ckpt_templ = None
    )

data = Hyperparams(
    data_path = None,
    bin_size = 1000,
    dataset_size = None,
    pad_token_id = None,
    max_sentence_length = 200,
    )


HPARAMS_REGISTRY['arch'] = arch
HPARAMS_REGISTRY['reg'] = reg
HPARAMS_REGISTRY['train'] = train
HPARAMS_REGISTRY['data'] = data

DEFAULTS['arch'] = arch
DEFAULTS['reg'] = reg
DEFAULTS['train'] = train
DEFAULTS['data'] = data
