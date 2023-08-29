import fire

"""
MHA: 4HMD
PFF: 2MF + F + M
EMB: VM

ENC_LAYER: MHA + PFF
DEC_LAYER: 2 * MHA + PFF 

DECODER: L * DEC_LAYER
ENCODER: L * ENC_LAYER

MODEL: DECODER + ENCODER + EMB
"""

def total_params(L,M,F,H,K,V,A):
    mha = 2*H*M*K + 2*H*M*V
    pff = 2*M*F + M + F
    emb = A*M
    enc_layer = mha + pff
    dec_layer = 2 * mha + pff
    decoder = L * dec_layer
    encoder = L * enc_layer
    model = decoder + encoder + emb
    print('L: num_layers')
    print('M: d_model, model dimension')
    print('F: d_ff, feed-forward dimension')
    print('H: number of heads')
    print('K: d_k, number of components in a key (and query)')
    print('V: d_v, number of components in a value')
    print('A: alphabet (vocabulary size)')
    print(f'L\t{L}')
    print(f'M\t{M}')
    print(f'H\t{H}')
    print(f'K\t{K}')
    print(f'V\t{V}')
    print(f'A\t{A}')
    print(f'mha\t{mha}')
    print(f'pff\t{pff}')
    print(f'all_mha\t{mha*L*2}')
    print(f'all_pff\t{pff*L*2}')
    print(f'emb\t{emb}')
    print(f'total\t{model}')


if __name__ == '__main__':
    fire.Fire(total_params)

