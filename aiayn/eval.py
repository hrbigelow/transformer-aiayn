# Evaluate results
from aiayn import bleu_tools
import fire

def main(ref_filename, hyp_filename):
    bleu = bleu_tools.bleu_wrapper(ref_filename, hyp_filename)
    print(f'Bleu score: {bleu:2.3f}')

if __name__ == '__main__':
    fire.Fire(main)


