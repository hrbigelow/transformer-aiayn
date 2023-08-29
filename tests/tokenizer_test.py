import fire
from aiayn import data

def main(tokenizer_path, text_file):
    tz = data.get_tokenizer(tokenizer_path)
    with open(text_file, 'r') as fh:
        for line in fh:
            line = line.strip()
            rt = tz.decode(tz.encode(line).ids)
            if line != rt:
                print(f'Round trip failed:\n'
                      f'original  :  {line}\n'
                      f'round-trip:  {rt}')

if __name__ == '__main__':
    fire.Fire(main)

    
    
