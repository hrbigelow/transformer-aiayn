import re
import requests
import urllib
import fire
import google.auth
from google.auth.transport.requests import Request
from jinja2 import Environment, BaseLoader


"""
Generate an HTML text page with side-by-side translations
"""
def get_auth_token():
    auth_req = Request()
    creds, project = google.auth.default()
    creds.refresh(auth_req)
    return creds.token

def download_file(token, path):
    header = dict(Authorization=f'Bearer {token}')
    _, _, bucket, file = path.split('/', 3) 
    file = urllib.parse.quote(file, safe='')
    uri = f'https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{file}'
    content = requests.get(uri, headers=header, params={'alt': 'media'})
    return content.text

def color_gradient(index, total):
    ratio = index / total
    grow = int(255 * ratio)
    shrink = int(255 * (1 - ratio))
    r, g, b = shrink, 0, grow
    return f'rgb({r},{g},{b})'

def get_checkpoint(file, ckpt_regex):
    m = re.search(ckpt_regex, file)
    return int(m.group(1)) if m else None

template_string = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grouped Sentences</title>
    <style>
        .group {
            margin-bottom: 30px;
        }
        .line {
            display: flex;
            align-items: center;
        }
        .checkpoint {
            flex: 1;
            padding: 0 1em;
            width: 5em;
        }
        .sentence {
            flex: 2;
            width: 90%;
        }
        {% for color in colors %}
        .color{{ loop.index }} {
            color: {{ color }}
        }
        {% endfor %}
    </style>
</head>
<body>
    {% for group in groups %}
        <div class="group">
        <a name="{{ loop.index }}" href="#{{ loop.index }}"></a>
        {% for sentence in group %}
            <div class="line color{{ loop.index }}">
            <span class="checkpoint">{{ ckpt[loop.index0] }}</span>
            <span class="sentence">{{ sentence }}</span>
            </div>
        {% endfor %}
        </div>
    {% endfor %}
</body>
</html>
"""

def main(output_html, input_file, target_file, ckpt_regex, *result_files):
    token = get_auth_token()
    result_sentences = []
    checkpoints = []
    for result_file in result_files:
        checkpoint = get_checkpoint(result_file, ckpt_regex)
        if checkpoint is None:
            continue
        print(f'Processing {checkpoint}')
        content = download_file(token, result_file)
        lines = [ l.rstrip() or 'EMPTY' for l in content.split('\n') ]
        result_sentences.append(lines)
        checkpoints.append(checkpoint)

    input_fh = open(input_file, 'r')
    target_fh = open(target_file, 'r')

    input_lines = [ l.rstrip() for l in input_fh.readlines() ]
    target_lines = [ l.rstrip() for l in target_fh.readlines() ]
    N = len(result_sentences) 

    env = Environment(loader=BaseLoader())
    template = env.from_string(template_string)

    colors = ['rgb(0,0,0)'] + [color_gradient(i, N) for i in range(N)] + ['rgb(0,0,0)']

    all_sentences = [input_lines] + result_sentences + [target_lines]
    all_checkpoints = ['INPUT'] + checkpoints + ['TARGET']
    groups = list(zip(*all_sentences)) # [group][sentence]
    html = template.render(colors=colors, groups=groups, ckpt=all_checkpoints)

    with open(output_html, 'w') as fh:
        fh.write(html)


if __name__ == '__main__':
    fire.Fire(main)

