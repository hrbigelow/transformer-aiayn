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

def download_file(token, bucket, file):
    header = dict(Authorization=f'Bearer {token}')
    file = urllib.parse.quote(file, safe='')
    uri = f'https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{file}'
    content = requests.get(uri, headers=header, params={'alt': 'media'})
    return content.text

def color_gradient(index, total):
    ratio = index / total
    r = 0
    g = int(255 * ratio)
    b = int(255 * (1 - ratio))
    return f'rgb({r},{g},{b})'

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
        .sentence {
            margin-top: 5px;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    {% for group in groups %}
    <div class="group">
        {% for color, sentence in group %}
        <p class="sentence" style="color: {{ color }}">{{ sentence }}</p>
        {% endfor %}
    </div>
    {% endfor %}
</body>
</html>
"""

def main(bucket, output_html, input_file, target_file, *result_files):
    token = get_auth_token()
    result_sentences = []
    for result_file in result_files:
        content = download_file(token, bucket, result_file)
        lines = content.split('\n')
        result_sentences.append(lines)

    input_fh = open(input_file, 'r')
    target_fh = open(target_file, 'r')

    input_lines = input_fh.readlines()
    target_lines = target_fh.readlines()
    N = len(result_sentences) 

    env = Environment(loader=BaseLoader())
    template = env.from_string(template_string)

    colors = ['rgb(0,0,0)'] + [color_gradient(i, N) for i in range(N)] + ['rgb(0,0,0)']
    all_sentences = [input_lines] + result_sentences + [target_lines]
    groups = [list(zip(colors, sen)) for sen in list(zip(*all_sentences))]
    html = template.render(groups=groups)

    with open(output_html, 'w') as fh:
        fh.write(html)


if __name__ == '__main__':
    fire.Fire(main)

