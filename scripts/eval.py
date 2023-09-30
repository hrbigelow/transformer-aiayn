# Evaluate results
import re
import json
import requests
import google.auth
from google.auth.transport.requests import Request
from streamvis.logger import DataLogger

from aiayn import bleu_tools
import fire

def one(ref_filename, hyp_filename):
    return bleu_tools.bleu_wrapper(ref_filename, hyp_filename)

def gcs_list(path_pattern):
    # path_pattern is a pattern starting with 'gs://' and possibly containing wildcards
    _, _, bucket, path = path_pattern.split('/', 3)

    auth_req = Request()
    creds, project = google.auth.default()
    creds.refresh(auth_req)
    headers = dict(Authorization=f'Bearer {creds.token}')
    params = dict(matchGlob=path, maxResults=1000)
    url = f'https://storage.googleapis.com/storage/v1/b/{bucket}/o/'
    response = requests.get(url, headers=headers, params=params)
    content = response.json()
    # print(content)
    items = content.get('items', [])
    make_path = lambda i: 'gs://' + i['bucket'] + '/' + i['name']
    files = [ make_path(item) for item in items ]
    return files

def get_checkpoint(file, ckpt_regex):
    m = re.search(ckpt_regex, file)
    return int(m.group(1))

def log_all(ref_filename, gcs_path, ckpt_regex, streamvis_path, scope, data_name):
    logger = DataLogger(scope)
    logger.init(streamvis_path, 100)
    file_list = gcs_list(gcs_path)
    ckpt_file_list = [ (get_checkpoint(file, ckpt_regex), file) for file in file_list ]
    ckpt_file_list.sort()
    for ckpt, file in ckpt_file_list:
        bleu = one(ref_filename, file) * 100.0
        logger.write(data_name, x=ckpt, y=bleu)
        print(f'{ckpt}: {bleu:2.3f}')
    logger.flush_buffer()


if __name__ == '__main__':
    cmds = dict(one=one, log_all=log_all)
    fire.Fire(cmds)


