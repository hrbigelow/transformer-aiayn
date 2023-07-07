# Introduction

An original implementation of the paper [Attention is All You
Need](https://arxiv.org/pdf/1706.03762.pdf) by Vaswani et al.

![Preliminary Loss](assets/jul7-loss-6k.png)

Shown above is loss using batch size 50,000 (number of input plus output tokens) on
the whole (4.5M samples) WMT-14 database.  720 sentence-pairs in each batch.
Training is on TPU.

## Getting Started

    pip install git+https://github.com/hrbigelow/transformer-aiayn.git

    # download and prepare the WMT14 dataset
    python -m aiayn.data \
      --cache_dir ~/ai/data/.cache \
      --data_dir ~/ai/data/wmt14 \
      --num_proc 8

    # pass the --shard option to prepare only a fraction of it:
    python -m aiayn.data \
      --cache_dir ~/ai/data/.cache \
      --data_dir ~/ai/data/wmt14 \
      --num_proc 8 \
      --shard '(100, 5)'

    # train the model
    python -m aiayn.train \
      None \
      --data_path ~/ai/data/wmt14 \
      --batch_size 64 \
      --update_every 4 \
      --ckpt_every 500 \
      --ckpt_templ ~/ai/ckpt/aiayn/may28-run{}.ckpt \
      --report_every 10 \
      --max_sentence_length 25 \
      --pubsub_project $PROJECT_ID \
      --pubsub_topic aiayn \
      --streamvis_run_name may26-tpu




