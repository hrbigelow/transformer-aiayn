import tensorflow_datasets as tfds

def download_wmt14_dataset():
    # Define the dataset configuration
    dataset_name = 'wmt14_translate/{}-{}'
    lang_pairs = [('en', 'de')]  # English to German
    # splits = tfds.Split.TRAIN.subsplit(tfds.percent[:100])  # Use the entire training set

    # Download the dataset
    dataset_config = tfds.translate.wmt.WmtConfig(
        description='wmt14_translate/en-de',
        version='0.0.1',
        language_pair=(src_lang, tgt_lang),
        subsets={
            tfds.Split.TRAIN: ['commoncrawl'],
        }
    )
    builder = tfds.builder(dataset_config)
    builder.download_and_prepare(download_dir='/home/henry/ai/data/', download=True,
            split='train')

download_wmt14_dataset()


def token_fn(item):
    en = tokenizer(item['en'])['input_ids']
    de = tokenizer(item['de'])['input_ids']
    return dict(en=en, de=de)


        


