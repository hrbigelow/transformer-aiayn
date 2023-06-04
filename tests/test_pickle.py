from aiayn.train import Run
from aiayn.hparams import setup_hparams
import pickle

run = Run(use_xla=True)
kwargs = dict(data_path='/home/henry/ai/data/wmt14') 
hps = setup_hparams('arch,reg,train,data,logging', kwargs)
run.init(hps)

# del run.model
mod = pickle.dumps(run)


