import os
import signal
import queue
import sys
import fire
import numpy as np
from aiayn import model, data, pause
from aiayn.data import load_token_histo
import torch
import torch.distributed as dist
from torch.optim import Adam
from streamvis import DataLogger 
from aiayn.hparams import setup_hparams, Hyperparams

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.experimental.pjrt_backend
except ImportError:
    pass

class Run(pause.Pause):
    """
    Encapsulates all hyperparams and state information for the whole run.
    Pause implements the checkpointing save/restore operation.
    """
    def __init__(self, use_xla):
        super().__init__(use_xla)

    def _validate(self, params):
        if params.pubsub_project is None and params.streamvis_log_file is None:
            raise RuntimeError(
                f'At least one of `pubsub_project` or `streamvis_log_file` must be provided')

        if params.batch_size % params.sub_batch_size != 0:
            raise RuntimeError(
                f'{params.batch_size=} must be disivible by {params.sub_batch_size=}')

    def _make(self, params, state=None):
        torch.manual_seed(self.params.random_seed)
        token_info = load_token_histo(params.data_path) 
        pad_token_id = token_info['pad_token_id']
        self.model = model.Model(params, token_info) 


        if params.infra_mode in ('tpu_colab', 'tpu_vm'):
            self.serial_exec = xmp.MpSerialExecutor()
            # self.wrapped_model = xmp.MpModelWrapper(self.model)

        self.logger = DataLogger('aiayn')
        if params.pubsub_project is not None:
            self.logger.init_pubsub(params.pubsub_project, params.pubsub_topic)
        if params.streamvis_log_file is not None:
            self.logger.init_write_log(params.streamvis_log_file)

        if state is None:
            return

        model_state = state['model']
        self.model.load_state_dict(model_state['weights'])
        self.model.rng.set_state(model_state['rng'])
        self.sampler.load(state['sampler'])
        self.opt.load_state_dict(state['optim'])

    def device_init(self, device):
        """
        Device specific initialization
        If on single GPU, called once.
        If on TPU, called in each spawned process
        """

        if self.use_xla:
            shard, num_shards = xm.get_ordinal(), xm.xrt_world_size()
            if self.params.batch_size % num_shards != 0:
                raise RuntimeError(
                    f'batch_size {self.params.batch_size} not divisible by {num_shards=}')

            self.shard_size = self.params.batch_size // num_shards
            self.num_shards = num_shards

            def get_ds():
                return data.get_dataset(self.params.data_path,
                        self.params.max_sentence_length, num_proc=4)
            self.dataset = self.serial_exec.run(get_ds)
            dataset_size = len(self.dataset)
            self.sampler = data.BatchedSampler(dataset_size, self.params.batch_size,
                    self.params.bin_size, self.params.random_seed, shard, num_shards)
            pad_loader = data.PadLoader(self.params.max_sentence_length,
                    self.model.pad_token_id, self.dataset, self.sampler) 
            para_loader = pl.ParallelLoader(pad_loader, [device])
            per_device_loader = para_loader.per_device_loader(device)
            self.loader = per_device_loader
            # self.model = self.wrapped_model.to(device)
            self.model = self.model.to(device)

            betas = self.params.adam_beta1, self.params.adam_beta2
            self.opt = Adam(self.model.parameters(), betas=betas, eps=self.params.adam_eps)
            self.sched = model.CustomScheduler(self.opt, self.params.M,
                    self.params.warmup_steps)

            if self.params.compile_backend is not None:
                import torch._dynamo
                torch._dynamo.config.verbose=True
                self.model = torch.compile(self.model,
                        backend=self.params.compile_backend)
            # print(f'Run::device_init: {shard=}, {num_shards=}', flush=True)
        else:
            pass

    def _get_state(self):
        state = {}
        state['model'] = self.model.get_state() 
        state['sampler'] = self.sampler.state()
        state['optim'] = self.opt.state_dict()
        return state

    def to(self, device):
        self.model.to(device)
        # self.loader.dataset.to(device)
        # self.opt.to(device)

def test_handler(signum, frame):
    print(f'in test_handler')

def _mp_fn(rank, use_pjrt, resume_ckpt, hps_overrides):
    if use_pjrt:
        dist.init_process_group('xla', init_method='pjrt://')
    # signal.signal(signal.SIGINT, test_handler)
    run = Run(True)

    if resume_ckpt is None:
        hps_keys = hps_overrides.pop('hps_keys')
        hps = setup_hparams(hps_keys, hps_overrides)
        run.init(hps)
    else:
        # print(f'Resuming from {path}')
        run.load(resume_ckpt, **hps_overrides)

    device = xm.xla_device()

    xm.master_print('Running with parameters:')
    xm.master_print(run.params)
    xm.master_print(f'Total model params: {run.model.total_params()}')

    run.device_init(device)
    # TODO: should be set according to save/restore
    run.sched.update(0)

    # run.model.to(device) doesn't work? 
    train_loop_xla(run)

def collect_log(tensor, step):
    result = xm.all_reduce(reduce_type=xm.REDUCE_SUM, inputs=tensor)
    xm.mark_step()
    print(f'got here in collect_log: xla:{xm.get_ordinal()}, {xm.is_master_ordinal()=}', flush=True)
    if xm.is_master_ordinal():
        cpu_result = result.cpu().tolist()
        for s, item in enumerate(cpu_result, step):
            xm.master_print(f'step: {s}, {item:5.4f}', flush=True)

def train_loop_xla(run):
    print(f'xla:{xm.get_ordinal()}: In train_loop_xla', flush=True)
    run.model.train()

    loss = torch.zeros(run.params.report_every, device=xm.xla_device())
    loss_fraction = (run.params.update_every * run.num_shards) ** -1

    for enc_input, dec_input, load_step, epoch in run.loader:
        step = load_step // run.params.update_every
        run.sched.update(step)
        # param = run.model.get_parameter('encoder.body.0.att.wq') 
        # print(f'xla:{xm.get_ordinal()}: {param[0,0,0:5]}')

        # enc_layer0 = r'encoder.body.0.att.wq'
        # layer0_norms = torch.empty(batch_shape)
        dec_output = run.model(enc_input, dec_input)

        # scale loss by batch_fraction
        xent = run.model.loss(dec_input, dec_output) * loss_fraction
        xent.backward()

        with torch.no_grad():
            loss[step % run.params.report_every] += xent

        # accumulate gradients over sub-batches
        if load_step % run.params.update_every == 0:
            xm.optimizer_step(run.opt)
            run.opt.zero_grad()

            if step % run.params.report_every == 0 and step > 0:
                combined_loss = xm.all_reduce(xm.REDUCE_SUM, loss)
                loss_vals = combined_loss.cpu()
                lr = run.sched.current_lr()
                xm.master_print(f'{epoch=}, {step=}, {lr=:6.5f}, loss={loss_vals}')
                loss.fill_(0.0)

        # zero out specific gradients for later inspection
        # layer0_grads = run.model.zero_gradients(enc_layer0)


        # norms: pat, B (only one pat here)
        # norms = run.model.grad_norms(enc_layer0, index_dims=(0,)) 
        # layer0_norms[sub_batch,:] = norms[0]

        # add back copied gradients
        # run.model.add_gradients(layer0_grads)

        # protect to avoid partial update
        # old_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)

        # signal.signal(signal.SIGTERM, old_handler)

        """
        loss_metrics = [loss]
        run.logger.tandem_lines('loss', step, loss_metrics, 'Viridis256')
        run.logger.tandem_lines('lr', step, [lr], 'Viridis256')
        run.logger.tandem_lines('epoch', step, [epoch])
        # run.logger.tandem_lines('en_lengths', step, en_lengths, 'Viridis256')
        # run.logger.tandem_lines('de_lengths', step, de_lengths, 'Viridis256')

        # layer0_norms = layer0_norms.reshape(run.params.batch_size).cpu().numpy()
        # logger.tandem_lines('enc_layer0', step, layer0_norms, 'Viridis256')
        for plot, pattern in param_patterns.items():
            norms = run.model.grad_norms(pattern) 
            run.logger.tandem_lines(plot, step, norms, 'Viridis256')
        """
        # combined_loss = xm.all_reduce(xm.REDUCE_SUM, loss)
        # xm.master_print(f'{epoch=}, {step=}, {loss=}', flush=True)
        # combined_loss_cpu = combined_loss.cpu()
        # xm.master_print(f'{step=}, {combined_loss_cpu=}', flush=True)

        # if step % run.params.report_every == 0:
            # xm.add_step_closure(collect_log, args=(loss, step))
            # xm.master_print(f'{epoch=}, {step=}, {lr=:7.6f}, {loss=:5.4f}', flush=True) 

        if step % run.params.ckpt_every == 0 and step > 0 and step != run.params.resume_ckpt:
            path = run.params.ckpt_templ.format(step)
            print(f'Saving {path}', flush=True)
            run.save(path)

        if step % 50 == 0 and step > 0:
            xm.master_print(met.metrics_report(), flush=True)

def main(hps_keys: str = 'arch,reg,train,data,logging', 
        resume_ckpt: str = None,
        data_path: str = None, 
        batch_size: int = None,
        update_every: int = None,
        ckpt_templ: str = None,
        max_sentence_length: int = None,
        report_every: int = None,
        ckpt_every: int = None,
        pubsub_project: str = None,
        pubsub_topic: str = None,
        streamvis_log_file: str = None,
        infra_mode: str = None,
        compile_backend: str = None):
    """
    :param resume_ckpt:
           if present, resume from this checkpoint
    :param data_path:
           path to dataset prepared using python -m aiayn.data script
    :param pubsub_project: the GCP project with Cloud Pub/Sub API enabled
    :param pubsub_topic: the GCP topic associated with pubsub_project
    :param streamvis_log_file: path to streamvis log file (optional) 

    # optional command-line overrides
    :param batch_size: SGD batch size
    :param update_every: number of loader steps to accumulate gradients for before
                         taking an optimizer step
    :param ckpt_templ: checkpoint file path containing literal {} to be substituted with 
                       step value
    :param max_sentence_length: skip data batches containing token sequences longer
                                than this, to avoid OOM errors
    :param report_every:
           every number of steps to issue progress message to stdout
    :param ckpt_every:
           create a checkpoint every `ckpt_every` steps
    :param compile_backend: torch.compile backend name to use.  Do not compile if None 
    :param infra_mode: one of tpu_colab, tpu_vm, gpu
    """
    hps_overrides = { k: v for k, v in locals().items() if v is not None }
    hps_overrides.pop('resume_ckpt', None)
    # make a copy

    """
    def shutdown_handler(signum, frame):
        run.logger.shutdown()
        if run.sched.step > 1000:
            path = run.params.ckpt_templ.format(run.sched.step)
            run.save(path)
        sys.exit(0)

    # signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    run.sched.update(0)

    param_patterns = dict(
        decoder_masked_attention = r'decoder.body.(\d+).mask_att..*',
        decoder_attention2 = r'decoder.body.(\d+).att2..*',
        decoder_feed_forward = r'decoder.body.(\d+).ff..*',
        enc_attention_wq = r'encoder.body.(\d+).att.wq',
        enc_attention_wk = r'encoder.body.(\d+).att.wk',
        enc_attention_wv = r'encoder.body.(\d+).att.wv',
        enc_feed_forward = r'encoder.body.(\d+).ff..*'
        )
    """
    # qu = mp.Queue()
    # def shutdown_handler(signum, frame):
        # parent.send('cleanup')

    # signal.signal(signal.SIGINT, shutdown_handler)

    if infra_mode == 'tpu_colab':
        args = False, resume_ckpt, hps_overrides
        xmp.spawn(_mp_fn, args=args, nprocs=8, start_method='fork')
    elif infra_mode == 'tpu_vm':
        args = True, resume_ckpt, hps_overrides
        xmp.spawn(_mp_fn, args=args)
    elif infra_mode == 'gpu':
        pass
    else:
        raise RuntimeError(
            f'Got {infra_mode=}, but must be one of \'tpu_colab\', \'tpu_vm\', or \'gpu\'')
        # num_cores = 8 if os.environ.get('TPU_NAME', None) else 1
        # xmp.spawn(_mp_fn, args=(run,), nprocs=num_cores, start_method='fork')
        # if you use MpModelWrapper, use 'fork' method
        # xmp.spawn(_mp_fn, args=(run,), nprocs=None, start_method='fork')
        # xmp.spawn(_mp_fn, args=(resume_ckpt, hps_overrides), nprocs=None)


if __name__ == '__main__':
    fire.Fire(main)

