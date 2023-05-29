import torch
from aiayn import hparams
from abc import ABC, abstractmethod
from inspect import signature

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    pass

class Pause(ABC):
    """
    """
    def __init__(self, device, use_xla=False):
        if use_xla:
            try:
                import torch_xla.core.xla_model as xm
            except ImportError:
                raise RuntimeError(f'`use_xla was set, but couldn\'t import torch_xla')
        self.device = device
        self.use_xla = use_xla
        self.check_sig(self._make, ('state',))
        self.check_sig(self._get_state, ())

    def check_sig(self, func, expected_args):
        # print(f'checking signature of {func} from self={type(self)}')
        args = list(signature(func).parameters.keys())
        if any(l != r for l,r in zip(args, expected_args)):
            raise TypeError(
                f'Can\'t instantiate State object.  {func.__name__} method must have '
                f'signature {func.__name__}({", ".join(expected_args)}).  '
                f'Found \'{", ".join(args)}\'')

    @abstractmethod
    def _make(self, state=None):
        """
        If `state` is provided, we are restoring from a checkpoint
        Private helper method for the public methods init and load.
        """
    
    @abstractmethod
    def _get_state(self):
        """
        Returns `state` to be used later with the load method
        """

    def get_params(self):
        return self.params

    def init(self, params):
        """
        Instantiate all class members, storing each instance as a named member
        """
        self.params = params
        return self._make()

    def load(self, path, **param_overrides):
        """
        Instantiate all target classes from information in the file at `path` 
        """
        ckpt = torch.load(path)
        self.params = hparams.Hyperparams(ckpt['params'])
        self.params.update(param_overrides)
        return self._make(ckpt['state'])

    def save(self, path):
        """
        Save everything to path
        """
        state = self._get_state()
        ckpt = dict(state=state, params=self.params)
        if self.use_xla:
            xm.save(ckpt, path)
            xm.master_print(f'Saved checkpoint {path} using xm.save')
        else:
            torch.save(ckpt, path)
            print(f'Saved checkpoint {path}')

