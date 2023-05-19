from abc import ABC, abstractmethod
from inspect import signature
from collections import OrderedDict

class Pause(ABC):
    """
    """
    def __init__(self, use_xla=False):
        if use_xla:
            try:
                import torch_xla.core.xla_model as xm
            except ImportError:
                raise RuntimeError(f'`use_xla was set, but couldn\'t import torch_xla')
        self.use_xla = use_xla
        self.check_sig(self._make, ('params', 'state'))
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
    def _make(self, params, state=None):
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
        return self._make(params)

    def load(self, path, **param_overrides):
        """
        Instantiate all target classes from information in the file at `path` 
        """
        ckpt = torch.load(path)
        ckpt['params'].update(param_overrides)
        self.params = ckpt['params']
        return self._make(ckpt['params'], ckpt['state'])

    def save(self, path):
        """
        Save everything to path
        """
        state = self._get_state()
        ckpt = dict(state=state, params=self.params)
        if self.use_xla:
            xm.save(ckpt, path)
        else:
            torch.save(ckpt, path)

