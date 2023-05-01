from typing import Dict
from abc import ABC, abstractmethod
from inspect import signature
from collections import OrderedDict
import torch
from torch import nn


class State(ABC):
    """
    https://numpy.org/doc/stable/reference/random/index.html#quick-start 
    """
    def __init__(self):
        self.obj = None
        self.check_sig(self.make, ('params', 'saved_state'))
        self.check_sig(self.state, ())
        
    def check_sig(self, func, expected_args):
        # print(f'checking signature of {func} from self={type(self)}')
        args = list(signature(func).parameters.keys())
        if any(l != r for l,r in zip(args, expected_args)):
            raise TypeError(
                f'Can\'t instantiate State object.  {func.__name__} method must have '
                f'signature {func}({", ".join(expected_args)}).'
                f'Found \'{", ".join(args)}\'')

    @abstractmethod
    def make(self):
        """
        Configure this instance such that get() returns a populated object.
        The derived class may act as a container class for a third-party object.  If
        so, make() will construct that and store it as self.obj.
        Alternatively, the derived class may itself provide the functionality needed.
        If so, make() will populate fields on the derived class, and get() will
        return self.
        """
        pass

    def get(self):
        # only override if not a container class
        return self.obj

    def to(self, device):
        pass

    @abstractmethod
    def state(self):
        """
        Return the current state of the object instance, suitable for saving and restoring.
        """
        pass

class Run:
    def __init__(self, **kwargs):
        """
        kwargs are name: class pairs.  Each class must be derived from `State`
        If add_dep is to be called, dep must come before name in the order given
        here.
        """
        self._params = None
        for name, make in kwargs.items():
            if not isinstance(make, State):
                raise TypeError(f'\'{name}\' was not an instance of `State`')
        self.objs = OrderedDict(kwargs)
        self.deps = {}

    def _make(self, params, state=None):
        # instantiate all objects from params and state (if present)
        self._params = params
        for name, obj in self.objs.items():
            deps = tuple(self.__dict__[d] for d in self.deps.get(name, ()))
            obj.make(self._params, state[name], *deps)
            self.__dict__[name] = obj.get()

    def init(self, params):
        """
        Instantiate all class members, storing each instance as a named member
        """
        return self._make(params)

    def load(self, path):
        """
        Load everything from the checkpoint path given 
        """
        state = torch.load(path)
        return self._make(state['_params'], state)

    def to(self, device):
        for obj in self.objs.values():
            obj.to(device)

    def save(self, path):
        """
        Save everything to the path
        """
        state = dict(_params=self._params) 
        for name in self.objs:
            state[name] = self.objs[name].state()
        torch.save(state, path)

    def add_deps(self, name, *deps):
        """
        Declare `name` to have dependencies in its construction.  `name` and all 
        `deps` must be names given in construction of Run.  `deps` must occur
        before `name` in this order.  

        The `deps` instances constructed will be used as arguments to `name`
        """
        missing = next((n for n in (name, *deps) if n not in self.objs), None)
        if missing:
            raise RuntimeError(
                f'Argument value \'{missing}\' is not found in the list of registered '
                f'objects: {", ".join(self.objs.keys())}')
        clist = list(self.objs.keys())
        ni = clist.index(name)
        out_of_order = next((dep for dep in deps if clist.index(dep) >= ni), None)
        if out_of_order:
            raise RuntimeError(
                f'deps contained \'{out_of_order}\' which occurs after \'{name}\''
                f'in the class list order')
        self.deps[name] = deps

