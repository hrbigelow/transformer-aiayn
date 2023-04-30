from typing import Dict
from abc import ABC, abstractmethod
from inspect import signature
from torch import nn


class State(ABC):
    """
    Instances of this class should provide the following semantics:
    It can be instantiated as one of:
        Derived(params)  - de-novo, before any saved state exists
        Derived(params, saved_state)  - resuming from a saved state

    params are parameters that remain constant throughout the life of the instance.
    Multiple different classes derived from State should all expect the same instance
    of params, as it represents a single source of truth to begin a run.

    A call to self.state() must return the entire non-constant state of the
    object which is needed to restore it.  The return value of self.state() will
    be used as saved_state in a subsequent re-instantiation.

    In particular, any dependence on random number generation (from numpy, python,
    torch cpu or torch cuda) implies that the rng state should be returned by
    self.state() and resumed during construction.

    The best practice is for instances of State to own the random number generators
    they use, and always use them explicitly whenever randomness is needed.
    See recommendation from:
    https://numpy.org/doc/stable/reference/random/index.html#quick-start 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Override this to call super().__init__() appropriately with an init method
        with signature:  (params, saved_state)
        params: hyperparmeters defining the base instance
        saved_state: if present, object state was saved and is being restored
        """
        self.check_sig(self.__init__, ('params', 'saved_state'))
        self.check_sig(self.state, ())
        
    @staticmethod
    def check_sig(func, expected_args):
        args = list(signature(func).parameters.keys())
        if args != list(expected_args):
            raise TypeError(
                f'Can\'t instantiate State object.  {func.__name__} method must have '
                f'signature {func.__name__}({", ".join(expected_args)}).'
                f'Found \'{", ".join(args)}\'')

    @abstractmethod
    def state(self):
        """
        Return the current state of the instance, suitable for saving and restoring.
        This is the state used in the constructor
        """
        pass

"""
# inherit State as first subclass
# __init__ and state must have signatures as shown:
class Derived(State, nn.Module):
    def __init__(self, params=None, saved_state=None):
        super().__init__()

    def state(self):
        pass
"""

class Run:
    def __init__(self, **kwargs):
        """
        kwargs are name: class pairs.  Each class must be derived from `State`
        If add_dep is to be called, dep must come before name in the order given
        here.
        """
        self._params = None
        for name, cls in kwargs:
            if not issubclass(cls, State):
                raise TypeError(f'\'{name}\' was not a subclass of `State`')
        self.classes = OrderedDict(kwargs)
        self.deps = {}

    def init(self, params):
        """
        Instantiate all class members, storing each instance as a named member
        """
        self._params = params
        for name, cls in self.classes:
            deps = tuple(self.__dict__[d] for d in self.deps.get(name, ()))
            instance = cls(self._params, {}, *deps)
            self.__dict__[name] = instance

    def add_deps(self, name, *deps):
        """
        Declare `name` to have dependencies in its construction.  `name` and all 
        `deps` must be names given in construction of Run.  `deps` must occur
        before `name` in this order.  

        The `deps` instances constructed will be used as arguments to `name`
        """
        missing = next((n for n in (name, *deps) if n not in self.classes), None)
        if missing:
            raise RuntimeError(
                f'Argument value \'{missing}\' is not found in the list of registered '
                f'classes: {"\, ".join(self.classes.keys())}')
        cl = list(self.classes.keys())
        ni = cl.index(name)
        out_of_order = next((dep for dep in deps if clist.index(dep) >= ni), None)
        if out_of_order:
            raise RuntimeError(
                f'deps contained \'{out_of_order}\' which occurs after \'{name}\''
                f'in the class list order')
        self.deps[name] = deps

    def load(self, path):
        """
        Load everything from the checkpoint path given 
        """
        state = torch.load(path)
        self._params = state['_params']
        for name, cls in self.classes:
            instance = cls(self._params, state[name])
            self.__dict__[name] = cls

    def save(self, path_template, step):
        """
        Save everything to the path
        """
        state = dict(_params=self._params) 
        for name in self.classes:
            state[name] = self.__dict__[name].state()

        path = path_template.format(step)
        t.save(path, state)

