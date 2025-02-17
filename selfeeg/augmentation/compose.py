from __future__ import annotations

import copy
import inspect
import random
from typing import Any, Dict

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "CircularAug",
    "DynamicSingleAug",
    "RandomAug",
    "SequentialAug",
    "StaticSingleAug",
]


class StaticSingleAug:
    """
    Single static augmentation with preset arguments.

    ``StaticSingleAug`` performs a single data augmentation
    where the optional arguments are previously set and given during initialization.
    No random choice of the arguments is performed. The class accepts multiple set of
    optional arguments. In this case they are called individually at each class
    call, in a circular manner. This means that the first call uses the first set of
    arguments, the second will use the second set of arguments, and so on.
    When the last set of arguments is used, the class will restart from the
    first set of arguments.

    To perform an augmentation, simply call the instantiated class
    (see provided example or check the introductory notebook).

    Parameters
    ----------
    augmentation: function
        The augmentation function to apply. It can be a custom function,
        but the first argument must be the element to augment.
    arguments: list, dict, list[list or dict], optional
        The set of arguments to pass to the augmentation function. It can be:

            1. None. In this case the default parameters of the function are used.

            2. list. In this case the function is called with the sintax
            ``augmentation(x, *arguments)``

            3. dict. In this case the function is called with the sintax
            ``augmentation(x, **arguments)``

            4. a list of dicts or lists. This is a particular case where multiple
            combinations of arguments are given. Each element of the list must be a
            list or a dict with the specific argument combination.
            Every time ``perform_augmentation`` is called, one of the given
            combinations is used to perform the data augmentation. The list
            is followed sequentially with repetition, meaning that the first call
            uses the first set of arguments of the list, the second call uses the
            second set of arguments, and so on. When the last element of the list
            is used, the function will restart scrolling the given list.

        Default = None

    Methods
    -------
    perform_augmentation(X: ArrayLike)
        Apply the augmentation with the given arguments.
        __call__() will call this method.

    Example
    -------
    >>> import selfeeg.augmentation as aug
    >>> import torch
    >>> BatchEEG = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))
    >>> Fs = 64
    >>> Aug_eye = aug.StaticSingleAug(
    ...     aug.add_eeg_artifact,
    ...     [{'Fs': Fs, 'artifact': 'eye', 'amplitude': 0.5, 'batch_equal': False},
    ...     {'Fs': Fs, 'artifact': 'eye', 'amplitude': 1.0, 'batch_equal': False}]
    ... )
    >>> BatchEEGaug1 = Aug_eye(BatchEEG)
    >>> BatchEEGaug2 = Aug_eye(BatchEEG)

    plot the augmentations (require matplotlib to be installed)

    >>> import matplotlib.pyplot as plt
    >>> plt.style.use('seaborn-v0_8-white')
    >>> plt.rcParams['figure.figsize'] = (15.0, 6.0)
    >>> plt.plot(BatchEEG[0,0],linewidth=2.5)
    >>> plt.plot(BatchEEGaug1[0,0])
    >>> plt.plot(BatchEEGaug2[0,0])
    >>> plt.tick_params(axis='both', which='major', labelsize=12)
    >>> plt.title('Example of addition of EEG eye blink artifact', fontsize=15)
    >>> plt.legend(['original sample', 'augmented sample amp 0.5',
    ...             'augmented sample amp 1'])
    >>> plt.show()

    """

    def __init__(
        self, augmentation: "function", arguments: list or dict or list[list or dict] = None
    ):

        if not (inspect.isfunction(augmentation) or inspect.isbuiltin(augmentation)):
            raise ValueError("augmentation must be a function to call")
        else:
            self.augmentation = augmentation

        self.arguments = arguments
        self.counter = 0
        self.maxcounter = 0
        self.multipleStaticArguments = False
        if arguments != None:
            if all(isinstance(i, list) or isinstance(i, dict) for i in arguments):
                self.multipleStaticArguments = True
                self.maxcounter = len(arguments)

    def perform_augmentation(self, X: ArrayLike) -> ArrayLike:

        if self.multipleStaticArguments:
            argument = self.arguments[self.counter]
            if isinstance(argument, list):
                Xaug = self.augmentation(X, *argument)
            else:
                Xaug = self.augmentation(X, **argument)

            self.counter += 1
            if self.counter == self.maxcounter:
                self.counter = 0
        else:
            if self.arguments == None:
                Xaug = self.augmentation(X)
            elif isinstance(self.arguments, list):
                Xaug = self.augmentation(X, *self.arguments)
            else:
                Xaug = self.augmentation(X, **self.arguments)

        return Xaug

    def __call__(self, X):
        return self.perform_augmentation(X)


class DynamicSingleAug:
    """
    Single augmentation with randomly chosen arguments.

    ``DynamicSingleAug`` performs a single data augmentation
    where the optional arguments are chosen at random from a given discrete or
    continuous range of values. Random choice of the arguments is performed
    at each call.

    To perform an augmentation, simply call the instantiated class
    (see provided example or check the introductory notebook)

    Parameters
    ----------
    augmentation: function
        The augmentation function to apply. It can be a custom function,
        but the first argument must be the element to augment.
    discrete_arg: dict, optional
        A dictionary specifying arguments whose value must be chosen
        within a discrete set. The dict must have:

            - keys as string with the name of one of the optional arguments
            - values as lists of elements to be randomly chosen.
              Single elements are allowed if a specific value for an argument
              needs to be set. In this case it is not mandatory to give it as list,
              as automatic conversion will be performed internally. In other words,
              a key-value pair given as ``{"arg": value}`` is allowed, since the
              conversion to ``{"arg": [value]}`` is automatically performed.

        Default = None
    range_arg: dict, optional
        A dictionary specifying arguments whose value must be chosen within
        a continuous range.
        The dict must have:

            - keys as string with the name of one of the optional arguments
            - values as two element lists specifying the range of values
              where to randomly select the argument value.

        Default = None
    range_type: dict or list, optional
        A dictionary or a list specifying if values in range_arg must be given
        to the augmentation function as integers. If given as a dict, keys must
        be the same as the one of range_arg argument. If given as a list, the
        length must be the same of range_arg.
        In particular:

            1. if range_type is a **dict**:

                - keys must be those in range_arg
                - values must be single element specifying if the argument must be
                  an integer. In this case, use a **boolean True** or a **string
                  'int'** to specify if the argument must be converted to an integer.

            2. if range_arg is a **list**:

                - values must be set as the values in the dict. The order is the
                  one used when iterating along the range_arg dict.

            3. if **None** is given, a list of True with length equal to range_arg
               is automatically created, since int arguments are more compatible
               compared to float ones.

        Default = None

    Methods
    -------
    perform_augmentation(x: ArrayLike)
        Apply the augmentation with the given arguments.
        __call__() will call this method.

    Note
    ----
    At least one of **discrete_arg** or **range_arg** arguments must be given,
    the class simply suggests to use ``StaticSingleAug``.

    Example
    -------
    >>> import selfeeg.augmentation as aug
    >>> import torch
    >>> BatchEEG = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))
    >>> Aug_warp = aug.DynamicSingleAug(
    ...     aug.warp_signal,
    ...     discrete_arg = {'batch_equal': [True, False]},
    ...     range_arg= {'segments': [5,15], 'stretch_strength': [1.5,2.5],
    ...                 'squeeze_strength': [0.4,2/3]},
    ...     range_type={'segments': True, 'stretch_strength': False,
    ...                 'squeeze_strength': False}
    ... )
    >>> BatchEEGaug1 = Aug_warp(BatchEEG)
    >>> BatchEEGaug2 = Aug_warp(BatchEEG)

    plot the augmentations (require matplotlib to be installed)

    >>> import matplotlib.pyplot as plt
    >>> plt.style.use('seaborn-v0_8-white')
    >>> plt.rcParams['figure.figsize'] = (15.0, 6.0)
    >>> plt.plot(BatchEEG[0,0],linewidth=2.5)
    >>> plt.plot(BatchEEGaug1[0,0])
    >>> plt.plot(BatchEEGaug2[0,0])
    >>> plt.tick_params(axis='both', which='major', labelsize=12)
    >>> plt.title('Example of dynamic aug with warp augmentation', fontsize=15)
    >>> plt.legend(['original sample', 'augmented sample 1', 'augmented sample 2'])
    >>> plt.show()

    """

    def __init__(
        self,
        augmentation,
        discrete_arg: Dict[str, Any] = None,
        range_arg: Dict[str, list[int or float, int or float]] = None,
        range_type: Dict[str, str or bool] or list[str or bool] = None,
    ):

        # set augmentation function
        if not (inspect.isfunction(augmentation) or inspect.isbuiltin(augmentation)):
            raise ValueError("augmentation must be a function to call")
        else:
            self.augmentation = augmentation

        # get function argument name
        self.argnames = inspect.getfullargspec(augmentation)[0][1:]

        # check if given discrete_arg keys are actually augmentation arguments
        self.discrete_arg = None
        if discrete_arg != None:
            if isinstance(discrete_arg, dict):
                if all(i in self.argnames for i in discrete_arg):
                    self.discrete_arg = discrete_arg
                else:
                    raise ValueError(
                        "keys of discrete_arg argument must be the "
                        "argument of the augmentation fun"
                    )
            else:
                raise ValueError("discrete_arg must be a dictionary")

        # check if given range_arg keys are actually augmentation arguments
        # also check if values are two element list
        self.range_arg = None
        if range_arg != None:
            if isinstance(range_arg, dict):
                if all(i in self.argnames for i in range_arg):
                    if all((isinstance(i, list) and len(i) == 2) for i in range_arg.values()):
                        self.range_arg = range_arg
                    else:
                        raise ValueError(
                            "range_arg values must be a length 2 " "list with min and max range"
                        )
                else:
                    raise ValueError(
                        "keys of range_arg argument must be the "
                        "argument of the augmentation function."
                    )
                for i in range_arg:
                    if not (isinstance(range_arg[i], list)):
                        range_arg[i] = [range_arg[i]]
            else:
                raise ValueError("range_arg must be a dictionary")

        # check if range_types keys are the same as range_args
        self.range_type = None
        if range_type != None:
            if isinstance(range_type, dict):
                if range_type.keys() == range_arg.keys():
                    self.range_type = range_type
                else:
                    raise ValueError("keys of range_type must be the same as range_arg")
            elif isinstance(range_type, list):
                if len(range_type) == len(self.range_arg):
                    self.range_type = range_type
                else:
                    raise ValueError("range_type must have the same length as range_args")
            else:
                raise ValueError("discrete_arg must be a dictionary or a list")
        else:
            if self.range_arg != None:
                self.range_type = [True] * len(self.range_arg)

        self.is_range_type_dict = True if isinstance(self.range_type, dict) else False

        self.given_arg = list(self.discrete_arg) if self.discrete_arg != None else []
        self.given_arg += list(self.range_arg) if self.range_arg != None else []

    def perform_augmentation(self, X: ArrayLike) -> ArrayLike:
        arguments = {i: None for i in self.given_arg}
        if self.discrete_arg != None:
            for i in self.discrete_arg:
                if isinstance(self.discrete_arg[i], list):
                    arguments[i] = random.choice(self.discrete_arg[i])  # nosec
                else:
                    arguments[i] = self.discrete_arg[i]

        cnt = 0  # counter if range_type is a list, it's a sort of enumerate
        if self.range_arg != None:
            for i in self.range_arg.keys():
                arguments[i] = random.uniform(self.range_arg[i][0], self.range_arg[i][1])  # nosec
                if self.is_range_type_dict:
                    if self.range_type[i] in ["int", True]:
                        arguments[i] = int(arguments[i])
                else:
                    if self.range_type[cnt] in ["int", True]:
                        arguments[i] = int(arguments[i])
                    cnt += 1

        Xaug = self.augmentation(X, **arguments)
        return Xaug

    def __call__(self, X):
        return self.perform_augmentation(X)


class SequentialAug:
    """
    Multiple augmentations applied sequentially.

    ``SequentialAug`` applies a sequence of augmentations in a specified order.
    No random choice between the given list of augmentation is performed, just
    pure call of all the augmentations in the specified order.

    To perform an augmentation, simply call the instantiated class
    (see provided example or check the introductory notebook)

    Parameters
    ----------
    *augmentations: "callable objects"
        The sequence of augmentations to apply at each call.
        It can be any callable object, but the first argument to pass must be
        the element to augment. It is suggested to give a sequence of
        ``StaticSingleAug`` or ``DynamicSingleAug`` instantiations.

    Note
    ----
    If you provide an augmentation implemented outside of this this library, be
    sure that the function will return a single output with the element to pass
    to the next augmentation function of the list.

    Note
    ----
    The function will automatically handle RandomAug instances with return_index
    set to True. In this case, an internal deepcopy with return_index set to false
    will be automatically created.

    Methods
    -------
    perform_augmentation(X: ArrayLike)
        Apply the augmentations with the given arguments and specified order.
        __call__() will call this method.


    Example
    -------
    >>> import selfeeg.augmentation as aug
    >>> import torch
    >>> BatchEEG = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))
    >>> Aug_eye = aug.StaticSingleAug(aug.add_eeg_artifact,{'Fs': 64, 'artifact': 'eye', 'amplitude': 0.5})
    >>> Aug_warp = aug.DynamicSingleAug(
    ...     aug.warp_signal,
    ...     discrete_arg = {'batch_equal': [True, False]},
    ...     range_arg= {'segments': [5,15], 'stretch_strength': [1.5,1.8],
    ...                 'squeeze_strength': [0.5,2/3]},
    ...     range_type={'segments': True, 'stretch_strength': False,
    ...                 'squeeze_strength': False}
    ... )
    >>> Sequence1= aug.SequentialAug(Aug_eye, Aug_warp)
    >>> BatchEEGaug1 = Sequence1(BatchEEG)
    >>> BatchEEGaug2 = Sequence1(BatchEEG)

    plot the augmentations (require matplotlib to be installed)

    >>> import matplotlib.pyplot as plt
    >>> plt.style.use('seaborn-v0_8-white')
    >>> plt.rcParams['figure.figsize'] = (15.0, 6.0)
    >>> plt.plot(BatchEEG[0,0],linewidth=2.5)
    >>> plt.plot(BatchEEGaug1[0,0])
    >>> plt.plot(BatchEEGaug2[0,0])
    >>> plt.tick_params(axis='both', which='major', labelsize=12)
    >>> plt.title(
    ...     'Sequential Aug with eye blink artifact and warp augmentation',
    ...     fontsize=15)
    >>> plt.legend(['original sample', 'augmented sample 1', 'augmented sample 2'])
    >>> plt.show()

    """

    def __init__(self, *augmentations):
        self.augs = [item for item in augmentations]
        self._search_for_random_aug_with_index()

    def perform_augmentation(self, X: ArrayLike) -> ArrayLike:
        Xaugs = self.augs[0](X)
        for i in range(1, len(self.augs)):
            Xaugs = self.augs[i](Xaugs)
        return Xaugs

    def __call__(self, X):
        return self.perform_augmentation(X)

    def _search_for_random_aug_with_index(self):
        for idx, item in enumerate(self.augs):
            if isinstance(item, RandomAug):
                if self.augs[idx].return_index == True:
                    self.augs[idx] = copy.deepcopy(item)
                    self.augs[idx].return_index = False
            elif isinstance(item, SequentialAug):
                item._search_for_random_aug_with_index()


class CircularAug:
    """
    Single Augmenter called sequentially from a list, following a circular order.

    ``CircularAug`` calls an Augmenter from a given sequence following the  order.
    Augmenters are called circularly. This means that the first call uses the first
    Augmenter from the input list, the second call will use the second, and so on.
    When the last Augmenter is called, the class will restart from the
    first one.

    To perform an augmentation, simply call the instantiated class
    (see provided example or check the introductory notebook)

    Parameters
    ----------
    *augmentations: "callable objects"
        The list of augmentations to apply at each call.
        It can be any callable object, but the first argument to pass must be
        the element to augment. It is suggested to give a sequence of
        ``StaticSingleAug`` or ``DynamicSingleAug`` instantiations.

    Note
    ----
    The function will automatically handle RandomAug instances with return_index
    set to True. In this case, an internal deepcopy with return_index set to false
    will be automatically created.

    Methods
    -------
    perform_augmentation(X: ArrayLike)
        Apply the augmentations with the given arguments and specified order.
        __call__() will call this method.


    Example
    -------
    >>> import selfeeg.augmentation as aug
    >>> import torch
    >>> BatchEEG = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))
    >>> Aug_eye = aug.StaticSingleAug(
    ...     aug.add_eeg_artifact,{'Fs': 64, 'artifact': 'eye', 'amplitude': 0.5})
    >>> Circular = aug.CircularAug(Aug_eye, aug.identity)
    >>> EEGeye = Circular(BatchEEG)
    >>> EEGid  = Circular(BatchEEG)

    """

    def __init__(self, *augmentations):
        self.augs = [item for item in augmentations]
        self._augcnt = 0
        self._augnumber = len(self.augs)
        self._search_for_random_aug_with_index()

    def perform_augmentation(self, X: ArrayLike) -> ArrayLike:
        Xaugs = self.augs[self._augcnt](X)
        self._update_counter()
        return Xaugs

    def __call__(self, X):
        return self.perform_augmentation(X)

    def _update_counter(self):
        self._augcnt += 1
        if self._augcnt == self._augnumber:
            self._augcnt = 0

    def _search_for_random_aug_with_index(self):
        for idx, item in enumerate(self.augs):
            if isinstance(item, RandomAug):
                if self.augs[idx].return_index == True:
                    self.augs[idx] = copy.deepcopy(item)
                    self.augs[idx].return_index = False
            elif isinstance(item, SequentialAug):
                item._search_for_random_aug_with_index()


class RandomAug:
    """
    Random augmentation chosen from a given set.

    ``RandomAug`` applies an augmentations selected randomly from a given set.

    To perform an augmentation, simply call the instantiated class
    (see provided example or check the introductory notebook)

    Parameters
    ----------
    *augmentations: "callable objects"
        The set of augmentations to randomly choose at each call.
        It can be any callable object, but the first arguments to pass must
        be the ArrayLike object to augment. It is suggested to give a set of
        ``StaticSingleAug`` or ``DynamicSingleAug`` instantiations.
    p: 1D ArrayLike, optional
        A 1D array or list with the weights associated to each augmentation
        (higher the weight, higher the frequency of choosing an augmentation
        of the list). Elements of p must be in the same order as the given
        augmentations.
        If given, p will be scaled so to have sum 1 (so you can give any value).
        If not given, all augmentations will be chosen with equal probability.

        Default = None
    return_index: bool, optional
        Whether to return an index identifying the selected augmentation or not.
        The index is simply the output of random.choice function used to select
        the augmentation from the given list. Indeces follow the augmentation order
        given during class instantiation.

        Default = False

    Methods
    -------
    perform_augmentation(X: ArrayLike)
        Apply a random augmentation from the given list of augmenters.
        __call__() will call this method.

    Example
    -------
    >>> import selfeeg.augmentation as aug
    >>> import numpy as np
    >>> import torch
    >>> BatchEEG = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))
    >>> Aug_eye = aug.StaticSingleAug(
    ...     aug.add_eeg_artifact,{'Fs': Fs, 'artifact': 'eye', 'amplitude': 0.5})
    >>> Aug_warp = aug.DynamicSingleAug(
    ...     aug.warp_signal,
    ...     discrete_arg = {'batch_equal': [True, False]},
    ...     range_arg= {'segments': [5,15], 'stretch_strength': [1.5,1.8],
    ...                 'squeeze_strength': [0.5,2/3]},
    ...     range_type={'segments': True, 'stretch_strength': False,
    ...                 'squeeze_strength': False}
    ... )
    >>> Sequence2= aug.RandomAug(Aug_eye, Aug_warp, p=[0.7, 0.3])
    >>> BatchEEGaug1 = Sequence2(BatchEEG)
    >>> BatchEEGaug2 = Sequence2(BatchEEG)

    plot the augmentations (require matplotlib to be installed)

    >>> # simulate 10000 augmentations calls this line is used in RandomAug
    >>> # to choose the index of the list of augmentations to call.
    >>> # Note that the size argument has been added to make computation faster, in
    >>> # the class only 1 value is returned
    >>> import matplotlib.pyplot as plt
    >>> idx=np.random.choice(Sequence2.nprange_, size= 1000, p=Sequence2.p)
    >>> counts=[(1000-len(np.nonzero(idx)[0]))/1000, len(np.nonzero(idx)[0])/1000]
    >>> plt.subplot(1,3,(1,2))
    >>> plt.plot(BatchEEG[0,0],linewidth=2.5)
    >>> plt.plot(BatchEEGaug1[0,0])
    >>> plt.plot(BatchEEGaug2[0,0])
    >>> plt.tick_params(axis='both', which='major', labelsize=12)
    >>> plt.title(
    ...     'Example of Random aug between eye blinking artifact and warp',
    ...     fontsize=15)
    >>> plt.legend(
    ...     ['original sample', 'augmented sample 1', 'augmented sample 2'],
    ...     loc='upper left')
    >>> plt.subplot(1,3,3)
    >>> plt.bar(['eye blinking', 'warp'],counts)
    >>> plt.tick_params(axis='both', which='major', labelsize=12)
    >>> plt.title('barplot of chosen augmentations', fontsize=15)
    >>> plt.ylabel('probability',fontsize=12)
    >>> plt.show()

    """

    def __init__(self, *augmentations, p=None, return_index=False):

        self.augs = [item for item in augmentations]
        self.N = len(self.augs)
        self.p = p
        self.return_index = return_index
        if p is not None:
            if len(p) != self.N:
                raise ValueError("length of p does not match the number of augmentations")
            self.p = np.array(p) + 0.0
            self.p /= np.sum(p)
        self.nprange_ = np.arange(0, self.N)

    def perform_augmentation(self, X):
        if self.p is None:
            idx = random.randint(0, self.N - 1)  # nosec
        else:
            idx = np.random.choice(self.nprange_, p=self.p)
        Xaugs = self.augs[idx](X)
        if self.return_index:
            return Xaugs, idx
        else:
            return Xaugs

    def __call__(self, X):
        return self.perform_augmentation(X)
