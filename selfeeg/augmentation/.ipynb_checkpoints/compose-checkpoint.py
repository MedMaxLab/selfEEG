import inspect
import random
from typing import Any, Dict

__all__ = ['StaticSingleAug', 'DynamicSingleAug', 'SequentialAug', 'RandomAug']
class StaticSingleAug():
    """
    StaticSingleAug is a class designed to perform a single data augmentation where the optional arguments
    are previously set and given during initialization. No random choice of the arguments is performed. The 
    class accepts multiple set of optional arguments. In this case they are called individually at each class
    call, in a sequential manner. This means that the first call use the first set of arguments, the second 
    will use the second set of arguments, and so on. When the last set is used, the class will restart from the
    first set of arguments.

    Parameters
    ----------
    
    augmentation: function
        The augmentation function to apply. It can be a custom function, but the first 
        argument must be the element to augment
    arguments: list, dict, list[list or dict], optional
        The arguments to give to the augmentation function. It can be:
            1) None. In this case the default parameters of the function are used. Remember that if there are
               other non optional arguments, the call will throw an error.
            2) a simple list. In this case the function is called with the sintax "augmentation(x, *arguments)"
            3) a simple dict. In this case the function is called with the sintax "augmentation(x, **arguments)"
            4) a list of dict or list. This is a particular case where multiple combination of arguments are given.
               Each element of the list must be a list or a dict with the specific argument combination. Every time
               the class is called, one of the given combination is used to perform the data augmentation. The list
               is followed sequentially with repetion, meaning that the first call use the first set of arguments of the 
               list, the second the second set of arguments, and so on. When the last element of the list is used, the 
               function will restart scrolling the given list
        Default: None  
    """
    
    def __init__(self, augmentation, arguments: list or dict or list[list or dict]=None):
        
        if not(inspect.isfunction(augmentation) or inspect.isbuiltin(augmentation)):
            raise ValueError('augmentation must be a function to call')
        else:
            self.augmentation=augmentation
        
        self.arguments=arguments
        self.counter=0
        self.maxcounter=0
        self.multipleStaticArguments=False
        if arguments !=None:
            if all(isinstance(i,list) or isinstance(i,dict) for i in arguments):
                self.multipleStaticArguments=True
                self.maxcounter=len(arguments)
        
    def PerformAugmentation(self, X):
        
        if self.multipleStaticArguments:
            argument=self.arguments[self.counter]
            if isinstance(argument, list):
                Xaug = self.augmentation(X, *argument)
            else:
                Xaug = self.augmentation(X, **argument)
            
            self.counter +=1
            if self.counter == self.maxcounter:
                self.counter=0 
        else:
            if self.arguments==None:
                Xaug= self.augmentation(X)
            elif isinstance(self.arguments, list):
                Xaug = self.augmentation(X, *self.arguments)
            else:
                Xaug = self.augmentation(X, **self.arguments)
        
        return Xaug
        
    def __call__(self, X):
        return self.PerformAugmentation(X)
        
        
        
class DynamicSingleAug():
    """
    DynamicSingleAug is a class designed to perform a single data augmentation where the optional arguments
    are chosen at random from a given discrete set or a given range. Random choice of the arguments is performed
    at each call.

    At least one of discrete_arg or range_arg arguments must be given, otherwise simply use a StaticSingleAug
        
    Parameters
    ----------
    
    augmentation: function
        The augmentation function to apply. It can be a custom function, but the first 
        argument must be the element to augment
    discrete_arg: dict, optional
        A dictionary specifying arguments whose value must be chosen within a discrete set. The dict must have:
            - Keys as string with the name of one of the optional arguments
            - Values as lists of elements to be randomly chosen. Single elements are accepted if a specific 
              value for an argument needs to be set. In this case it's not mandatory to give it as list, as 
              automatic conversion will be performed.
        Default: None
    range_arg: dict, optional
        A dictionary specifying arguments whose value must be chosen within a continuous range. The dict must have:
            - Keys as string with the name of one of the optional arguments
            - Values as two element lists specifying the range of values where to randomly select the argument value. 
        Default: None
    range_type: dict or list, optional
        A dictionary or a list specifying if values in range_arg must be given to the augmentation function as integers:
        If given as a dict, keys must be the same as the one of range_arg (no more or less, the same). If given as a list,
        the length must be the same of range_arg. In particular:
            1) if range_type is a dict. 
                - Keys must be the ones in range_arg
                - Values must be single element specifying if the argument must be an integer. In this case, use a boolean
                  True or a string 'int' to specify if the argument must be converted to an int.
            2) if range_arg is a list.
                - Values must be set as the values in the dict. The order is the one used when iterating along the
                  range_arg dict
            3) if None is given, a list of True with length equal to range_arg is automatically created, since int
               arguments are more compatible with float ones.
        Default: None  
    """
    def __init__(self, 
                 augmentation, 
                 discrete_arg: Dict[str, Any]=None, 
                 range_arg: Dict[str, list[ int or float, int or float]]=None,
                 range_type: Dict[str, str or bool] or list[str or bool]=None
                ):
        
        # set augmentation function
        if not(inspect.isfunction(augmentation) or inspect.isbuiltin(augmentation)):
            raise ValueError('augmentation must be a function to call')
        else:
            self.augmentation=augmentation
        
        # get function argument name
        self.argnames= inspect.getfullargspec(augmentation)[0][1:]
        
        # check if given discrete_arg keys are actually augmentation arguments
        self.discrete_arg=None
        if discrete_arg != None:
            if isinstance(discrete_arg, dict):
                if all(i in self.argnames for i in discrete_arg):
                    self.discrete_arg=discrete_arg
                else:
                    raise ValueError('keys of discrete_arg argument must be the argument of the augmentation fun')
            else:
                raise ValueError('discrete_arg must be a dictionary')
        
        # check if given range_arg keys are actually augmentation arguments 
        # also check if values are two element list
        self.range_arg=None
        if range_arg != None:
            if isinstance(range_arg, dict):
                if all(i in self.argnames for i in range_arg):
                    if all( (isinstance(i,list) and len(i)==2) for i in range_arg.values()):
                        self.range_arg=range_arg
                    else:
                        raise ValueError('range_arg values must be a len 2 list with min and max range')
                else:
                    raise ValueError('keys of range_arg argument must be the argument of the augmentation fun')
                for i in range_arg:
                    if not(isinstance(range_arg[i],list)):
                        range_arg[i] = [range_arg[i]]
            else:
                raise ValueError('range_arg must be a dictionary')
        
        # check if range_types keys are the same as range_args
        self.range_type=None
        if range_type!=None:
            if isinstance(range_type, dict):
                if range_type.keys() == range_arg.keys():
                    self.range_type=range_type
                else:
                    raise ValueError('keys of range_type must be the same as range_arg')
            elif isinstance(range_type, list):
                if len(range_type)==len(self.range_arg):
                    self.range_type=range_type
                else:
                    raise ValueError('range_type must have the same length as range_args')
            else:
                raise ValueError('discrete_arg must be a dictionary or a list')
        else:
            if self.range_arg!=None:
                self.range_type=[True]*len(self.range_arg)
        
        self.is_range_type_dict= True if isinstance(self.range_type, dict) else False
        
        self.given_arg = list(self.discrete_arg) if self.discrete_arg!=None else []
        self.given_arg += list(self.range_arg) if self.range_arg!=None else []
        
    
    def PerformAugmentation(self, X):    
        arguments={i:None for i in self.given_arg}
        if self.discrete_arg!=None:
            for i in self.discrete_arg:
                if isinstance(self.discrete_arg[i],list): 
                    arguments[i] = random.choice(self.discrete_arg[i]) 
                else:
                    arguments[i]= self.discrete_arg[i] 
        
        cnt=0 # counter if range_type is a list, it's a sort of enumerate
        if self.range_arg!=None:
            for i in self.range_arg.keys():
                arguments[i]=random.uniform(self.range_arg[i][0], self.range_arg[i][1])
                if self.is_range_type_dict:
                    if self.range_type[i] in ['int', True]:
                        arguments[i] = int(arguments[i])
                else:
                    if self.range_type[cnt] in ['int', True]:
                        arguments[i] = int(arguments[i])
                    cnt+=1
        
        Xaug = self.augmentation(X, **arguments)
        return Xaug
        
    def __call__(self, X):
        return self.PerformAugmentation(X)


    
class SequentialAug():
    
    def __init__(self,*augmentations):

        self.augs=[item for item in augmentations]
     
    def PerformAugmentation(self, X): 
        
        Xaugs = self.augs[0](X)
        for i in range(1,len(self.augs)):
            Xaugs = self.augs[i](Xaugs)
        return Xaugs
            
    def __call__(self, X):
        return self.PerformAugmentation(X)

class RandomAug():
    """
    RandomAug perform a random augmentation from a list of arguments.
    Class must be initialized giving a sequence 
    """
    def __init__(self,*augmentations):
        
        self.augs = [item for item in augmentations]
        self.N = len(self.augs)
     
    def PerformAugmentation(self, X): 

        idx=random.randint(0,self.N-1)
        Xaugs = self.augs[idx](X)
        return Xaugs
            
    def __call__(self, X):
        return self.PerformAugmentation(X)