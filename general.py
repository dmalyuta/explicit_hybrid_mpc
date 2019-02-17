"""
General data, classes, etc.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2018 University of Washington. All rights reserved.
"""

import progressbar

INFO  = ">>> "
WARN  = "??? "
ERROR = "!!! "

class ControlError(Exception):
    """
    Custom exception of this code.
    
    Parameters
    ----------
    message : str
        Descriptive test.
    """
    def __init__(self,message,*args,**kwargs):
        Exception.__init__(self,ERROR+message,*args,**kwargs)

def info(message):
    """
    Print an info message to prompt.
    
    Parameters
    ----------
    message : str
        Message to print.
    """
    print(INFO+message)
    
def warning(message):
    """
    Print a warning message to prompt.
    
    Parameters
    ----------
    message : str
        Message to print.
    """
    print(WARN+message)

def makeWidgets(description):
    """
    Make widgets list for progressbar.
    
    Parameters
    ----------
    description : str
        Goes in front, e.g. <descrption> ...
  
    Returns
    -------
    widgets : list
        Widgets for progressbar.
    """
    widgets=['<%s> '%(description),progressbar.Percentage(),' ',
             progressbar.Bar(),' (',progressbar.ETA(),')']
    return widgets

def fullrange(a,b=None):
    """
    Return an integer range iterator, which includes the final point
    and which starts at 1 by default.
    
    Parameters
    ----------
    a : int
        Interval start. If b is not specified, interval start is
        taken to be 1 and a is used as the interval end.
    b : int, optional
        Interval end.
        
    Returns
    -------
    : list
        Integer range {a,a+1,a+2,...,b-1,b}.
    """
    if b is None:
        return range(1,a+1)
    else:
        return range(a,b+1)
