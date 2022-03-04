import numpy as np
import math
import torch
# from colorama import init
from termcolor import colored


def printColoredError( diff, tol=1.e-5, accTol=1.e-2 ):
    if( diff < tol):
        print(colored(diff, 'green'))
    elif( diff < accTol ):
        print(colored(diff, 'yellow'))
    else:
        print(colored(diff, 'red'))
    