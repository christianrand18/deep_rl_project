from copy import deepcopy

import numpy as np
import torch.nn as nn

def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])

def nestdictsum(dict):
    return sum([sum([dict[i][t] for t in dict[i]]) for i in dict])