#import matplotlib in MAC-OS

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

#Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized

#method1:

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#method2:

conda install nomkl 

#(I use this method and it worked)