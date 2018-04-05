# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
#ori = conv1[0].reshape((-1))
#plt.plot()

from math import frexp,ldexp, copysign
from sys import float_info


def quant(x):
    f,e = frexp(x)
    f=(f//0.125)*0.125
    return ldexp(f,e)

fun = np.vectorize(quant)
#plt.plot(ori,'.')
#plt.plot(qnt,'.')
#plt.plot(ori-qnt)

conv1=[fun(i) for i in conv1]
conv10=[fun(i) for i in conv10]
fire2squeeze1x1 = [ fun(i) for i in fire2squeeze1x1]
fire2expand1x1 = [ fun(i) for i in fire2expand1x1]
fire2expand3x3 = [ fun(i) for i in fire2expand3x3]
fire3squeeze1x1 = [ fun(i) for i in fire3squeeze1x1]
fire3expand1x1 = [ fun(i) for i in fire3expand1x1]
fire3expand3x3 = [ fun(i) for i in fire3expand3x3]
fire4squeeze1x1 = [ fun(i) for i in fire4squeeze1x1]
fire4expand1x1 = [ fun(i) for i in fire4expand1x1]
fire4expand3x3 = [ fun(i) for i in fire4expand3x3]
fire5squeeze1x1 = [ fun(i) for i in fire5squeeze1x1]
fire5expand1x1 = [ fun(i) for i in fire5expand1x1]
fire5expand3x3 = [ fun(i) for i in fire5expand3x3]
fire6squeeze1x1 = [ fun(i) for i in fire6squeeze1x1]
fire6expand1x1 = [ fun(i) for i in fire6expand1x1]
fire6expand3x3 = [ fun(i) for i in fire6expand3x3]
fire7squeeze1x1 = [ fun(i) for i in fire7squeeze1x1]
fire7expand1x1 = [ fun(i) for i in fire7expand1x1]
fire7expand3x3 = [ fun(i) for i in fire7expand3x3]
fire8squeeze1x1 = [ fun(i) for i in fire8squeeze1x1]
fire8expand1x1 = [ fun(i) for i in fire8expand1x1]
fire8expand3x3 = [ fun(i) for i in fire8expand3x3]
fire9squeeze1x1 = [ fun(i) for i in fire9squeeze1x1]
fire9expand1x1 = [ fun(i) for i in fire9expand1x1]
fire9expand3x3 = [ fun(i) for i in fire9expand3x3]


dic = {'fire2squeeze1x1' : fire2squeeze1x1,
        'fire2expand1x1' : fire2expand1x1,
        'fire2expand3x3' : fire2expand3x3,
        'fire3squeeze1x1' : fire3squeeze1x1,
        'fire3expand1x1' : fire3expand1x1,
        'fire3expand3x3' : fire3expand3x3,
        'fire4squeeze1x1' : fire4squeeze1x1,
        'fire4expand1x1' : fire4expand1x1,
        'fire4expand3x3' : fire4expand3x3,
        'fire5squeeze1x1' : fire5squeeze1x1,
        'fire5expand1x1' : fire5expand1x1,
        'fire5expand3x3' : fire5expand3x3,
        'fire6squeeze1x1' : fire6squeeze1x1,
        'fire6expand1x1' : fire6expand1x1,
        'fire6expand3x3' : fire6expand3x3,
        'fire7squeeze1x1' : fire7squeeze1x1,
        'fire7expand1x1' : fire7expand1x1,
        'fire7expand3x3' : fire7expand3x3,
        'fire8squeeze1x1' : fire8squeeze1x1,
        'fire8expand1x1' : fire8expand1x1,
        'fire8expand3x3' : fire8expand3x3,
        'fire9squeeze1x1' : fire9squeeze1x1,
        'fire9expand1x1' : fire9expand1x1,
        'fire9expand3x3' : fire9expand3x3,
        'conv1':conv1,
        'conv10':conv10,
        'globals__':globals__,
        'version__':version__
       }

scipy.io.savemat('squ_quant.mat', dic)

#for i in range(2,10):
#    print("fire"+str(i)+"squeeze1x1 = [ fun(i) for i in fire"+str(i)+"squeeze1x1]")
#    print("fire"+str(i)+"expand1x1 = [ fun(i) for i in fire"+str(i)+"expand1x1]")
#    print("fire"+str(i)+"expand3x3 = [ fun(i) for i in fire"+str(i)+"expand3x3]")

for i in range(2,10):
    print("'fire"+str(i)+"squeeze1x1' : fire"+str(i)+"squeeze1x1,")
    print("'fire"+str(i)+"expand1x1' : fire"+str(i)+"expand1x1,")
    print("'fire"+str(i)+"expand3x3' : fire"+str(i)+"expand3x3,")

