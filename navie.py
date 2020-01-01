# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimization import *


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0*y

def loss(x,y):
    return 0.5 * (x**2 + y**2)

TH = 0.001

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

optimizer_dict = {}
optimizer_dict["SGD"] = SGD(lr=0.95)
optimizer_dict["Momentum"] = Momentum(lr=0.01,momentum=0.9)
optimizer_dict["Nesterov"] = Nesterov(lr=0.01,momentum=0.9)
optimizer_dict["AdaGrad"] = AdaGrad(lr=0.01)
optimizer_dict["RMSprop"] = RMSprop(lr=0.01,decay_rate=0.99)
optimizer_dict["Adam"] = Adam(lr=0.001,beta1=0.9,beta2=0.999)

key = "Adam" #使用する最適化手法を選択する
optimizers = OrderedDict()
optimizers[key] = optimizer_dict[key]

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(10000):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
        if loss(params['x'],params['y']) < TH:
            break

    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("figure/"+key+"_navie.png")