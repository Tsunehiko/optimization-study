import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from net import Net
from optimization import *
from util import smooth_curve

from torch.utils.tensorboard import SummaryWriter

log_dir = "logs/compare"

#データの作成
data = np.array([[1.20,0],[1.25,0],[1.30,0],[1.35,0],[1.40,1],[1.45,0],[1.50,1],[1.55,0],[1.60,1],[1.65,1],[1.70,1],[1.75,1]])
x_train = np.ones(data.shape)
x_train[:,0] = data[:,0]
t_train = data[:,1]

#各種設定
max_iterations = 10000
train_size = x_train.shape[0]
batch_size = 3
iter_per_epoch = max(train_size / batch_size, 1)
log_interval = 100
TH = 1.0e-11 #収束判定の閾値

optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = Net()
    train_loss[key] = []

for key in optimizers.keys():
    print("----------" + key + "---------")
    writer = SummaryWriter(log_dir=log_dir+"/"+key)
    for i in range(max_iterations):
        train_index = np.random.permutation(train_size)
        for batch in range(0,train_size,batch_size):
            batch_index = train_index[batch:batch+batch_size]
            x_batch = x_train[batch_index]
            t_batch = t_train[batch_index]
        
            grads = networks[key].grad(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)
            
            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        writer.add_scalar(key,train_loss[key][-1],i)
            
        if i % log_interval == 0 and i != 0:
            print("iteration " + str(i) + ":" + str(train_loss[key][-1]))

    writer.close()

markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations * iter_per_epoch)
for key in optimizers.keys():
    print(len(train_loss[key]))
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
