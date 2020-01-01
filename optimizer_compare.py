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
# TODO:MNISTのデータを使えるようにする
a = np.linspace(1.1,1.5,64)
b = np.linspace(1.5,1.9,64)
a1 = np.zeros((64,2))
b1 = np.ones((64,2))
a1[:,0] = a
b1[:,0] = b
data = np.concatenate([a1,b1])
x_train = np.ones(data.shape)
x_train[:,0] = data[:,0]
t_train = data[:,1]

#各種設定
epochs = 10000
train_size = x_train.shape[0]
batch_size = 16
iter_per_epoch = max(train_size // batch_size, 1)
log_interval = 100
TH = 1.0e-11 #収束判定の閾値
former_loss = 10

optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['Nesterov'] = Nesterov()
optimizers['AdaGrad'] = AdaGrad()
optimizers['RMSprop'] = RMSprop()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = Net()
    train_loss[key] = []

for key in optimizers.keys():
    print("----------" + key + "---------")
    writer = SummaryWriter(log_dir=log_dir+"/"+key)
    for i in range(epochs):
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

x = np.arange(epochs * iter_per_epoch)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
