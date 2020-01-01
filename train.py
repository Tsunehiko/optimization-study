import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from net import Net
from optimization import *

from torch.utils.tensorboard import SummaryWriter

data = np.array([[1.20,0],[1.25,0],[1.30,0],[1.35,0],[1.40,1],[1.45,0],[1.50,1],[1.55,0],[1.60,1],[1.65,1],[1.70,1],[1.75,1]])
x_train = np.ones(data.shape)
x_train[:,0] = data[:,0]
t_train = data[:,1]

network = Net()
optimizer_dict = {}
optimizer_dict["SGD"] = SGD(lr=0.1)
optimizer_dict["Momentum"] = Momentum(lr=0.01,momentum=0.9)
optimizer_dict["Nesterov"] = Nesterov(lr=0.01,momentum=0.9)
optimizer_dict["AdaGrad"] = AdaGrad(lr=1.5)
optimizer_dict["RMSprop"] = RMSprop(lr=0.01,decay_rate=0.99)
optimizer_dict["Adam"] = Adam(lr=0.01,beta1=0.9,beta2=0.999)

key = "AdaGrad" #使用する最適化手法を選択する
optimizer = optimizer_dict[key]
writer = SummaryWriter(log_dir="logs/train/"+key+"1.5")

TH = 1.0e-12 #収束判定の閾値

iters_num = 100000
train_size = x_train.shape[0]
batch_size = 3
former_loss = 1.0
#iter_per_epoch = max(train_size / batch_size, 1)
log_interval = 100

nt = 100
t = np.linspace(1.1, 1.8, nt) #xの値
xx = np.ones((nt,2))
xx[:,0] = t

for i in range(iters_num):
    train_index = np.random.permutation(train_size)
    for batch in range(0,train_size,batch_size):
        batch_index = train_index[batch:batch+batch_size]
        x_batch = x_train[batch_index]
        t_batch = t_train[batch_index]
    
        grads = network.grad(x_batch, t_batch)

        optimizer.update(network.params, grads)
        
    loss = network.loss(x_batch, t_batch)
    if (loss - former_loss) ** 2 < TH:
        print("誤差は規定値まで減少")
        break
    former_loss = loss

    writer.add_scalar("SGD",loss,i)
        
    if i % log_interval == 0 and i != 0:
        zz = network.predict(xx)
        plt.plot(t,zz,color='y')
        print(i,network.params['W'],loss)
    
writer.close()
        
zz = network.predict(xx)
plt.plot(t,zz,color="b",linewidth=2)
plt.ylim(-0.2,1.2)
plt.hlines([0,1],1.1,1.8,linestyles="dashed")
plt.plot(data[:,0],data[:,1],"ro",markersize=10)
plt.savefig("figure/"+key+"_train.png")