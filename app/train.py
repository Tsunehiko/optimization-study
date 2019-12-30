import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from net import Net
from optimization import *

data = np.array([[1.20,0],[1.25,0],[1.30,0],[1.35,0],[1.40,1],[1.45,0],[1.50,1],[1.55,0],[1.60,1],[1.65,1],[1.70,1],[1.75,1]])
x_train = np.ones(data.shape)
x_train[:,0] = data[:,0]
t_train = data[:,1]

network = Net()
optimizer = SGD()

TH = 1.0e-11 #収束判定の閾値

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 3
learning_rate = 0.1
former_loss = 1.0
iter_per_epoch = max(train_size / batch_size, 100)

train_loss_list = []

nt = 100
t = np.linspace(1.1, 1.8, nt) #xの値
xx = np.ones((nt,2))
xx[:,0] = t

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    grads = network.grad(x_batch, t_batch)
    
    # パラメータの更新
    optimizer.update(network.params, grads)
    
    loss = network.loss(x_batch, t_batch)
    if (loss - former_loss) ** 2 < TH:
        print("誤差は規定値まで減少")
        break
    former_loss = loss

    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        zz = network.predict(xx)
        plt.plot(t,zz,color='y')
        print(i,network.params['W'],loss)
        
zz = network.predict(xx)
plt.plot(t,zz,color="b",linewidth=2)
plt.ylim(-0.2,1.2)
plt.hlines([0,1],1.1,1.8,linestyles="dashed")
plt.plot(data[:,0],data[:,1],"ro",markersize=10)
plt.show()

'''
# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
'''