import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

def sigm(z):
    return 1/(1+np.exp(-z))

def verify():
    TH = 1.0e-11
    p = 0.9
    e = 0.001

    nt = 100
    t = np.linspace(1.1, 1.8, nt)
    xx = np.ones((nt,2))
    xx[:,0] = t

    data = np.array([[1.20,0],[1.25,0],[1.30,0],[1.35,0],[1.40,1],[1.45,0],[1.50,1],[1.55,0],[1.60,1],[1.65,1],[1.70,1],[1.75,1]])
    N, nd = data.shape

    X = np.ones((N,nd))
    X[:,0] = data[:,0]
    true = data[:,1]
    w = np.array([1.0,0.0])
    dw = np.array([0.0,0.0])
    alpha = np.array([0,0])
    g = np.array([0.0,0.0])
    err0 = 1.0
    for i in range(100000):
        for j in range(N):
            zj = sigm(X[j,:].dot(w))
            dw = -((true[j]-zj)*zj*(1-zj))*(X[j,:])
            g = p*g+(1-p)*dw*dw
            alpha = 0.1/np.sqrt(g+e)
            w = w - alpha*dw
        z = sigm(X.dot(w))
        err = np.sqrt((true-z).dot((true-z)))
        if (err-err0)**2<TH:
            break
        err0 = err
        if i % 100 == 0:
            zz = sigm(xx.dot(w))
            plt.plot(t,zz,color="y")
            print(i,w,err)

    zz = sigm(xx.dot(w))
    plt.plot(t,zz,color="b",linewidth=2)
    plt.ylim(-0.2,1.2)
    plt.hlines([0,1],1.1,1.8,linestyles="dashed")
    plt.plot(data[:,0],data[:,1],"ro",markersize=10)
    plt.show()

if __name__ == "__main__":
    verify()