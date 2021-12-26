from sklearn.datasets import make_circles
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import time


n = 500
p = 2

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.1375)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], color='skyblue')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='salmon')
plt.axis('equal')
plt.show()

Y = Y[:, np.newaxis]

class Layer:
    def __init__(self, input_size, output_size, bias=True, activation=None):
        self.use_bias    = bias
        self.activation  = activation
        self.input_size  = input_size
        self.output_size = output_size
        
        self.weigths = np.random.rand(input_size, output_size) * 2 - 1
        self.bias    = np.random.rand(1, self.output_size) * 2 - 1

class Activation(object):
    @staticmethod
    def sigmoid(x, derivate=False):
        if derivate:
            return (x * (1 - x))
        else:
            return (1 / (1 + (np.e ** (-x))))
		
class Loss(object):
    @staticmethod
    def MSE(fake, real, derivate=False):
        if derivate:
            return (fake - real)
        else:
            return np.mean((fake - real) ** 2)
		
plt.plot(np.linspace(-5, 5, 100), Activation.sigmoid(np.linspace(-5, 5, 100)))
plt.show()

plt.plot(np.linspace(-5, 5, 100), Activation.sigmoid(np.linspace(-5, 5, 100), derivate=True))
plt.show()



def createNN(topology, activation, use_bias=True):
    nn = list()
    for l, layer in enumerate(topology[:-1]):
        nn.append(Layer(topology[l], topology[l+1], bias=use_bias, activation=activation))
        
    return nn


def train(nn, x, y, loss_fn=Loss.MSE, lr=0.001, training=True):
    # Forward
    out = [(None, x)]
    
    for l, layer in enumerate(nn):
        z = out[-1][1] @ nn[l].weigths
        if nn[l].use_bias:
            z = z + nn[l].bias

        a = z
        if nn[l].activation:
            a = nn[l].activation(a)
            
        out.append(tuple((z, a)))
        
    loss = loss_fn(out[-1][1], Y)
    
    if training:
        # Backward
        deltas = list()
        
        for l in reversed(range(0, len(nn))):
            
            z = out[l+1][0]
            a = out[l+1][1]
            
            if l == (len(nn) - 1): # Last layer
                deltas.insert(0, (loss_fn(a, Y, derivate=True) * nn[l].activation(a, derivate=True)))
            else:
                deltas.insert(0, (deltas[0] @ weigths.T * nn[l].activation(a, derivate=True)))
                
            weigths = nn[l].weigths
                
            if nn[l].use_bias:
                nn[l].bias = nn[l].bias - np.mean(deltas[0], axis=0, keepdims=True) * lr
                
            nn[l].weigths = nn[l].weigths - out[l][1].T @ deltas[0] * lr
                
    return out[-1][1]



loss = list()

nn = createNN([p, 4, 8, 1], Activation.sigmoid, use_bias=False)

for i in range(10000):
    pred = train(nn, X, Y, lr=0.005)
    
    if i % 50 == 0:
        loss.append(Loss.MSE(pred, Y))
        
        _x0 = np.linspace(-1.5, 1.5, 50)
        _x1 = np.linspace(-1.5, 1.5, 50)
        _y  = np.zeros((50, 50))
        
        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _y[i0, i1] = train(nn, np.array([[x0, x1]]), Y, training=False)[0][0]
                
        plt.pcolormesh(_x0, _x1, _y, cmap='coolwarm')
        plt.axis('equal')
        
        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], color='skyblue')
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], color='salmon')
        
        clear_output(wait=True)
        plt.show()
        
        plt.plot(range(len(loss)), loss)
        plt.show()
        
        time.sleep(0.5)
