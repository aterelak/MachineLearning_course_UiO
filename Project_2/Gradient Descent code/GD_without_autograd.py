from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt

def learning_schedule(t, t0, t1):
        return t0/(t+t1)


class GD_without_autograd:
    def own_inversion(X):
        X_TX = X.T @ X

        linreg_theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        print("our own inversion:")
        print(linreg_theta)
        H = (2/n) * X_TX
        EigVal, EigVect = np.linalg.eig(H)
        return linreg_theta

    def Base_GD(X, y, eta, Niterations, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        n = X.shape[0]
        
        # Base Gradient Descent
        for interation in range(Niterations):
            if lmbd == 0:
                grads = 2.0/n * X.T @ ((X @ theta)-y)
            else:
                grads = 2.0/n * X.T @ ((X @ theta)-y)+2 * lmbd * theta
            theta -= eta*grads

        return theta

    # Gradient Descent with Momentum

    def Momentum_GD(X, y, eta, Niterations, momentum, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        n = X.shape[0]

        change = 0.0
        for iteration in range(Niterations):
            if lmbd == 0:
                grads = 2.0/n * X.T @ ((X @ theta)-y)
            else:
                grads = 2.0/n * X.T @ ((X @ theta)-y)+2 * lmbd * theta
            change = eta * grads + momentum * change
            theta -= change
        return theta

    # Normal Stochastic Gradient Descent

    def SGD(X, y, eta, epochs, minibatch_size, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        minibatch_num = int(X.shape[0]/minibatch_size)
        t0,t1 = 5,50

        for epoch in range(epochs):
            for batch_n in range(minibatch_num):
                randInd = minibatch_size*np.random.randint(minibatch_num)
                xi = X[randInd:randInd+minibatch_size]
                yi = y[randInd:randInd+minibatch_size]
                if lmbd == 0:
                    grads = 2.0/minibatch_size * xi.T @ ((xi @ theta)-yi)
                else:
                    grads = 2.0/minibatch_size * xi.T @ ((xi @ theta)-yi)+2 * lmbd * theta
                eta = learning_schedule(epoch*minibatch_size+batch_n, t0, t1)
                theta -= eta*grads

        return theta


    # Momentum Stochastic GD

    def Momentum_SGD(X, y, eta, epochs, minibatch_size, momentum, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        minibatch_num = int(X.shape[0]/minibatch_size)
        t0,t1 = 5,50
        change = 0.0

        for epoch in range(epochs):
            for batch_n in range(minibatch_num):
                randInd = minibatch_size*np.random.randint(minibatch_num)
                xi = X[randInd:randInd+minibatch_size]
                yi = y[randInd:randInd+minibatch_size]
                if lmbd == 0:
                    grads = 2.0/minibatch_size * xi.T @ ((xi @ theta)-yi)
                else:
                    grads = 2.0/minibatch_size * xi.T @ ((xi @ theta)-yi)+2 * lmbd * theta
                eta = learning_schedule(epoch*minibatch_size+batch_n, t0, t1)
                change = eta*grads + momentum * change
                theta -= change

        return theta


    def RMS_prop(X, y, eta, M, n_epochs, lmbd = 0):
        # RMS prop
        # Define parameters for Stochastic Gradient Descent
        m = int(X.shape[0]/M) #number of minibatches
        # Guess for unknown parameters theta
        theta = np.random.randn(X.shape[1],1)
        # Value for learning rate

        rho = 0.99
        # Including AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8
        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                if lmbd == 0:
                    gradients = 2.0/M * xi.T @ ((xi @ theta)-yi)
                else:
                    gradients = 2.0/M * xi.T @ ((xi @ theta)-yi)+2 * lmbd * theta
                Giter += (rho*Giter+(1-rho)*gradients*gradients)
                update = gradients*eta/(delta+np.sqrt(Giter))
                theta -= update
        return theta

    def ADAM(X, y, M, n_epochs, eta, beta1, beta2, lmbd = 0):
        # ADAM
        # Define parameters for Stochastic Gradient Descent
        m = int(X.shape[0]/M) #number of minibatches
        # Guess for unknown parameters theta
        theta = np.random.randn(3,1)
        # Value for learning rate
        # Including AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8
        iter = 0
        for epoch in range(n_epochs):
            mom_1 = 0.0
            mom_2 = 0.0
            iter += 1
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                if lmbd == 0:
                    gradients = 2.0/M * xi.T @ ((xi @ theta)-yi)
                else:
                    gradients = 2.0/M * xi.T @ ((xi @ theta)-yi)+2 * lmbd * theta
                mom_1 = beta1*mom_1 + (1-beta1)*gradients
                mom_2 = beta2*mom_2 + (1-beta2)*gradients*gradients
                term_1 = mom_1/(1.0-beta1**iter)
                term_2 = mom_2/(1.0-beta2**iter)
                update = eta*term_1/(np.sqrt(term_2)+delta)
                theta -= update
        return theta



if __name__ == "__main__":
    n=100
    x = 2*np.random.rand(n,1)
    y = 2+(4*x)+(5*(x**2))
    np.random.seed(2906)
    X = np.c_[np.ones((n,1)),x, x**2]