from random import random, seed
import numpy as np
import autograd.numpy as npA
import matplotlib.pyplot as plt
from autograd import grad, elementwise_grad

def CostOLS(beta, X, y, n):
    return (1.0/n)*npA.sum((y-X @ beta)**2)

def CostOLS_Ridge(beta, X, y, n, lmbd):
    return (1.0/n)*npA.sum((y-X @ beta)**2)+2 * lmbd * beta

def NewCostOLS(y,X,theta):
    return npA.sum((y-X @ theta)**2)

def NewCostOLS_Ridge(y,X,theta, lmbd):
    return npA.sum((y-X @ theta)**2)+2*lmbd*theta

class GD_with_autograd:
    def own_inversion(X):
        X_TX = X.T @ X

        linreg_theta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        print("our own inversion:")
        print(linreg_theta)

        # PLAIN GD

        H = (2/n) * X_TX
        EigVal, EigVect = np.linalg.eig(H)
        return linreg_theta

    def GD(X, y, eta, Niterations, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        n = X.shape[0]
        if lmbd == 0:
            training_gradient = grad(CostOLS)
        else:
            training_gradient = elementwise_grad(CostOLS_Ridge)

        # Base Gradient Descent
        for interation in range(Niterations):
            if lmbd == 0:
                grads = training_gradient(theta, X, y, n)
            else:
                grads = training_gradient(theta, X, y, n, lmbd)
            theta -= eta*grads


        return theta


    # GD with AdaGrad, plain GD without momentum

    def AdaGD(X, y, eta, Niterations, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        n = X.shape[0]
        if lmbd == 0:
            training_gradient = grad(CostOLS)
        else:
            training_gradient = elementwise_grad(CostOLS_Ridge)
        Giter = 0.0
        delta = 1e-8
        # Base Gradient Descent
        for interation in range(Niterations):
            if lmbd == 0:
                grads = training_gradient(theta, X, y, n)
            else:
                grads = training_gradient(theta, X, y, n, lmbd)
            Giter += grads*grads
            update = grads*eta/(delta+np.sqrt(Giter))
            theta -= update

        return theta

    # GD with AdaGrad, plain GD with momentum

    def AdaGD_momentum(X, y, eta, Niterations, momentum, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        n = X.shape[0]
        if lmbd == 0:
            training_gradient = grad(CostOLS)
        else:
            training_gradient = elementwise_grad(CostOLS_Ridge)
        Giter = 0
        delta = 1e-8
        update = 0
        # Base Gradient Descent
        for interation in range(Niterations):
            if lmbd == 0:
                grads = training_gradient(theta, X, y, n)
            else:
                grads = training_gradient(theta, X, y, n, lmbd)
            Giter += grads*grads
            new_update = grads*eta/(delta+np.sqrt(Giter)) + momentum * update
            theta -= new_update
            update = new_update

        return theta



    # SGD with AdaGrad, without momentum

    def AdaSGD(X, y, n_epochs, eta, M, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        if lmbd == 0:
            training_gradient = grad(NewCostOLS,2)
        else:
            training_gradient = elementwise_grad(NewCostOLS_Ridge,2)
        
        # Define parameters for Stochastic Gradient Descent
        m = int(X.shape[0]/M) #number of minibatches
        # Guess for unknown parameters theta
        # Value for learning rate
        # Including AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8
        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                if lmbd == 0:
                    gradients = (1.0/M)*training_gradient(yi, xi, theta)
                else:
                    gradients = (1.0/M)*training_gradient(yi, xi, theta, lmbd)
                Giter += gradients*gradients
                update = gradients*eta/(delta+np.sqrt(Giter))
                theta -= update
        return theta



    def AdaSGD_momentum(X, y , n_epochs, eta, M, momentum, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        if lmbd == 0:
            training_gradient = grad(NewCostOLS,2)
        else:
            training_gradient = elementwise_grad(NewCostOLS_Ridge,2)
        # SGD with AdaGrad, with momentum
        # Define parameters for Stochastic Gradient Descent
        m = int(X.shape[0]/M) #number of minibatches
        # Guess for unknown parameters theta
        # Value for learning rate
        # Including AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8
        old_update = 0
        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                if lmbd == 0:
                    gradients = (1.0/M)*training_gradient(yi, xi, theta)
                else:
                    gradients = (1.0/M)*training_gradient(yi, xi, theta, lmbd)
                Giter += gradients*gradients
                update = gradients*eta/(delta+np.sqrt(Giter)) + momentum * old_update
                theta -= update
                old_update = update
        return theta

    def RMS_prop(X,y, n_epochs, M, eta, rho, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        if lmbd == 0:
            training_gradient = grad(NewCostOLS,2)
        else:
            training_gradient = elementwise_grad(NewCostOLS_Ridge,2)
        # RMS prop
        # Define parameters for Stochastic Gradient Descent
        m = int(X.shape[0]/M) #number of minibatches
        # Guess for unknown parameters theta
        # Value for learning rate

        # Including AdaGrad parameter to avoid possible division by zero
        delta  = 1e-8
        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                random_index = M*np.random.randint(m)
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                if lmbd == 0:
                    gradients = (1.0/M)*training_gradient(yi, xi, theta)
                else:
                    gradients = (1.0/M)*training_gradient(yi, xi, theta, lmbd)
                Giter += (rho*Giter+(1-rho)*gradients*gradients)
                update = gradients*eta/(delta+np.sqrt(Giter))
                theta -= update
        return theta

    def ADAM(X, y, n_epochs, M, eta, beta1, beta2, lmbd = 0):
        theta = np.random.randn(X.shape[1],1)
        if lmbd == 0:
            training_gradient = grad(NewCostOLS,2)
        else:
            training_gradient = elementwise_grad(NewCostOLS_Ridge,2)
        # ADAM
        # Define parameters for Stochastic Gradient Descent
        m = int(X.shape[0]/M) #number of minibatches
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
                    gradients = (1.0/M)*training_gradient(yi, xi, theta)
                else:
                    gradients = (1.0/M)*training_gradient(yi, xi, theta, lmbd)
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


    X = np.c_[np.ones((n,1)),x, x**2]