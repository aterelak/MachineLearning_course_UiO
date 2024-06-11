import Cost as cost
import Grad as grads
import Activator as activator
import numpy as np
import autograd.numpy as npA
from autograd import elementwise_grad, grad
import operator
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

class LogisticReg:
    def __init__(self, eta, epochs, lmbd) -> None:
        self.eta = eta
        self.epochs = epochs
        self.lmbd = lmbd
    
    def Initialize(self, X, yTarget, seed = 2906, M = 2):
        self.X = X
        self.yTarget = yTarget
        self.M = M
        np.random.seed(seed)
        self.n_inputs, self.n_features  = X.shape
        self.n_categ = 1
        self.out_weights = np.random.randn(self.n_features)
        self.out_bias = np.zeros(self.n_categ) + 0.01

    def Train(self):
        self.update()
        #self.Score()

    def update(self):
        m = self.X.shape[1]
        base_grad = grads.BaseGd(self.eta)
        delta = 1e-8

        for _ in range(self.epochs):
            Gwo = 0.0
            Gbo = 0.0
            for _ in range(m):
                random_index = self.M*np.random.randint(m)
                xi = self.X[random_index:random_index+self.M]
                yi = self.yTarget[random_index:random_index+self.M]
                
                a_o = self.Predict(xi)

                error_output = a_o - yi
                dWo = np.matmul(xi.T, error_output.T)
                dBo = np.sum(error_output, axis=0)
                dWo += self.lmbd * self.out_weights
                Gwo += dWo*dWo
                Gbo += dBo*dBo
                self.out_weights -= base_grad.update(dWo) / (delta+np.sqrt(Gwo))
                self.out_bias -= base_grad.update(dBo) / (delta+np.sqrt(Gbo))

    def Predict(self, xi):
        sigmoid = activator.sigmoid
        z_o = np.matmul(xi, self.out_weights) + self.out_bias
        a_o = sigmoid(z_o)
        return np.where(a_o > 0.5, 1, 0)
    
    def Score(self, X,yTarget):
        return accuracy_score(self.Predict(X), yTarget)
    

class NeuralNetwork:
    def __init__(self, eta, epochs, lmbd) -> None:
        self.eta = eta
        self.epochs = epochs
        self.lmbd = lmbd

    def Initialize(self, X, yTarget, n_categ, n_hidden_neurons, seed = 2906, M = 2,):
        self.X = X
        self.yTarget = yTarget
        self.M = M
        np.random.seed(seed)
        self.n_inputs, self.n_features  = self.X.shape
        self.n_h_neurons = n_hidden_neurons
        self.n_categ = n_categ
        self.hidden_weights = np.random.randn(self.n_features , self.n_h_neurons)
        self.hidden_bias = np.zeros(self.n_h_neurons) + 0.01
        self.out_weights = np.random.randn(self.n_h_neurons, self.n_categ)
        self.out_bias = np.zeros(self.n_categ) + 0.01

    def Train(self):        
        base_grad = grads.BaseGd(self.eta)
        m = self.X.shape[1]
        delta = 1e-8
        #print("Accuracy without training: " + str(accuracy_score(Predict(X),yTarget)))
        #print("R2 without training: " + str(r2_score(Predict(X),yTarget)))

        for _ in range(self.epochs):
            Gwh = 0.0
            Gbh = 0.0
            Gwo = 0.0
            Gbo = 0.0
            for _ in range(m):
                random_index = self.M*np.random.randint(m)
                xi = self.X[random_index:random_index+self.M]
                yi = self.yTarget[random_index:random_index+self.M]
                dWo, dBo, dWh, dBh = self.BackProp(xi, yi)
                dWo += self.lmbd * self.out_weights
                dWh += self.lmbd * self.hidden_weights
                Gwh += dWh*dWh
                Gbh += dBh*dBh
                Gwo += dWo*dWo
                Gbo += dBo*dBo
                self.out_weights -= base_grad.update(dWo) / (delta+np.sqrt(Gwo))
                self.out_bias -= base_grad.update(dBo) / (delta+np.sqrt(Gbo))
                self.hidden_weights -= base_grad.update(dWh) / (delta+np.sqrt(Gwh))
                self.hidden_bias -= base_grad.update(dBh) / (delta+np.sqrt(Gbh))

        #print("Accuracy after training: " + str(accuracy_score(Predict(X),yTarget)))
        #print("R2 without training: " + str(r2_score(Predict(X),yTarget)))

    def FeedForward(self, X):
        sigmoid = activator.sigmoid
        z_h = np.matmul(X,self.hidden_weights) + self.hidden_bias
        
        a_h = sigmoid(z_h)
        
        z_o = np.matmul(a_h, self.out_weights) + self.out_bias
        
        prob = sigmoid(z_o)
        #prob = z_o
        return prob, a_h

    def BackProp(self, xi, yi):
        #crossDer = grad(CrossEntropy,0)
        #sigmoidDer =  elementwise_grad(sigmoid,0)
        
        prob, a_h = self.FeedForward(xi)
        #error_output = crossDer(prob.T, Y) * sigmoidDer(prob).T @ a_h
        error_output = prob.T - yi
        error_hidden = np.matmul(error_output.T, self.out_weights.T) * a_h * (1 - a_h)
        
        output_weights_gradient = np.matmul(a_h.T, error_output.T)
        output_bias_gradient = np.sum(error_output, axis=1)

        hidden_weights_gradient = np.matmul(xi.T, error_hidden)
        hidden_bias_gradient = np.sum(error_hidden, axis=0)

        return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

    def Predict(self, X):
        prob, _ = self.FeedForward(X)
        return [1 if x > 0.5 else 0 for x in prob]
    
    def Score(self,X,yTarget):
        return accuracy_score(self.Predict(X),yTarget)



    
