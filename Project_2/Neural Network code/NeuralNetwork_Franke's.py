import Cost as cost
import Grad as grads
import Activator as activator
import numpy as np
import autograd.numpy as npA
from autograd import elementwise_grad, grad
import operator
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+npA.exp(-x))

def CrossEntropy(X, target):
    return -(1.0 / target.size) * npA.sum(target * npA.log(X + 10e-10))

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4



def Train(X, yTarget, eta, epochs, M, lmbd):
    #Set up dimensions of the NN
    n_inputs, n_features  = X.shape
    n_features = 2
    n_h_neurons = 2
    n_categ = 1


    global hidden_weights
    hidden_weights = np.random.randn(n_features , n_h_neurons)
    global hidden_bias 
    hidden_bias = np.zeros(n_h_neurons) + 0.01

    global out_weights 
    out_weights = np.random.randn(n_h_neurons, n_categ)
    global out_bias 
    out_bias = np.zeros(n_categ) + 0.01
    
    m = X.shape[1]
    base_grad = grads.BaseGd(eta)
    delta = 1e-8

    #print("MSE without training: " + str(mean_squared_error(Predict(X),yTarget)))
    #print("R2 without training: " + str(r2_score(Predict(X),yTarget)))

    for _ in range(epochs):
        Gwh = 0.0
        Gbh = 0.0
        Gwo = 0.0
        Gbo = 0.0
        for _ in range(m):
            random_index = M*np.random.randint(m)
            xi = X[random_index:random_index+M]
            yi = yTarget[random_index:random_index+M]
            dWo, dBo, dWh, dBh = BackProp(xi, yi)
            dWo += lmbd * out_weights
            dWh += lmbd * hidden_weights
            Gwh += dWh*dWh
            Gbh += dBh*dBh
            Gwo += dWo*dWo
            Gbo += dBo*dBo
            out_weights -= base_grad.update(dWo) / (delta+np.sqrt(Gwo))
            out_bias -= base_grad.update(dBo) / (delta+np.sqrt(Gbo))
            hidden_weights -= base_grad.update(dWh) / (delta+np.sqrt(Gwh))
            hidden_bias -= base_grad.update(dBh) / (delta+np.sqrt(Gbh))

    #print("MSE after training: " + str(mean_squared_error(Predict(X),yTarget)))
    #print("R2 without training: " + str(r2_score(Predict(X),yTarget)))

def FeedForward(X):
    activationFunc = activator.sigmoid
    z_h = np.matmul(X,hidden_weights)+ hidden_bias
    
    a_h = activationFunc(z_h)
    
    z_o = np.matmul(a_h, out_weights) + out_bias
    
    prob = z_o
    return prob, a_h

def BackProp(X, Y):
    
    prob, a_h = FeedForward(X)
    error_output = prob.T - Y
    error_hidden = np.matmul(error_output.T, out_weights.T) * a_h * (1 - a_h)
    
    output_weights_gradient = np.matmul(a_h.T, error_output.T)
    output_bias_gradient = np.sum(error_output, axis=1)

    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

def Predict(X):
    prob, _ = FeedForward(X)
    return prob #[1 if x > 0.5 else 0 for x in prob]


if __name__ == "__main__":

    #Set up the Input Matrix X with random numbers inside
    np.random.seed(2906)
    N = 1000

    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    yTarget = FrankeFunction(x, y)
    
    X = np.column_stack((x,y))

    X_Train, X_Test, y_Train, y_Test = train_test_split(X, yTarget, test_size=0.8, shuffle=True)
    
    #Define data for gradient descent

    #eta = 0.01
    n_epochs  = 1000
    M = 2
    #lmbd = 0.01
    k = j = 0
    
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for eta in eta_vals:
        j = 0
        for lmbd in lmbd_vals:
            Train(X_Train,y_Train,eta,n_epochs,M, lmbd)
            test_accuracy[k][j] = mean_squared_error(Predict(X_Test), y_Test)
            j += 1
        k += 1

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="PiYG", vmax=1.75)
    ax.set_title("MSE based on lambda and eta values using sigmoid")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('.\Images\MSEFrankesNeural_Sigmoid.png', bbox_inches='tight')


