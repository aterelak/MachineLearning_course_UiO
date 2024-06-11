import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from GD_without_autograd import GD_without_autograd
from GD_with_autograd import GD_with_autograd


np.random.seed(2906)
N = 1000
x = 2*np.random.rand(N,1)
y = 2+(3*x)+(8*(x**2))
degree = 4

X = np.c_[np.ones((N,1)),x, x**2, x**3]


eta_vals = np.logspace(-5, 1, 7)

test_accuracy = np.zeros((1,len(eta_vals)))
k = j = 0
n_epochs = 100
minibatch = 10
momentum = 0.5
rho = 0.99
beta1 = 0.9
beta2 = 0.999


for eta in eta_vals:
    j = 0
    #for lmbd in lmbd_vals:
    gd = GD_with_autograd.ADAM(X, y, n_epochs, minibatch, eta, beta1, beta2)
    ypred = X.dot(gd)
    test_accuracy[0][k] = mean_squared_error(ypred,y)
        #j += 1
    k += 1

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="PiYG", vmax=1.75)
ax.set_title("MSE OLS ADAM, with Adagrad")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('.\Images\ADAMOLS.png', bbox_inches='tight')


lmbd_vals = np.logspace(-5, 1, 7)
test_accuracy_ridge = np.zeros((len(eta_vals),len(lmbd_vals)))
k = j = 0

for eta in eta_vals:
    j = 0
    for lmbd in lmbd_vals:
        gd = GD_with_autograd.ADAM(X, y, n_epochs, minibatch, eta, beta1, beta2, lmbd=lmbd)
        ypred = X.dot(gd)
        test_accuracy_ridge[k][j] = mean_squared_error(ypred,y)
        j += 1
    k += 1

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy_ridge, annot=True, ax=ax, cmap="PiYG", vmax=1.75)
ax.set_title("MSE Ridge ADAM, with Adagrad")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig('.\Images\ADAMRidge.png', bbox_inches='tight')