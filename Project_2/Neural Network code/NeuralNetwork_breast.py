from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from Object_neurals import NeuralNetwork
import scikitplot as skplt
from tqdm import tqdm

# def sigmoid(x):
#     return 1/(1+npA.exp(-x))

# def CrossEntropy(X, target):
#     return -(1.0 / target.size) * npA.sum(target * npA.log(X + 10e-10))

if __name__ == "__main__":

    #Set up the Input Matrix X with random numbers inside
    np.random.seed(2906)
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=15) 
  
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features
    yTarget = breast_cancer_wisconsin_diagnostic.data.targets

    yTarget = [0 if x == 2 else 1 for x in yTarget["Class"]]
    X = X.to_numpy()

    X_Train, X_Test, y_Train, y_Test = train_test_split(X, yTarget, test_size=0.8, shuffle=True)
    
    #Define data for gradient descent
    #print(X.shape)
    n_epochs  = 1000
    M = 2
    k = j = 0
    n_h_neurons = 2
    n_categ = 1
    
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for k in tqdm(range(len(eta_vals))):
        j = 0
        for j in tqdm(range(len(lmbd_vals))):
            NN = NeuralNetwork(eta_vals[k], n_epochs, lmbd_vals[j])
            NN.Initialize(X_Train,y_Train,n_categ, n_h_neurons)
            NN.Train()
            test_accuracy[k][j] = NN.Score(X_Test, y_Test)

    

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="PiYG", vmax=1.75)
    ax.set_title("Test Accuracy based on various lambda and eta values")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig('.\Images\AccuracyWisconsinNeural.png', bbox_inches='tight')
    
    # eta = 1e-1
    # lmbd = 1e-4
    # NN = NeuralNetwork(eta, n_epochs, lmbd)
    # NN.Initialize(X,yTarget,n_categ, n_h_neurons)
    # NN.Train()
    # y_pred = NN.Predict(X)

    # skplt.metrics.plot_confusion_matrix(yTarget, y_pred, normalize=True)
    # plt.savefig('.\Images\Accuracy_e-1_l-4.png', bbox_inches='tight')

