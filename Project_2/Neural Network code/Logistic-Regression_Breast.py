from ucimlrepo import fetch_ucirepo 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Object_neurals import LogisticReg
from sklearn.model_selection import train_test_split
import scikitplot as skplt

if __name__ == "__main__":
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    yTarget = breast_cancer_wisconsin_diagnostic.data.targets 

    yTarget = [0 if x == 'M' else 1 for x in yTarget["Diagnosis"]]
    X = X.to_numpy()

    X_Train, X_Test, y_Train, y_Test = train_test_split(X, yTarget, test_size=0.8, shuffle=True)

    eta = 0.1
    lmbd = 0.01
    epochs = 1000
    M = 2
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    k = j = 0
    for eta in eta_vals:
        j = 0
        for lmbd in lmbd_vals:
            Logistic = LogisticReg(eta, epochs, lmbd)
            Logistic.Initialize(X_Train,y_Train, M=M)
            Logistic.Train()
            test_accuracy[k][j] = Logistic.Score(X_Test, y_Test)
            j += 1
        k += 1
   fig, ax = plt.subplots(figsize = (10, 10))
   sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="PiYG", vmax=1.75)
   ax.set_title("Test Accuracy based on various lambda and eta values")
   ax.set_ylabel("$\eta$")
   ax.set_xlabel("$\lambda$")
   plt.savefig('.\Images\AccuracyWisconsinLogistic.png', bbox_inches='tight')
    
   #eta = 1e-2
   #lmbd = 1e0
   #Logistic = LogisticReg(eta, epochs, lmbd)
   #Logistic.Initialize(X_Train,y_Train, M=M)
   #Logistic.Train()
   #y_pred = Logistic.Predict(X)

   #skplt.metrics.plot_confusion_matrix(yTarget, y_pred, normalize=True)
   #plt.savefig('.\Images\Accuracy_e-2_l0.png', bbox_inches='tight')

    
