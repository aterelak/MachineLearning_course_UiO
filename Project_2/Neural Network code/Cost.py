import autograd.numpy as np
def CostOLS(target):
    
    def function(X):
        return (1/target.shape[0]) * np.sum((target-X)**2)
        
    return function

def CostLogReg(target):
    
    def function(X):
        return -(1.0 / target.shape[0]) * np.sum((target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10)))
        
    return function

def CostCrossEntropy(target):
    
    def function(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))
        
    return function