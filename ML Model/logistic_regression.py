import numpy as np

def sigmoid(X):
    return (1/(1+np.exp(-X)))

class LogisticRegression:
    
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            
            linear_pred  = np.dot(X, self.weights) + self.bias 
            #y = mx + c //// y = weight * x + bias

            predictions = sigmoid(linear_pred) #predictions used to properly train the model using Gradient Descent. 

            dw = (1/n_sample) * np.dot(X.T, (predictions - y))
            db = (1/n_sample) * np.sum(predictions - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_pred)

        class_preds = [0 if y<=0.5 else 1 for y in predictions]

        return class_preds