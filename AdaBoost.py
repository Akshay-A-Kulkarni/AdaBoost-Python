import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn import metrics
%matplotlib inline

class AdaBoost():
    
    """
    AdaBoost class, a Boosting method that uses a number of weak classifiers in 
    ensemble to make a strong classifier. The implementation uses decision
    stumps, which is a one level Decision Tree. 
    
    @params:
    
    𝑇: int The number of weak classifiers that will be used. 
    """
    def __init__(self, 𝑇 = 10):
        self.n_clf = 𝑇
        self.clfs= None
        self.betas = None
        
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))
        # Initialising lists to saev models and respective betas
        self.clfs = []
        self.betas = []
        # Iterate through classifiers
        for _ in range(self.n_clf):
            # creating stump classifier
            sklearn_stump = sklearn.tree.DecisionTreeClassifier(random_state=0, max_depth=1)
            # Fitting tree with input weights
            sklearn_stump.fit(X,y,sample_weight=w)
            # Evaluating model
            predictions = sklearn_stump.predict(X)
            # Calculating error
            accuracy = metrics.accuracy_score(y, predictions,sample_weight=w)
            error = 1 - accuracy
            # Calculate the alpha which is used to update the sample weights,
            # Alpha is also an approximation of this classifier's proficiency
            beta = 0.5 * math.log((1.0 - error) / (error + 1e-10))
            # Saving correspoding beta for the clasifier
            self.betas.append(beta)
            # Calculate new weights 
            # Missclassified samples gets larger weights and correctly classified samples smaller
            w *= np.exp(-beta * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(sklearn_stump)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        # Iterating through each classifier
        for i in range(self.n_clf):
            # Getting predictions
            predictions = np.expand_dims(self.clfs[i].predict(X),1)
            # Add predictions weighted by the classifiers alpha
            # (alpha indicative of classifier's proficiency)
            y_pred += self.betas[i] * predictions

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred

def main():
    X = features.to_numpy()
    y = response.to_numpy()
    
    # Changing labels to {-1, 1}
    y[y == 0] = -1
    y[y == 1] = 1

    # Creating train test splits 
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size = 0.25, random_state = 0, stratify=y)
    
    # Adaboost classification with T weak classifiers
    aclf = AdaBoost(𝑇=50)
    aclf.fit(X_train, y_train)
    y_pred = aclf.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)
      
main() 