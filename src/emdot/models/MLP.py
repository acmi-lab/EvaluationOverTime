from tkinter import Y
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

from emdot.models.ExpModel import ExpModel

class MLP(ExpModel):
    def __init__(
        self, 
        hidden_layer_sizes: tuple, 
        learning_rate_init: float, 
        max_iter: int = 100):
        
        """Initializes MLP object
        
        Args:
            hidden_layer_sizes: the ith element represents the number of neurons in the ith hidden layer
            learning_rate_init: learning rate schedule for weight updates.
            max_iter: maximum iterations for model training
        """
        super().__init__(name="MLP")
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                   learning_rate_init=learning_rate_init, 
                                   max_iter=max_iter)
    
    def fit(self, X_train, y_train):
        """Training ML models given training data

        Args:
            X_train: features for training
            y_train: labels for training

        """
        self.model.fit(X_train, y_train)
        
    def evaluate(self, x, y):
        """Evaluating ML models given data

        Args:
            X: features of data for evaluation
            y: labels of data for evaluation

        """
        y_true = y
        y_score = self.model.predict_proba(x)[:, 1]
        y_pred = self.model.predict(x)

        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return {
            "auc": auc,
            "auprc": auprc,
            "acc": acc,
            "f1": f1
        }
    
        