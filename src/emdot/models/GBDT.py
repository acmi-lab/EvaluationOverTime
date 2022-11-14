"""
Class for GBDT (Gradient Boosting Decision Tree) models
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

from emdot.models.ExpModel import ExpModel

class GBDT(ExpModel):
    def __init__(
        self, 
        n_estimators: int, 
        max_depth: int, 
        learning_rate: float
        ):
        """Initializes GBDT object
        
        Args:
            n_estimators: the number of boosting stages to perform
            max_depth: the maximum depth of the individual regression estimators
            learning_rate: learning rate shrinks the contribution of each tree by learning_rate

        """

        super().__init__(name="GBDT")

        self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    
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
    