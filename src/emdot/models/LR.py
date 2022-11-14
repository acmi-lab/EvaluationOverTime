from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

from emdot.models.ExpModel import ExpModel

class LR(ExpModel):
    def __init__(
        self, 
        C: float, 
        max_iter: int = 100):
        """Initializes LR object
        
        Args:
            C: inverse of regularization strength
            max_iter: maximum iterations for model training

        """
        super().__init__(name="LR")
        self.model = LogisticRegression(C=C, max_iter=max_iter)

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

    def get_coefs(self, feature_names):
        """Gets the coefficients of features in LR model.
        
        Args:
            feature_names: list of features used for evaluation
            
        Returns:
            dictionary containing features and corresponding coefficients
            
        """
        assert(hasattr(self.model, 'coef_'))
        dict_coef = {}
        for coef_tmp, column in zip(self.model.coef_[0], feature_names):
            dict_coef[column] = coef_tmp
        return dict_coef