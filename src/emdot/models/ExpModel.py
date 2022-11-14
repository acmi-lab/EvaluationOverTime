"""
Abstract class for ML models.
"""

from abc import ABC, abstractmethod

class ExpModel(ABC):
    @abstractmethod
    def __init__(self, name):
        """Initializes ML model object.

        """
        self.name = name

    @abstractmethod
    def fit(self, X_train, y_train):
        """Training ML models given training data

        Args:
            X_training: features for training
            y_training: labels for training

        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluating ML models given data

        Args:
            X: features of data for evaluation
            y: labels of data for evaluation
        """
        pass

    def get_name(self):
        """Get the name of ML models.
        
        Returns: string of model name
        """
        return self.name

    @abstractmethod
    def get_coefs(self, feature_names):
        """Get coefficients of model if there are any."""
        return None
    