from sklearn.linear_model import LogisticRegression
import numpy as np

class ResolutionModel:
    """Classifier for Resolution Type label."""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the resolution type classifier."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict resolution type labels."""
        return self.model.predict(X) 