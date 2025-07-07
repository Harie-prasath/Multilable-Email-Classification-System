import numpy as np
from models.intent_model import IntentModel
from models.tone_model import ToneModel
from models.resolution_model import ResolutionModel
from scipy.sparse import hstack

class ChainedModel:
    """Chained multi-output classifier: Intent -> Tone -> Resolution Type."""
    def __init__(self):
        self.intent_model = IntentModel()
        self.tone_model = ToneModel()
        self.resolution_model = ResolutionModel()

    def fit(self, X, y_intent, y_tone, y_resolution):
        """Train the chained models in sequence."""
        # Train intent
        self.intent_model.fit(X, y_intent)
        # Predict intent for training set
        intent_pred = self.intent_model.predict(X).reshape(-1, 1)
        # Add intent prediction to features for tone
        X_tone = hstack([X, intent_pred])
        self.tone_model.fit(X_tone, y_tone)
        # Predict tone for training set
        tone_pred = self.tone_model.predict(X_tone).reshape(-1, 1)
        # Add intent and tone predictions to features for resolution
        X_resolution = hstack([X, intent_pred, tone_pred])
        self.resolution_model.fit(X_resolution, y_resolution)

    def predict(self, X):
        """Predict all three labels in sequence."""
        intent_pred = self.intent_model.predict(X).reshape(-1, 1)
        X_tone = hstack([X, intent_pred])
        tone_pred = self.tone_model.predict(X_tone).reshape(-1, 1)
        X_resolution = hstack([X, intent_pred, tone_pred])
        resolution_pred = self.resolution_model.predict(X_resolution)
        return intent_pred.ravel(), tone_pred.ravel(), resolution_pred.ravel() 