import pickle

def save_model(model, path: str):
    """Save a model to disk using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str):
    """Load a model from disk using pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f) 