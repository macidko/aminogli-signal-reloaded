from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, task='classification', **params):
        self.task = task
        self.params = params
        if task == 'classification':
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        return self
