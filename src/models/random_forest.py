from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import json
import os
from datetime import datetime
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

    def save(self, path, metrics=None):
        """
        Modeli kaydeder ve aynı isimle metadata (parametreler, skorlar) içeren bir .json dosyası oluşturur.
        """
        joblib.dump(self.model, path)
        # Metadata dosyası oluştur
        meta = {
            'params': self.params,
            'task': self.task,
            'saved_at': datetime.now().isoformat()
        }
        if metrics is not None:
            meta['metrics'] = metrics
        meta_path = os.path.splitext(path)[0] + '_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path):
        self.model = joblib.load(path)
        return self
