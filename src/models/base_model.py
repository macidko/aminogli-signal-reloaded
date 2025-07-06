from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, path):
        """Opsiyonel: Modeli kaydet."""
        pass

    def load(self, path):
        """Opsiyonel: Kaydedilmiş modeli yükle."""
        pass
