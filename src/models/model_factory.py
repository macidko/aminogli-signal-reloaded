from .random_forest import RandomForestModel
# from .lightgbm import LightGBMModel  # İleride eklenebilir
# from .xgboost import XGBoostModel    # İleride eklenebilir
# from .lstm import LSTMModel          # İleride eklenebilir

MODEL_REGISTRY = {
    'random_forest': RandomForestModel,
    # 'lightgbm': LightGBMModel,
    # 'xgboost': XGBoostModel,
    # 'lstm': LSTMModel,
}

def get_model(model_name, task='classification', **params):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](task=task, **params)

# Kullanım örneği (üretim ortamında kaldırılmalı):
# model = get_model('random_forest', task='classification', n_estimators=100)
