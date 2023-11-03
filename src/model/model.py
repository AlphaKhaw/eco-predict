from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.base.base_model import BaseModel
from src.enums.enums import ModelType, get_model_enum

MODELS = {
    ModelType.RANDOM_FOREST: RandomForestRegressor,
    ModelType.XGBOOST: XGBRegressor,
}


class Model(BaseModel):
    def __init__(self, model_type: str, **kwargs):
        self.model_type = get_model_enum(model_type)
        self.model = get_model_instance(model_type, **kwargs)

    def __repr__(self):
        return repr(self.model)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        return self.model.feature_importances_


def get_model_instance(model_type, **kwargs):
    model_enum = get_model_enum(model_type)
    model_class = MODELS.get(model_enum)
    if model_class:
        return model_class(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
