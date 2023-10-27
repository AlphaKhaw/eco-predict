from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator):
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def feature_importance(self):
        raise NotImplementedError
