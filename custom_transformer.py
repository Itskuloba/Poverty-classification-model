from sklearn.base import BaseEstimator, TransformerMixin

class ToDataFrameTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        import pandas as pd
        return pd.DataFrame(X)
