import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline


class PandasColumnTransformer(ColumnTransformer, TransformerMixin):  # type: ignore
    """output transformed data as pandas DataFrame with feature names"""

    def transform(self, X) -> pd.DataFrame:  # type: ignore
        result = super().transform(X)
        feature_names = []
        for trans in self.transformers_:
            print(trans)
            if isinstance(trans[1], Pipeline):
                for step in trans[1].steps:
                    if isinstance(step[1], TransformerMixin):
                        feature_names.extend(step[1].feature_names)
            elif isinstance(trans[1], TransformerMixin):
                feature_names.extend(trans[1].feature_names)
        return pd.DataFrame(result, columns=feature_names)

    def fit_transform(self, X, y=None) -> pd.DataFrame:  # type: ignore
        result = super().fit_transform(X, y)
        feature_names = []
        for trans in self.transformers_:
            print(trans)
            if isinstance(trans[1], Pipeline):
                for step in trans[1].steps:
                    if isinstance(step[1], TransformerMixin):
                        feature_names.extend(step[1].feature_names)
            elif isinstance(trans[1], TransformerMixin):
                feature_names.extend(trans[1].feature_names)
        return pd.DataFrame(result, columns=feature_names)


class PandasFeatureUnion(FeatureUnion, TransformerMixin):  # type: ignore
    """output transformed data as pandas DataFrame with feature names"""

    def transform(self, X) -> pd.DataFrame:  # type: ignore
        result = super().transform(X)
        feature_names = []
        for trans in self.transformer_list:
            print(trans)
            if isinstance(trans[1], Pipeline):
                for step in trans[1].steps:
                    if isinstance(step[1], TransformerMixin):
                        feature_names.extend(step[1].feature_names)
            elif isinstance(trans[1], TransformerMixin):
                feature_names.extend(trans[1].feature_names)
        return pd.DataFrame(result, columns=feature_names)

    def fit_transform(self, X, y=None) -> pd.DataFrame:  # type: ignore
        result = super().fit_transform(X, y)
        feature_names = []
        for trans in self.transformer_list:
            print(trans)
            if isinstance(trans[1], Pipeline):
                for step in trans[1].steps:
                    if isinstance(step[1], TransformerMixin):
                        feature_names.extend(step[1].feature_names)
            elif isinstance(trans[1], TransformerMixin):
                feature_names.extend(trans[1].feature_names)
        return pd.DataFrame(result, columns=feature_names)
