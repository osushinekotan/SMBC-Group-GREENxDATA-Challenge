# type: ignore

import re

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreatedAtTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "created_at") -> None:
        self.col = col

    @property
    def feature_names(self) -> list[str]:
        return ["created_at__year", "created_at__month", "created_at__day"]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ts = pd.to_datetime(X[self.col])

        output_df = pd.DataFrame()
        output_df = output_df.assign(
            created_at__year=ts.dt.year,
            created_at__month=ts.dt.month,
            created_at__day=ts.dt.day,
        )

        return output_df[self.feature_names]


class CurbLocationTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "curb_loc") -> None:
        self.col = col
        self.mapping = {"OnCurb": 1, "OffsetFromCurb": 0}

    @property
    def feature_names(self) -> list[str]:
        return ["curb_loc_binary"]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = [X[self.col].map(self.mapping)]
        output_df = pd.DataFrame({name: feature for name, feature in zip(self.feature_names, features)})
        return output_df


class StreetWidthTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "steward") -> None:
        self.col = col
        self.mapping = {"1or2": 0, "3or4": 1, "4orMore": 2}

    def fit(self, X, y=None):
        return self

    @property
    def feature_names(self) -> list[str]:
        return ["steward_rank"]

    def transform(self, X, y=None):
        features = [X[self.col].map(self.mapping)]
        output_df = pd.DataFrame({name: feature for name, feature in zip(self.feature_names, features)})
        return output_df


class GuardsTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "guards") -> None:
        self.col = col
        self.mapping = {"Helpful": 0, "Unsure": 1, "Harmful": 2}

    def fit(self, X, y=None):
        return self

    @property
    def feature_names(self) -> list[str]:
        return ["guards_rank"]

    def transform(self, X, y=None):
        features = [X[self.col].map(self.mapping)]
        output_df = pd.DataFrame({name: feature for name, feature in zip(self.feature_names, features)})
        return output_df


class SidewalkTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "sidewalk") -> None:
        self.col = col
        self.mapping = {"NoDamage": 0, "Damage": 1}

    def fit(self, X, y=None):
        return self

    @property
    def feature_names(self) -> list[str]:
        return ["sidewalk_binary"]

    def transform(self, X, y=None):
        features = [X[self.col].map(self.mapping)]
        output_df = pd.DataFrame({name: feature for name, feature in zip(self.feature_names, features)})
        return output_df


class UserTypeTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "user_type") -> None:
        self.col = col
        self.mapping = {"Volunteer": 0, "TreesCount Staff": 1, "NYC Parks Staff": 2}

    def fit(self, X, y=None):
        return self

    @property
    def feature_names(self) -> list[str]:
        return ["user_type_rank", "is_volunteer"]

    def transform(self, X, y=None):
        user_types = X[self.col].map(self.mapping)
        features = [user_types, user_types == 0]
        output_df = pd.DataFrame({name: feature for name, feature in zip(self.feature_names, features)})
        return output_df


class ProblemsTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "problems") -> None:
        self.col = col

    def fit(self, X, y=None):
        return self

    @property
    def feature_names(self) -> list[str]:
        return [
            "num_problems",
            "problem_is_Nan",
            "problem_is_Stones",
            "problem_is_Branch",
            "problem_is_Lights",
            "problem_is_Trunk",
            "problem_is_Other",
            "problem_is_Wires",
            "problem_is_Rope",
            "problem_is_Metal",
            "problem_is_Grates",
            "problem_is_Root",
            "problem_is_Sneakers",
        ]

    def make_num_problems(self, problems: pd.Series | list[str]) -> list:
        num_problems = [len(re.split("(?=[A-Z])", problem)[1:]) if problem != "nan" else np.nan for problem in problems]
        return num_problems

    def make_problems_onehot(self, X) -> pd.DataFrame:
        df = X[[self.col]].copy()
        for index, item in df[["problems"]].fillna("Nan").iterrows():
            elements = re.split("(?=[A-Z])", item["problems"])
            for element in elements:
                if element:
                    df.at[index, element] = 1
            if "Other" in item:
                df.at[index, "Other"] = 1
        return df.drop(columns=["problems"]).fillna(0).astype(int).add_prefix("problem_is_")

    def transform(self, X, y=None):
        features_num_problems_df = pd.DataFrame(
            {self.feature_names[0]: self.make_num_problems(X[self.col].fillna("nan"))}
        )
        features_problems_onehot_df = self.make_problems_onehot(X)[self.feature_names[1:]]
        output_df = pd.concat([features_num_problems_df, features_problems_onehot_df], axis=1)
        return output_df


class NtaTransformerV01(BaseEstimator, TransformerMixin):
    def __init__(self, col: str = "nta") -> None:
        self.col = col
        self.oe = ce.OrdinalEncoder()

    def fit(self, X, y=None):
        df = self.parse_nta(X)
        self.oe.fit(df)
        return self

    @property
    def feature_names(self) -> list[str]:
        return ["oe_nta", "oe_nta_char", "oe_nta_num"]

    def parse_nta(self, X):
        df = X[[self.col]].copy()
        df["nta_char"] = df[self.col].str[:2]
        df["nta_num"] = df[self.col].str[2:]
        return df

    def transform(self, X, y=None):
        df = self.parse_nta(X)
        output_df = self.oe.transform(df).add_prefix("oe_")
        return output_df[self.feature_names]


class RawTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    @property
    def feature_names(self) -> list[str]:
        return self.cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output_df = X[self.cols]
        return output_df[self.feature_names]


class OrdinalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=list[str]) -> None:
        self.cols = cols
        self.oe = ce.OrdinalEncoder()

    @property
    def feature_names(self) -> list[str]:
        return [f"oe_{col}" for col in self.cols]

    def fit(self, X, y=None):
        self.oe.fit(X[self.cols].astype(str))
        return self

    def transform(self, X, y=None):
        output_df = self.oe.transform(X[self.cols].astype(str)).add_prefix("oe_")
        return output_df[self.feature_names]


class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dummy=None) -> None:
        self.dummy = dummy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.feature_names = X.columns.tolist()
        return X
