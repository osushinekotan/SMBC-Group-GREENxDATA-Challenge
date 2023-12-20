# type: ignore


import re

import category_encoders as ce
import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(search_from="../", indicator=".project-root", pythonpath=True)

from src.experiment.base import BaseFeatureExtractor


class CreatedAtFeatureExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        ts = pd.to_datetime(input_df["created_at"])

        output_df = pd.DataFrame()
        output_df = output_df.assign(
            created_at__year=ts.dt.year,
            created_at__month=ts.dt.month,
            created_at__day=ts.dt.day,
        )
        output_df["tree_age"] = 2023 - output_df["created_at__year"]
        output_df["tree_age_bins10"] = pd.cut(output_df["tree_age"], bins=10, labels=False)

        return output_df


class TreeDbhFeatureExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        tree_dbh_s = input_df["tree_dbh"].astype(str).str.zfill(2)
        output_df = pd.DataFrame(tree_dbh_s.apply(list).tolist(), columns=["tree_dbh_01", "tree_dbh_02"]).astype(int)
        output_df["tree_dbh_bins10"] = pd.cut(input_df["tree_dbh"], bins=10, labels=False)

        return output_df


class CurbLocationFeatureExtractorV1(BaseFeatureExtractor):
    def __init__(self):
        self.mapping = {"OnCurb": 1, "OffsetFromCurb": 0}

    def transform(self, input_df):
        output_df = pd.DataFrame({"curb_loc_binary": input_df["curb_loc"].map(self.mapping).tolist()})
        return output_df


class StreetWidthFeatureExtractorV1(BaseFeatureExtractor):
    def __init__(self):
        self.mapping = {"1or2": 0, "3or4": 1, "4orMore": 2}

    def transform(self, input_df):
        output_df = pd.DataFrame({"steward_rank": input_df["steward"].map(self.mapping).tolist()})
        return output_df


class GuardsFeatureExtractorV1(BaseFeatureExtractor):
    def __init__(self):
        self.mapping = {"Helpful": 0, "Unsure": 1, "Harmful": 2}

    def transform(self, input_df):
        output_df = pd.DataFrame({"guards_rank": input_df["guards"].map(self.mapping).tolist()})
        return output_df


class SidewalkFeatureExtractorV1(BaseFeatureExtractor):
    def __init__(self) -> None:
        self.mapping = {"NoDamage": 0, "Damage": 1}

    def transform(self, input_df):
        output_df = pd.DataFrame({"sidewalk_binary": input_df["sidewalk"].map(self.mapping).tolist()})
        return output_df


class UserTypeFeatureExtractorV1(BaseFeatureExtractor):
    def __init__(self) -> None:
        self.mapping = {"Volunteer": 0, "TreesCount Staff": 1, "NYC Parks Staff": 2}

    def transform(self, input_df):
        user_types = input_df["user_type"].map(self.mapping)

        output_df = pd.DataFrame()
        output_df["user_type_rank"] = input_df["user_type"].map(self.mapping)
        output_df["is_volunteer"] = np.array(user_types == 0, dtype=int)

        return output_df


class ProblemsFeatureExtractorV1(BaseFeatureExtractor):
    # must concat train and test
    def make_num_problems_df(self, problems: pd.Series | list[str]) -> list:
        num_problems = [len(re.split("(?=[A-Z])", problem)[1:]) if problem != "nan" else np.nan for problem in problems]
        return pd.DataFrame({"num_problems": num_problems})

    def make_problems_onehot_df(self, input_df) -> pd.DataFrame:
        df = input_df[["problems"]].copy()
        for index, item in df[["problems"]].fillna("Nan").iterrows():
            elements = re.split("(?=[A-Z])", item["problems"])
            for element in elements:
                if element:
                    df.at[index, element] = 1
            if "Other" in item:
                df.at[index, "Other"] = 1
        return df.drop(columns=["problems"]).fillna(0).astype(int).add_prefix("problem_is_")

    def transform(self, input_df):
        features_num_problems_df = self.make_num_problems_df(input_df["problems"].fillna("nan"))
        features_problems_onehot_df = self.make_problems_onehot_df(input_df)
        output_df = pd.concat([features_num_problems_df, features_problems_onehot_df], axis=1)
        return output_df


class FirstProblemOrdinalFeatureExtractorV1(BaseFeatureExtractor):
    def __init__(self, parents: list[BaseFeatureExtractor] | None = None):
        super().__init__(parents)
        self.oe = ce.OrdinalEncoder()

    def make_first_problem_df(self, problems: pd.Series | list[str]) -> pd.DataFrame:
        df = pd.DataFrame()
        df["first_problem"] = pd.Series(problems).fillna("Nan").apply(lambda x: re.split("(?=[A-Z])", x)[1])
        return df

    def fit(self, input_df):
        df = self.make_first_problem_df(input_df["problems"])
        self.oe.fit(df[["first_problem"]])
        return self

    def transform(self, input_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = self.make_first_problem_df(input_df["problems"])
        output_df = self.oe.transform(df[["first_problem"]]).add_prefix("oe_")
        return output_df


class NtaFeatureExtractorV1(BaseFeatureExtractor):
    def __init__(self) -> None:
        self.oe = ce.OrdinalEncoder()

    def fit(self, input_df):
        df = self.parse_nta(input_df)
        self.oe.fit(df)
        return self

    def parse_nta(self, input_df):
        df = input_df[["nta"]].copy()
        df["nta_char"] = df["nta"].str[:2]
        df["nta_num"] = df["nta"].str[2:]
        return df

    def transform(self, X, y=None):
        df = self.parse_nta(X)
        output_df = self.oe.transform(df).add_prefix("oe_")
        return output_df


class RawTransformer(BaseFeatureExtractor):
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def transform(self, input_df):
        output_df = input_df[self.cols].copy()
        return output_df.astype(np.float32)


class OrdinalFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, cols=list[str]) -> None:
        self.cols = cols
        self.oe = ce.OrdinalEncoder()

    def fit(self, input_df):
        self.oe.fit(input_df[self.cols].astype(str))
        return self

    def transform(self, input_df):
        output_df = self.oe.transform(input_df[self.cols].astype(str)).add_prefix("oe_")
        return output_df
