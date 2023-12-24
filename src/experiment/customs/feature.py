# type: ignore


import re

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
            # created_at__day=ts.dt.day,
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


class SpcCommonFeatureExtractorV1(BaseFeatureExtractor):
    @staticmethod
    def clean(string: str) -> str:
        return string.replace("'", "")

    @staticmethod
    def split_tree_name(tree_name: str) -> list[str]:
        """
        "English oak" -> ["English", "", "oak"]
        "crimson king maple" -> ["", "crimson king", "maple"]
        "silver maple" -> ["", "silver", "maple"]
        """
        if tree_name is np.nan:
            return ["", "", ""]
        words = tree_name.split()
        country = words[0] if words[0].istitle() else ""
        main_type = " ".join(words[1:-1]) if country else " ".join(words[:-1])
        sub_type = words[-1]

        return [country, main_type, sub_type]

    def transform(self, input_df: pd.DataFrame):
        output_df = (
            input_df["spc_common"]
            .map(self.clean, na_action="ignore")
            .map(self.split_tree_name, na_action="ignore")
            .apply(pd.Series)
            .replace("", np.nan)
        )
        output_df.columns = ["spc_common_country", "spc_common_main_type", "spc_common_sub_type"]
        return output_df


class SpcLatinFeatureExtractorV1(BaseFeatureExtractor):
    @staticmethod
    def extract_genus_species(name):
        """
        Extracts the genus and species from a given Latin name.

        Parameters:
        name (str): The Latin name of the tree.

        Returns:
        tuple: A tuple containing the genus and species.
        """
        if name is np.nan:
            return ["", ""]
        parts = name.split()
        genus = parts[0] if parts else ""
        species = " ".join(parts[1:]) if len(parts) > 1 else ""
        return genus, species

    def transform(self, input_df: pd.DataFrame):
        output_df = input_df["spc_latin"].apply(self.extract_genus_species).apply(pd.Series).replace("", np.nan)
        output_df.columns = ["spc_latin_genus", "spc_latin_species"]
        return output_df


class NtaFeatureExtractorV1(BaseFeatureExtractor):
    def parse_nta(self, input_df):
        df = input_df[["nta"]].copy()
        output_df = pd.DataFrame()
        output_df["nta_char"] = df["nta"].str[:2]
        output_df["nta_num"] = df["nta"].str[2:]
        return output_df

    def transform(self, input_df: pd.DataFrame):
        df = self.parse_nta(input_df)
        return df


class RawTransformer(BaseFeatureExtractor):
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def transform(self, input_df):
        output_df = input_df[self.cols].copy()
        return output_df.astype(np.float32)
