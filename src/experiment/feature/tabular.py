from itertools import combinations
from typing import Callable

import category_encoders as ce
import numpy as np
import pandas as pd

from src.experiment.base import BaseFeatureExtractor


class AggregatedFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        group_keys: list[str],
        group_values: list[str],
        agg_methods: list[str | Callable],
        extr_agg_methods: list[str] = [],
        parents: list[BaseFeatureExtractor] | None = None,
    ):
        super().__init__(parents)
        self.parents = parents

        self.group_keys = list(group_keys)
        self.group_values = list(group_values)
        self.agg_methods = list(agg_methods)
        self.extr_agg_methods = list(extr_agg_methods)

        if "z-score" in self.extr_agg_methods:
            if "mean" not in self.agg_methods:
                self.agg_methods.append("mean")
            if "std" not in self.agg_methods:
                self.agg_methods.append("std")

        self.df_agg = None

    def fit(self, input_df: pd.DataFrame) -> None:  # type: ignore
        agg_dfs = []
        for agg_method in self.agg_methods:
            method_name = agg_method if isinstance(agg_method, str) else agg_method.__name__
            for col in self.group_values:
                new_col = f"agg_{method_name}_{col}_grpby_{'_'.join(self.group_keys)}"
                df_agg = (input_df[self.group_keys + [col]].groupby(self.group_keys)[col].agg(agg_method)).to_frame(
                    name=new_col
                )
                agg_dfs.append(df_agg)

        self.df_agg = pd.concat(agg_dfs, axis=1).reset_index()

    def calculate_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.group_values:
            mean_col = f"agg_mean_{col}_grpby_{'_'.join(self.group_keys)}"
            std_col = f"agg_std_{col}_grpby_{'_'.join(self.group_keys)}"
            z_score_col = f"z-score_{col}_grpby_{'_'.join(self.group_keys)}"
            df[z_score_col] = (df[col] - df[mean_col]) / df[std_col].replace(0, np.nan)
        return df

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        # 集約データフレームを入力データフレームに結合
        new_df = input_df[self.group_keys + self.group_values].merge(self.df_agg, how="left", on=self.group_keys)

        # 追加集約方法の計算
        if "z-score" in self.extr_agg_methods:
            new_df = self.calculate_z_scores(new_df)
            return new_df.drop(columns=self.group_values + self.group_keys)

        new_df = new_df.drop(columns=self.group_keys)
        return new_df


def _q_25(x):  # type: ignore
    return x.quantile(0.25)


def _q_75(x):  # type: ignore
    return x.quantile(0.75)


def _q_10(x):  # type: ignore
    return x.quantile(0.1)


def _q_90(x):  # type: ignore
    return x.quantile(0.9)


class TargetEncoder(BaseFeatureExtractor):
    def __init__(
        self,
        group_keys: list[str],
        target_value: str,
        agg_methods: list[str | Callable],
        fold: str = "fold",
        max_cardinality: int = 1000000,
        parents: list[BaseFeatureExtractor] | None = None,
    ):
        super().__init__(parents)
        self.parents = parents
        self.group_keys = list(group_keys)
        self.target_value = target_value
        self.agg_methods = list(agg_methods)
        self.fold = fold
        self.encodings = {}  # type: ignore

        self.group_keys_str = "_".join(self.group_keys)
        self.max_cardinality = max_cardinality

    def fit(self, input_df: pd.DataFrame) -> None:  # type: ignore
        # 各フォールドでの集約
        for fold_val in input_df[self.fold].unique():
            if fold_val < 0:
                continue
            other_folds_df = input_df[input_df[self.fold] != fold_val]
            self.encodings[fold_val] = self._calculate_fold_encodings(other_folds_df)

        # 全体の集約（テストデータ用）
        self.encodings["overall"] = self._calculate_fold_encodings(input_df)

    def _convert_agg_method(self, agg_methods: list) -> list:
        converted_agg_methods = []
        for agg_method in agg_methods:
            if agg_method == "q25":
                agg_method = _q_25
            elif agg_method == "q75":
                agg_method = _q_75
            elif agg_method == "q10":
                agg_method = _q_10
            elif agg_method == "q90":
                agg_method = _q_90
            converted_agg_methods.append(agg_method)
        return converted_agg_methods

    def _calculate_fold_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        agg_methods = self._convert_agg_method(self.agg_methods)
        agg_df = df.groupby(self.group_keys).agg({self.target_value: agg_methods})
        agg_df.columns = [f"te_{method}_{self.target_value}_grpby_{self.group_keys_str}" for method in agg_methods]
        return agg_df.reset_index()

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        encoded_columns = []

        for fold_val, encodings in self.encodings.items():
            if len(encodings) >= self.max_cardinality:
                return pd.DataFrame()  # いい感じに変更したい (te で持たないようにしたい)

            # フォールドごとのデータを抽出
            if fold_val != "overall":
                fold_df = input_df[input_df[self.fold] == fold_val]
            else:
                fold_df = input_df[input_df[self.fold].isna() | (input_df[self.fold] < 0)]

            encoded_df = fold_df[self.group_keys].merge(encodings, on=self.group_keys, how="left")
            encoded_df = encoded_df.drop(columns=self.group_keys)

            # インデックスと結合して追加
            encoded_df.index = fold_df.index
            encoded_columns.append(encoded_df)

        # 全てのエンコードされたデータフレームを結合
        output_df = pd.concat(encoded_columns).sort_index()
        return output_df


class OrdinalFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        input_cols: list[str],
        parents: list[BaseFeatureExtractor] | None = None,
    ):
        super().__init__(parents)
        self.input_cols: list[str] = input_cols
        self.oe = ce.OrdinalEncoder()

    def fit(self, input_df: pd.DataFrame):  # type: ignore
        self.oe.fit(input_df[self.input_cols].astype(str))
        return self

    def transform(self, input_df):  # type: ignore
        output_df = self.oe.transform(input_df[self.input_cols].astype(str)).add_prefix("oe_")
        return output_df


class ConcatCombinationOrdinalEncoder(BaseFeatureExtractor):
    """Generate combination of string columns. (xfeat style)"""

    def __init__(
        self,
        input_cols: list[str] | None = None,
        include_cols: list[str] | None = None,
        output_prefix: str = "",
        output_suffix: str = "_combi",
        fillna: str = "_NaN_",
        r: int = 2,
        max_cardinality: int = 1000000,
        parents: list[BaseFeatureExtractor] | None = None,
    ):
        super().__init__(parents)
        self.parents = parents
        self._input_cols = input_cols or []
        self._include_cols = include_cols or []
        self._output_prefix = output_prefix
        self._output_suffix = output_suffix
        self._r = r
        self._fillna = fillna
        self.max_cardinality = max_cardinality
        self.oe = ce.OrdinalEncoder()

    def make_categorical_feature_df(self, input_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        cols = []

        n_fixed_cols = len(self._include_cols)
        df = input_df.copy()

        for cols_pairs in combinations(self._input_cols, r=self._r - n_fixed_cols):
            fixed_cols_str = "".join(self._include_cols)
            pairs_cols_str = "".join(cols_pairs)
            new_col = self._output_prefix + fixed_cols_str + pairs_cols_str + self._output_suffix
            cols.append(new_col)

            concat_cols = self._include_cols + list(cols_pairs)
            new_ser = None
            for col in concat_cols:
                if new_ser is None:
                    new_ser = df[col].fillna(self._fillna).astype(str).copy()
                else:
                    new_ser = new_ser + df[col].fillna(self._fillna).astype(str)  # type: ignore

            df[new_col] = new_ser

        use_cols = []
        for col in cols:
            if df[col].nunique() < self.max_cardinality:
                use_cols.append(col)

        return df[use_cols]

    def fit(self, input_df: pd.DataFrame, **kwargs):  # type: ignore
        df = self.make_categorical_feature_df(input_df)
        self.oe.fit(df)
        return self

    def transform(self, input_df: pd.DataFrame, **kwargs) -> pd.DataFrame:  # type: ignore
        df = self.make_categorical_feature_df(input_df)
        output_df = self.oe.transform(df).add_prefix("oe_")
        return output_df
