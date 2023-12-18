from typing import Callable

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


class TargetEncoder(BaseFeatureExtractor):
    def __init__(
        self,
        group_keys: list[str],
        target_value: str,
        agg_methods: list[str | Callable],
        fold: str = "fold",
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

    def fit(self, input_df: pd.DataFrame) -> None:  # type: ignore
        # 各フォールドでの集約
        for fold_val in input_df[self.fold].unique():
            if fold_val < 0:
                continue
            other_folds_df = input_df[input_df[self.fold] != fold_val]
            self.encodings[fold_val] = self._calculate_fold_encodings(other_folds_df)

        # 全体の集約（テストデータ用）
        self.encodings["overall"] = self._calculate_fold_encodings(input_df)

    def _calculate_fold_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        agg_df = df.groupby(self.group_keys).agg({self.target_value: self.agg_methods})
        agg_df.columns = [f"te_{method}_{self.target_value}_grpby_{self.group_keys_str}" for method in self.agg_methods]
        return agg_df.reset_index()

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        encoded_columns = []

        for fold_val, encodings in self.encodings.items():
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
