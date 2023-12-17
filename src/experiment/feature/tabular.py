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

        self.group_keys = group_keys
        self.group_values = group_values
        self.agg_methods = agg_methods
        self.extr_agg_methods = extr_agg_methods

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
        new_df = input_df.merge(self.df_agg, how="left", on=self.group_keys)

        # 追加集約方法の計算
        if "z-score" in self.extr_agg_methods:
            new_df = self.calculate_z_scores(new_df)

        return new_df
