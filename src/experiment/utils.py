import hashlib
import json

import pandas as pd
from sklearn.model_selection import BaseCrossValidator


def assign_fold_index(
    train_df: pd.DataFrame,
    kfold: BaseCrossValidator,
    y_col: str,
    group_col: str | None = None,
) -> pd.DataFrame:
    train_df["fold"] = -1

    strategy = (
        kfold.split(X=train_df, y=train_df[y_col])
        if group_col is None
        else kfold.split(X=train_df, y=train_df[y_col], groups=train_df[group_col])
    )
    for fold_index, (_, valid_index) in enumerate(strategy):
        train_df.loc[valid_index, "fold"] = fold_index
    return train_df


def make_uid(source_dict: dict) -> str:
    dict_str = json.dumps(source_dict, sort_keys=True)
    return hashlib.sha256(dict_str.encode()).hexdigest()
