import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
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


def visualize_feature_importance(
    estimators: list,
    feature_columns: list[str],
    plot_type: str = "boxen",
    top_n: int | None = None,
) -> Figure | pd.DataFrame:
    feature_importance_df = pd.DataFrame()

    for i, model in enumerate(estimators):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importances_  # type:ignore
        _df["column"] = feature_columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, _df],
            axis=0,
            ignore_index=True,
        )

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index
    )
    if top_n is not None:
        order = order[:top_n]

    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.25)))
    plot_params = dict(
        data=feature_importance_df,
        x="feature_importance",
        y="column",
        order=order,
        ax=ax,
        palette="viridis",
        orient="h",
    )
    if plot_type == "boxen":
        sns.boxenplot(**plot_params)
    elif plot_type == "bar":
        sns.barplot(**plot_params)
    else:
        raise NotImplementedError()

    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()
    return fig, feature_importance_df
