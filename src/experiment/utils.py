import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
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
    data = {}
    for key, value in source_dict.items():
        try:
            json.dumps(value)  # 値が JSON シリアライズ可能かテストする
            data[key] = value
        except TypeError:
            data[key] = str(value)  # シリアライズできない場合は文字列として扱う
    dict_str = json.dumps(data, sort_keys=True)
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


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = False) -> Figure:
    """
    This function computes and returns a confusion matrix as a matplotlib figure.
    :param y_true: Array of true labels
    :param y_pred: Array of predicted labels
    :param normalize: Boolean, whether to normalize the confusion matrix or not
    :return: Matplotlib figure object containing the confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    return fig


def plot_label_distributions(proba_matrix: np.ndarray) -> Figure:
    """
    Plots the distribution of probabilities for each label in the given matrix and returns the figure.

    Parameters:
    proba_matrix (numpy.ndarray): A matrix of shape (n, num_labels) containing probabilities.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plots.
    """
    num_labels = proba_matrix.shape[1]
    fig, ax = plt.subplots()

    # Create a distribution plot for each label
    for i in range(num_labels):
        sns.kdeplot(proba_matrix[:, i], ax=ax, fill=True, label=f"Label {i+1}")

    ax.set_title("Probability Distributions for Each Label")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.legend()

    return fig
