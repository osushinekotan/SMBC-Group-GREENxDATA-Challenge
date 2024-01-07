import logging
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
import rootutils
from lightgbm import LGBMModel
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBModel

rootutils.setup_root(search_from="../", indicator=".project-root", pythonpath=True)
from src.experiment.utils import make_uid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

estimator_types = Union[XGBModel, LGBMModel]


def train_cv_tabular_v1(
    df: pd.DataFrame,
    estimator: estimator_types,
    feature_columns: list[str],
    target_columns: str,
    fit_params: dict | None,
    output_dir: Path,
    train_folds: list[int] | None = None,
    overwrite: bool = False,
    use_xgb_class_weight: bool | None = False,
    use_eval_set: bool = True,
) -> list[estimator_types]:
    """train cv for xgboost estimator"""
    estimators = []

    if fit_params is None:
        fit_params = {}

    if train_folds is None:
        train_folds = sorted(df["fold"].unique())

    if use_xgb_class_weight is None:
        use_xgb_class_weight = False

    for i_fold in train_folds:
        logger.info(f"start training fold={i_fold} 🚀 ")
        fit_estimator = clone(estimator)

        output_df_fold = output_dir / f"fold{i_fold}"
        output_df_fold.mkdir(exist_ok=True, parents=True)

        estimator_uid = make_uid(fit_estimator.__dict__)
        estimator_name = fit_estimator.__class__.__name__
        estimator_name_with_uid = f"{estimator_name}_{estimator_uid}"
        estimator_path = output_df_fold / f"{estimator_name}.pkl"

        if estimator_path.exists() and (not overwrite):
            logger.info(f"skip fitting in fold{i_fold}")
            fit_estimator = joblib.load(estimator_path)
            estimators.append(fit_estimator)
            continue

        # split train and valid
        train_df = df.query(f"fold != {i_fold}").reset_index(drop=True)
        valid_df = df.query(f"fold == {i_fold}").query("data == 'train'").reset_index(drop=True)
        tr_x, tr_y = train_df[feature_columns], train_df[target_columns]
        va_x, va_y = valid_df[feature_columns], valid_df[target_columns]

        logger.info(f"estimator : {estimator_name_with_uid}")

        if use_xgb_class_weight:
            if estimator_name == "XGBModel":
                fit_params["sample_weight"] = compute_sample_weight(class_weight="balanced", y=tr_y)
                fit_params["sample_weight_eval_set"] = [compute_sample_weight(class_weight="balanced", y=va_y)]
        if use_eval_set:
            fit_params["eval_set"] = [(va_x, va_y)]

        fit_estimator.fit(X=tr_x, y=tr_y, **fit_params)
        estimators.append(fit_estimator)

        joblib.dump(fit_estimator, estimator_path)

    return estimators


def predict_cv_tabular_v1(
    df: pd.DataFrame,
    estimators: list[estimator_types],
    feature_columns: list[str],
    train_folds: list[int] | None = None,
    test: bool = False,
    result_columns: list[str] | None = None,
    predict_proba: bool = True,
) -> pd.DataFrame:
    if result_columns is None:
        result_columns = [col for col in df.columns if col not in feature_columns]

    if train_folds is None:
        train_folds = list(range(len(estimators)))

    def _predict_i(df: pd.DataFrame, i_fold: int, estimator: estimator_types) -> pd.DataFrame:
        logger.info(f"fold{i_fold} predict : test={test}")
        if not test:
            df = df.query(f"fold == {i_fold}").reset_index(drop=True)

        va_x = df[feature_columns]

        if predict_proba:
            va_pred = estimator.predict_proba(va_x)
        else:
            va_pred = estimator.predict(va_x)
        i_result_df = df[result_columns].assign(pred=va_pred.tolist())
        if test:
            i_result_df = i_result_df.assign(est_fold=i_fold)
        return i_result_df

    valid_result_df = pd.concat(
        [_predict_i(df, i_fold, estimator) for i_fold, estimator in zip(train_folds, estimators)],
        axis=0,
        ignore_index=True,
    )
    return valid_result_df
