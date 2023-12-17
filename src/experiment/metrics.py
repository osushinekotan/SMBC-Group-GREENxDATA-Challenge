import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def macro_f1_from_proba(y_true: np.ndarray | list | pd.Series, y_pred: np.ndarray | list) -> float:
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="macro")
