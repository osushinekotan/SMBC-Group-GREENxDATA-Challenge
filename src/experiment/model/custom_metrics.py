# type: ignore
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def lgb_macro_auc(y_true, y_pred):
    auc = roc_auc_score(y_true=y_true, y_score=y_pred, multi_class="ovr")
    return "macro_auc", auc, True


def lgb_macro_f1(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true=y_true, y_pred=y_pred_label, average="macro")
    return "macro_f1", f1, True


def lgb_py_minus_macro_f1(y_pred, data):
    y_true = data.get_label()
    score = f1_score(np.argmax(y_pred, axis=1), y_true, average="macro")
    return "custom", -score, False


def xgb_macro_f1(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true=y_true, y_pred=y_pred_label, average="macro")
    return f1
