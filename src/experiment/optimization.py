import numpy as np
from sklearn.metrics import f1_score


def _update_pred(proba_matrix: np.ndarray, label_index: int, threshold: float, maxsize: int = 1) -> np.ndarray:
    mask = proba_matrix[:, label_index] >= threshold
    proba_matrix[mask, label_index] = maxsize
    return proba_matrix


def find_optimal_threshold_for_label(
    proba_matrix: np.ndarray, true_labels: list | np.ndarray, label_indices: list[int]
) -> tuple[dict, np.ndarray]:
    maxsize = 100
    proba_matrix_ = proba_matrix.copy()
    result_dict = {}
    for label_index in label_indices:
        maxsize -= 1
        best_threshold = 0
        best_f1 = 0
        for threshold in np.linspace(0, 1, 100):
            tmp_proba_matrix_ = proba_matrix_.copy()
            updated_proba = _update_pred(
                tmp_proba_matrix_, label_index=label_index, threshold=threshold, maxsize=maxsize
            )
            preds = np.argmax(updated_proba, axis=1)
            f1 = f1_score(true_labels, preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        proba_matrix_ = _update_pred(proba_matrix, label_index=label_index, threshold=best_threshold, maxsize=maxsize)
        result_dict[label_index] = {"threshold": best_threshold, "f1": best_f1}

    pred_label = np.argmax(proba_matrix_, axis=1)
    return result_dict, pred_label


def decode_label(proba_matrix: np.ndarray, thresholds: dict) -> np.ndarray:
    maxsize = 100
    proba_matrix_ = proba_matrix.copy()
    for label_index, res in thresholds.items():
        maxsize -= 1
        threshold = res["threshold"]
        proba_matrix_ = _update_pred(proba_matrix_, label_index=label_index, threshold=threshold, maxsize=maxsize)

    pred_label = np.argmax(proba_matrix_, axis=1)
    return pred_label
