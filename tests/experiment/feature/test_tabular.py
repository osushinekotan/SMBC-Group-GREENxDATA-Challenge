# type: ignore
import pandas as pd
import pytest

from src.experiment.feature.tabular import TargetEncoder


# テスト用のサンプルデータ
@pytest.fixture
def sample_data():
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": ["A", "B", "A", "B", "A"],
        "target": [1.2, 2.3, 3.1, 4.5, 5.0],
        "fold": [0, 1, 0, 1, -1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def encoder():
    group_keys = ["feature2"]
    target_value = "target"
    agg_methods = ["mean", "sum"]
    fold = "fold"
    return TargetEncoder(group_keys, target_value, agg_methods, fold)


def test_fold_minus_one_encoding_with_hardcoded_values(encoder, sample_data):
    encoder.fit(sample_data)
    transformed = encoder.transform(sample_data)

    # ハードコードされた正解の値（これらは実際の値に基づいて設定する必要があります）
    expected_mean_A = 3.1  # 'A'グループの平均値の期待値
    expected_sum_A = 9.3  # 'A'グループの合計値の期待値

    mean_val = transformed.loc[4, "te_mean_target_grpby_feature2"]
    sum_val = transformed.loc[4, "te_sum_target_grpby_feature2"]

    assert mean_val == pytest.approx(expected_mean_A)
    assert sum_val == pytest.approx(expected_sum_A)
