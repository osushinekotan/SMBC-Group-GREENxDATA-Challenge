import abc
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class CustomEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        # 除外する属性のリスト
        exclude_attrs = ["parents"]

        if hasattr(obj, "__dict__"):
            # 除外リストに含まれていない属性のみを辞書に含める
            data = {}
            for key, value in obj.__dict__.items():
                if key not in exclude_attrs:
                    # さらなるネストされたオブジェクトを処理する
                    try:
                        json.dumps(value)  # 値が JSON シリアライズ可能かテストする
                        data[key] = value
                    except TypeError:
                        data[key] = str(value)  # シリアライズできない場合は文字列として扱う
            return data

        return json.JSONEncoder.default(self, obj)


class BaseFeatureExtractor(abc.ABC):
    def __init__(self, parents: list["BaseFeatureExtractor"] | None = None):
        self.parents = parents

    def fit(self, input_df: pd.DataFrame, **kwargs):  # type: ignore
        return self

    @abc.abstractmethod
    def transform(self, input_df: pd.DataFrame, **kwargs) -> pd.DataFrame:  # type: ignore
        raise NotImplementedError()

    def load(self, filepath: Path) -> None:
        self.__dict__.update(joblib.load(filepath))

    def save(self, filepath: Path) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value=self.__dict__, filename=filepath)

    @property
    def uid(self) -> str:
        dict_str = json.dumps(self, cls=CustomEncoder, sort_keys=True)
        return hashlib.sha256(dict_str.encode()).hexdigest()
