import logging
from pathlib import Path

import joblib
import pandas as pd

from src.experiment.base import BaseFeatureExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_extractor(
    input_df: pd.DataFrame,
    extractor: BaseFeatureExtractor,
    dirpath: Path,
    fit: bool,
    cache: bool,
) -> pd.DataFrame:
    # parent がある場合は、その適用結果を concat する
    exist_parents = False
    if hasattr(extractor, "parents"):
        if extractor.parents:
            exist_parents = True

    if exist_parents:
        parent_df = run_extractors(
            input_df=input_df,
            extractors=list(extractor.parents),  # type: ignore
            dirpath=dirpath,
            fit=fit,
            prefix="",  # parent は prefix なし
        )
        tmp_df = pd.concat([input_df, parent_df], axis=1)
    else:
        tmp_df = input_df

    name = f"{extractor.__class__.__name__}_{extractor.uid}"
    parent_dir = dirpath / name
    logger.info(f"<{name}>")

    parent_dir.mkdir(parents=True, exist_ok=True)
    filepath_for_dict = parent_dir / "__dict__.pkl"
    filepath_for_df = parent_dir / "output.pkl"

    if not fit:
        # test dataframe に対しては fit しない
        extractor.load(filepath=filepath_for_dict)
        transformed_df = extractor.transform(tmp_df)
        return transformed_df

    if cache and fit and (filepath_for_df.is_file()):
        # cache を使いたいかつ test ではないかつファイルがある時
        transformed_df = joblib.load(filepath_for_df)
        return transformed_df

    extractor.fit(tmp_df)
    extractor.save(filepath=filepath_for_dict)

    transformed_df = extractor.transform(tmp_df)
    joblib.dump(transformed_df, filepath_for_df)

    return transformed_df


def run_extractors(
    input_df: pd.DataFrame,
    extractors: list[BaseFeatureExtractor],
    dirpath: Path = Path("./stores/features/"),
    fit: bool = False,
    cache: bool = True,
    prefix: str = "f_",
) -> pd.DataFrame:
    output_df = pd.concat(
        [run_extractor(input_df, extractor, dirpath, fit, cache) for extractor in list(extractors)],
        axis=1,
    ).reset_index(drop=True)

    return output_df.add_prefix(prefix)
