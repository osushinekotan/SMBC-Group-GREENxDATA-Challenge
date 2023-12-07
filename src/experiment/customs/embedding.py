# type: ignore
import re

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


def split_and_lower(s: str, exceptions: list) -> str:
    if s in exceptions:
        return s
    return " ".join([word.lower() for word in re.split("(?=[A-Z0-9])", s)])[1:]


def replace_other(s: str) -> str:
    parts = s.split("Other")
    # 末尾が 'Other' で終わる場合
    if s.endswith("Other"):
        return "And".join(parts[:-1]) + "AndOther"
    # 末尾が 'Other' で終わらない場合
    return "And".join(parts)


class ColumnsEmbedderV01(BaseEstimator, TransformerMixin):
    """must use train and test concatenated data"""

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 512,
        batch_size: int = 16,
        prompt: str | None = None,
    ) -> None:
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = self.base_prompt

        self.null_value = "[UNK]"
        self.exeptions = [self.null_value]

        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = self.max_seq_length

    @property
    def base_prompt(self):
        prompts = """
This data, recorded on {created_at}, describes a tree with a diameter at breast height (DBH) of {tree_dbh} inches, located on the {curb_loc} in {boroname}.
It is a {spc_common} ({spc_latin}), commonly found in the area of {nta_name} ({nta}). 
The tree is under the stewardship of {steward} and is protected by a {guards} guard. 
Various factors including {problems} influence its condition, and the surrounding sidewalk condition is described as {sidewalk}. 
This information was collected by a {user_type} from the local area of {zip_city}.
        """
        return prompts.replace("\n", "").replace("\t", "")

    def parse_default(self, texts: list[str] | pd.Series) -> list[str]:
        """
        - curb_loc
        - steward
        - guards
        - sidewalk
        - user_type
        - spc_common
        - boroname
        """
        texts = [split_and_lower(text, exceptions=self.exeptions) for text in texts]
        return texts

    def parse_problems(self, texts: list[str] | pd.Series) -> list[str]:
        """
        - problems
        """
        texts = [split_and_lower(replace_other(text), exceptions=self.exeptions) for text in texts]
        return texts

    def make_prompts(self, df: pd.DataFrame) -> pd.Series:
        source_df = df.fillna(self.null_value)
        output_df = pd.DataFrame()
        default_parse_targets = [
            "curb_loc",
            "steward",
            "guards",
            "sidewalk",
            "user_type",
            "spc_common",
        ]
        for col in default_parse_targets:
            output_df[col] = self.parse_default(source_df[col])

        output_df["problems"] = self.parse_problems(source_df["problems"])
        raw_cols = ["created_at", "tree_dbh", "spc_latin", "nta", "zip_city", "nta_name", "boroname"]

        for col in raw_cols:
            output_df[col] = source_df[col].astype(str)

        output_df["prompt"] = output_df.apply(lambda x: self.prompt.format(**x), axis=1)
        return output_df["prompt"].tolist()

    @property
    def feature_names(self) -> list[str]:
        return [f"column_embedding_v01_{i:03}" for i in range(self.model.get_sentence_embedding_dimension())]

    def fit(self, X, y=None):
        output_df = pd.DataFrame({"prompt": self.make_prompts(X)})
        embeddings = self.model.encode(output_df["prompt"].tolist(), show_progress_bar=True, batch_size=self.batch_size)
        self.embeddings_df = pd.DataFrame(
            embeddings, columns=[f"column_embedding_v01_{i:03}" for i in range(embeddings.shape[1])]
        ).assign(prompt=output_df["prompt"].tolist())
        return self

    def transform(self, X, y=None):
        output_df = pd.DataFrame({"prompt": self.make_prompts(X)})
        output_df = pd.merge(output_df, self.embeddings_df, on="prompt", how="left")
        output_df = output_df.drop(columns=["prompt"])
        return output_df[self.feature_names]
