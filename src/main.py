import logging
import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run experiment notebook."""

    env = os.environ.copy()
    env["OVERRIDES"] = cfg.experiment_name

    compiled_notebook_path = Path(cfg.paths.output_dir) / f"{cfg.notebook}.ipynb"
    logger.info(f"Overrides: {env['OVERRIDES']}, Notebook: {cfg.notebook}")

    command = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--output",
        compiled_notebook_path.as_posix(),
        "--ExecutePreprocessor.timeout=None",
        "--ExecutePreprocessor.allow_errors=True",
        (Path(cfg.paths.notebooks_dir) / f"{cfg.notebook}.ipynb").as_posix(),
    ]  # Unsure how to set environment variables using nbconvert.ExecutePreprocessor

    try:
        subprocess.run(command, env=env, check=True)
        logger.info("Finished running notebook successfully! 🎉")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run notebook 😭: {e}")
        raise


if __name__ == "__main__":
    main()
