"""Centralized configuration for Exercise 4 dependency parsing experiments."""

from pathlib import Path


# Project paths.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
EX4_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = EX4_DIR / "data"
CHECKPOINT_DIR: Path = EX4_DIR / "checkpoints"
RESULTS_DIR: Path = EX4_DIR / "results"

# Independent execution flags.
RUN_PART_1_PERCEPTRON: bool = True
RUN_PART_2_ATTENTION: bool = True
FORCE_RETRAIN: bool = False  # If True, ignore existing checkpoints and re-run training/evaluation.

# Reproducibility and hardware.
RANDOM_SEED: int = 42
DEVICE: str = "cpu"

# Dataset configuration.
NLTK_CORPUS_NAME: str = "dependency_treebank"
TEST_SPLIT_RATIO: float = 0.10
ROOT_WORD: str = "ROOT"
ROOT_POS: str = "ROOT"

# Structured perceptron configuration.
PERCEPTRON_EPOCHS: int = 2
PERCEPTRON_LEARNING_RATE: float = 1.0
PERCEPTRON_CHECKPOINT: Path = CHECKPOINT_DIR / "averaged_perceptron.pkl"

# Attention-based parser configuration.
BERT_MODEL_NAME: str = "bert-base-uncased"
ATTENTION_LAYER_SETTINGS: tuple[tuple[int, str], ...] = (
    (0, "mean"),
    (5, "mean"),
    (11, "mean"),
)
ATTENTION_CHECKPOINT: Path = CHECKPOINT_DIR / "attention_results.pkl"
MAX_BERT_WORDPIECE_LENGTH: int = 510

# Output configuration.
LOG_LEVEL: str = "DEBUG"
LOG_USE_COLORS: bool = True
LOG_EVERY_SENTENCES: int = 150
PLOT_RESULTS: bool = True
METRICS_PLOT_PATH: Path = RESULTS_DIR / "ex4_uas_comparison.html"
