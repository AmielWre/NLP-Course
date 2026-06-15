"""Unified entry point for Exercise 4 dependency parsing.

Function Call Tree:
    main()
        -> dataset.prepare_dependency_data()
        -> run_part_1()
            -> models.load_or_train_perceptron()
            -> AveragedPerceptronParser.evaluate()
        -> run_part_2()
            -> models.load_or_evaluate_attention()
        -> utils.plot_uas_results()
        -> _print_metrics()

Cross-module Communication:
    config.py controls all paths, hyperparameters, flags, and model choices.
    dataset.py provides shared train/test partitions. models.py executes both
    parsing pipelines. utils.py handles checkpoint validation, UAS, CLE decoding,
    and plotting.
"""

from __future__ import annotations

from typing import Dict, Sequence

from ex4 import config
from ex4.dataset import Sentence, prepare_dependency_data
from ex4.models import AveragedPerceptronParser, load_or_evaluate_attention, load_or_train_perceptron
from ex4.utils import get_logger, plot_uas_results


LOGGER = get_logger(__name__)


def run_part_1(train_sentences: Sequence[Sentence], test_sentences: Sequence[Sentence]) -> Dict[str, float]:
    """Run the averaged perceptron MST parser pipeline.

    Args:
        train_sentences: Training sentence dictionaries.
        test_sentences: Static test sentence dictionaries.

    Returns:
        Mapping containing the perceptron mean UAS metric.

    Raises:
        ValueError: If evaluation receives no test sentences.

    Notes:
        Training is skipped when a checkpoint with the same config signature
        exists. The checkpoint stores the averaged sparse weight vector.
    """
    parser: AveragedPerceptronParser = load_or_train_perceptron(train_sentences)
    return {"perceptron_mst": parser.evaluate(test_sentences)}


def run_part_2(test_sentences: Sequence[Sentence]) -> Dict[str, float]:
    """Run the BERT attention-based parsing pipeline.

    Args:
        test_sentences: Static test sentence dictionaries.

    Returns:
        Mapping from attention layer setting to mean UAS.

    Raises:
        ImportError: If Hugging Face dependencies are missing and no checkpoint exists.

    Notes:
        This function performs no training. It downloads/loads BERT only when
        Part 2 is enabled and no valid cached metrics checkpoint exists.
    """
    return load_or_evaluate_attention(test_sentences)


def main() -> None:
    """Execute enabled Exercise 4 pipelines and report UAS metrics.

    Raises:
        ImportError: If enabled dependencies are not installed.
        OSError: If checkpoint or plot outputs cannot be written.

    Notes:
        main.py deliberately acts as an orchestrator. All execution behavior
        comes from Boolean flags and constants in config.py.
    """
    metrics: Dict[str, float] = {}  # Mapping from experiment label to mean UAS metric.
    if not (config.RUN_PART_1_PERCEPTRON or config.RUN_PART_2_ATTENTION):
        LOGGER.info("Both exercise parts are disabled in config.py.")
        return

    try:
        LOGGER.info("Exercise 4 run started.")
        train_sentences, test_sentences = prepare_dependency_data()
        if config.RUN_PART_1_PERCEPTRON:
            LOGGER.info("Part 1 enabled: perceptron MST parser.")
            metrics.update(run_part_1(train_sentences, test_sentences))
        if config.RUN_PART_2_ATTENTION:
            LOGGER.info("Part 2 enabled: attention-based parser.")
            metrics.update(run_part_2(test_sentences))

        plot_uas_results(metrics)
        _print_metrics(metrics)
    except Exception:
        LOGGER.exception("Exercise 4 run failed.")
        raise


def _print_metrics(metrics: Dict[str, float]) -> None:
    """Print final UAS metrics in a compact side-by-side format.

    Args:
        metrics: Mapping from experiment label to mean UAS.
    """
    LOGGER.info("Exercise 4 Mean UAS")
    for label, value in metrics.items():
        LOGGER.info("%-28s %.4f", label, value)


if __name__ == "__main__":
    main()
