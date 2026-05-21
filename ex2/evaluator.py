"""
Evaluation metrics and analysis for POS taggers.

This module computes error rates, accuracy metrics, and confusion matrices for
evaluating POS tagging performance.
"""

import logging
from typing import List, Tuple, Dict, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Static class to evaluate POS tagger predictions against gold standard tags.

    Computes accuracy, error rates, and confusion matrices with analysis
    of the most frequent error pairs.
    """

    @staticmethod
    def separate_known_unknown(
        gold_tags: List[str],
        pred_tags: List[str],
        known_words: Set[str],
        words: List[str],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Separate known and unknown word predictions.

        Args:
            gold_tags: Gold standard tags.
            pred_tags: Predicted tags.
            known_words: Set of words in training vocabulary.
            words: List of words corresponding to tags.

        Returns:
            Tuple of (gold_known, pred_known, gold_unknown, pred_unknown).
        """
        gold_known, pred_known = [], []
        gold_unknown, pred_unknown = [], []

        for word, gold, pred in zip(words, gold_tags, pred_tags):
            if word in known_words:
                gold_known.append(gold)
                pred_known.append(pred)
            else:
                gold_unknown.append(gold)
                pred_unknown.append(pred)

        return gold_known, pred_known, gold_unknown, pred_unknown

    @staticmethod
    def compute_accuracy(gold_tags: List[str], pred_tags: List[str]) -> float:
        """
        Compute accuracy (fraction of correctly predicted tags).

        Args:
            gold_tags: Gold standard tags.
            pred_tags: Predicted tags.

        Returns:
            Accuracy in range [0, 1].
        """
        if not gold_tags:
            return 0.0
        correct = sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
        return correct / len(gold_tags)

    @staticmethod
    def compute_error_rate(gold_tags: List[str], pred_tags: List[str]) -> float:
        """
        Compute error rate (1 - accuracy).

        Args:
            gold_tags: Gold standard tags.
            pred_tags: Predicted tags.

        Returns:
            Error rate in range [0, 1].
        """
        return 1.0 - Evaluator.compute_accuracy(gold_tags, pred_tags)

    @staticmethod
    def build_confusion_matrix(
        gold_tags: List[str], pred_tags: List[str], tag_list: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Build a confusion matrix.

        The entry (i, j) is the count of tags where the true tag is i and
        the predicted tag is j.

        Args:
            gold_tags: Gold standard tags.
            pred_tags: Predicted tags.
            tag_list: List of all possible tags (defines matrix order).

        Returns:
            Confusion matrix as nested Dict[true_tag][pred_tag] = count.
        """
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for gold, pred in zip(gold_tags, pred_tags):
            matrix[gold][pred] += 1

        return matrix

    @staticmethod
    def analyze_confusion_matrix(
        matrix: Dict[str, Dict[str, int]], top_errors: int = 10
    ) -> List[Tuple[str, str, int]]:
        """
        Analyze confusion matrix to find most frequent error pairs.

        Args:
            matrix: Confusion matrix from build_confusion_matrix.
            top_errors: Number of top error pairs to return.

        Returns:
            List of tuples (true_tag, pred_tag, count), sorted by count descending.
        """
        errors: List[Tuple[str, str, int]] = []

        for true_tag, pred_dict in matrix.items():
            for pred_tag, count in pred_dict.items():
                if true_tag != pred_tag:  # Only count as error if prediction differs
                    errors.append((true_tag, pred_tag, count))

        # Sort by count descending
        errors.sort(key=lambda x: x[2], reverse=True)
        return errors[:top_errors]


    @staticmethod
    def evaluate_with_known_unknown_split(
        gold_sentences: List[List[str]],
        pred_sentences: List[List[str]],
        word_sentences: List[List[str]],
        known_words: Set[str],
        tagger_name: str = "Tagger",
    ) -> Dict:
        """
        Comprehensive evaluation with known/unknown split.

        Args:
            gold_sentences: Gold truth standard tagged sentences (tags).
            pred_sentences: Predicted tagged sentences (tags).
            word_sentences: Original word sentences (for known/unknown classification).
            known_words: Set of words in training vocabulary.
            tagger_name: Name of tagger for logging.

        Returns:
            Dictionary with detailed evaluation metrics including known/unknown split.
        """
        # Flatten
        gold_tags_flat = [tag for sent in gold_sentences for tag in sent]
        pred_tags_flat = [tag for sent in pred_sentences for tag in sent]
        words_flat = [word for sent in word_sentences for word in sent]

        # Separate known and unknown
        gold_known, pred_known, gold_unknown, pred_unknown = Evaluator.separate_known_unknown(
            gold_tags_flat, pred_tags_flat, known_words, words_flat
        )
        logger.debug(f"Separated known and unknown words: {len(gold_known)} known, {len(gold_unknown)} unknown.")

        # Compute error rates
        known_error = Evaluator.compute_error_rate(gold_known, pred_known) if gold_known else 0.0
        unknown_error = Evaluator.compute_error_rate(gold_unknown, pred_unknown) if gold_unknown else 0.0
        overall_error = Evaluator.compute_error_rate(gold_tags_flat, pred_tags_flat)

        # Compute accuracies
        known_accuracy = 1.0 - known_error
        unknown_accuracy = 1.0 - unknown_error
        overall_accuracy = 1.0 - overall_error

        logger.info("\n" + "-" * 25 + f" {tagger_name} Performance " + "-" * 25)
        logger.info(f"  Known Words Error Rate:   {known_error:.4f} ({len(gold_known)} words)")
        logger.info(f"  Unknown Words Error Rate: {unknown_error:.4f} ({len(gold_unknown)} words)")
        logger.info(f"  Overall Error Rate:       {overall_error:.4f}")
        logger.info(f"  " + "-" * 20)
        logger.info(f"  Known Words Accuracy:     {known_accuracy:.4f}")
        logger.info(f"  Unknown Words Accuracy:   {unknown_accuracy:.4f}")
        logger.info(f"  Overall Accuracy:         {overall_accuracy:.4f}")

        return {
            "name": tagger_name,
            "known_error_rate": known_error,
            "unknown_error_rate": unknown_error,
            "overall_error_rate": overall_error,
            "known_accuracy": known_accuracy,
            "unknown_accuracy": unknown_accuracy,
            "overall_accuracy": overall_accuracy,
            "num_known": len(gold_known),
            "num_unknown": len(gold_unknown),
            "num_total": len(gold_tags_flat),
        }

    @staticmethod
    def build_and_analyze_confusion_matrix(
        gold_sentences: List[List[str]],
        pred_sentences: List[List[str]],
        tag_list: List[str],
        tagger_name: str = "Tagger",
        top_errors: int = 10,
    ) -> Tuple[Dict[str, Dict[str, int]], List[Tuple[str, str, int]]]:
        """
        Build confusion matrix and analyze error patterns.

        Args:
            gold_sentences: Gold standard tagged sentences.
            pred_sentences: Predicted tagged sentences.
            tag_list: Sorted list of all possible tags.
            tagger_name: Name of tagger for logging.
            top_errors: Number of top error pairs to report.

        Returns:
            Tuple of (confusion_matrix, top_error_pairs).
        """
        # Flatten
        gold_tags_flat = [tag for sent in gold_sentences for tag in sent]
        pred_tags_flat = [tag for sent in pred_sentences for tag in sent]

        # Build matrix
        matrix = Evaluator.build_confusion_matrix(gold_tags_flat, pred_tags_flat, tag_list)

        # Analyze
        top_errors_list = Evaluator.analyze_confusion_matrix(matrix, top_errors)

        logger.info(f"\n{tagger_name} Confusion Matrix Analysis:")
        logger.info(f"  Top {top_errors} Most Frequent Error Pairs:")
        for i, (true_tag, pred_tag, count) in enumerate(top_errors_list, 1):
            logger.info(f"    {i}. True={true_tag:6s} Pred={pred_tag:6s} Count={count:5d}")

        return matrix, top_errors_list
