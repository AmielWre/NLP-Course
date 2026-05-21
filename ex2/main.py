"""
Main orchestration script for POS tagging pipeline.

This script coordinates the full POS tagging pipeline including:
- Data loading and preprocessing
- Training multiple tagger variants
- Evaluation with error rates and confusion matrices
- Comprehensive performance comparison
"""

import logging
import sys
from typing import List, Tuple, Dict, Set
from collections import Counter

from data_loader import DataLoader
from baseline_tagger import BaselineTagger
from hmm_tagger import BigramHMMTagger
from pseudo_words import PseudoWordConverter
from evaluator import Evaluator


# Configure logging
def setup_logging() -> None:
    """Set up professional logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        # time, name, line, level, message
        format="%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def extract_words_and_tags(
    sentences: List[List[Tuple[str, str]]]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Extract words and tags from sentence tuples.

    Args:
        sentences: List of sentences with (word, tag) tuples.

    Returns:
        Tuple of (word_sentences, tag_sentences),
        where each is a list of sentences, and each sentence is a list of words or tags respectively.
        For example, word_sentences[0] is a list of words in the first sentence, and tag_sentences[0] is a list of tags for those words.
    """
    word_sents = []
    tag_sents = []
    for sent in sentences:
        words, tags = zip(*sent) if sent else ([], [])
        word_sents.append(list(words))
        tag_sents.append(list(tags))
    return word_sents, tag_sents


def compute_word_frequencies(
    sentences: List[List[Tuple[str, str]]]
) -> Dict[str, int]:
    """
    Compute word frequencies from sentences.

    Args:
        sentences: List of training sentences.

    Returns:
        Dict mapping words to frequencies.
    """
    freq = Counter()
    for sent in sentences:
        for word, _ in sent:
            freq[word] += 1
    return dict(freq)


def run_baseline_tagger(
    train_sents: List[List[Tuple[str, str]]],
    test_word_sents: List[List[str]],
    test_tag_sents: List[List[str]],
) -> Dict:
    """
    Train and evaluate baseline tagger.

    Args:
        train_sents: Training sentences.
        test_word_sents: Test words.
        test_tag_sents: Test tags (gold standard).

    Returns:
        Dictionary with evaluation results.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80 + "\n" + "PART (b): BASELINE TAGGER - Most Likely Tag" + "\n" + "=" * 80)

    # Train baseline tagger
    baseline = BaselineTagger(unknown_tag="NN")
    baseline.train(train_sents)

    # Get predictions
    pred_sents = baseline.tag_corpus(test_word_sents)

    # Get known words
    known_words = baseline.get_vocabulary()

    # Evaluate
    results = Evaluator.evaluate_with_known_unknown_split(
        test_tag_sents,
        pred_sents,
        test_word_sents,
        known_words,
        tagger_name="Baseline Tagger"
    )

    return results


def run_hmm_mle_tagger(
    train_sents: List[List[Tuple[str, str]]],
    test_word_sents: List[List[str]],
    test_tag_sents: List[List[str]],
    train_word_sents: List[List[str]],
) -> Dict:
    """
    Train and evaluate HMM tagger with MLE (no smoothing).

    Args:
        train_sents: Training sentences.
        test_word_sents: Test words.
        test_tag_sents: Test tags (gold standard).
        train_word_sents: Training words (for known/unknown split).

    Returns:
        Dictionary with evaluation results.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80 + "\n" + "PART (c): BIGRAM HMM TAGGER - MLE (No Smoothing)" + "\n" + "=" * 80)

    # Train HMM tagger
    hmm = BigramHMMTagger(smoothing="none")
    hmm.train(train_sents)

    # Get predictions using Viterbi
    pred_sents = hmm.tag_corpus(test_word_sents)

    # Get known words
    train_words = set()
    for sent in train_word_sents:
        train_words.update(sent)

    # Evaluate
    results = Evaluator.evaluate_with_known_unknown_split(
        test_tag_sents,
        pred_sents,
        test_word_sents,
        train_words,
        tagger_name="HMM Bigram (MLE)",
    )

    return results


def run_hmm_addone_tagger(
    train_sents: List[List[Tuple[str, str]]],
    test_word_sents: List[List[str]],
    test_tag_sents: List[List[str]],
    train_word_sents: List[List[str]],
) -> Dict:
    """
    Train and evaluate HMM tagger with Add-One smoothing.

    Args:
        train_sents: Training sentences.
        test_word_sents: Test words.
        test_tag_sents: Test tags (gold standard).
        train_word_sents: Training words (for known/unknown split).

    Returns:
        Dictionary with evaluation results.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80 + "\n" + "PART (d): BIGRAM HMM TAGGER - Add-One (Laplace) Smoothing" + "\n" + "=" * 80)

    # Train HMM tagger with add-one smoothing
    hmm_smooth = BigramHMMTagger(smoothing="add-one", smoothing_param=1.0)
    hmm_smooth.train(train_sents)

    # Get predictions
    pred_sents = hmm_smooth.tag_corpus(test_word_sents)

    # Get known words
    train_words = set()
    for sent in train_word_sents:
        train_words.update(sent)

    # Evaluate
    results = Evaluator.evaluate_with_known_unknown_split(
        test_tag_sents,
        pred_sents,
        test_word_sents,
        train_words,
        tagger_name="HMM Bigram (Add-One Smoothing)",
    )

    return results


def run_pseudo_words_mle_tagger(
    train_sents: List[List[Tuple[str, str]]],
    test_word_sents: List[List[str]],
    test_tag_sents: List[List[str]],
    train_word_sents: List[List[str]],
    min_freq: int = 5,
) -> Dict[str, float]:
    """
    Train and evaluate HMM tagger with pseudo-words and MLE.

    Args:
        train_sents: Training sentences.
        test_word_sents: Test words.
        test_tag_sents: Test tags (gold standard).
        train_word_sents: Training words (for known/unknown split).
        min_freq: Minimum frequency threshold for pseudo-word replacement.

    Returns:
        Dictionary with evaluation results.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80 + "\n" + "PART (e.1): PSEUDO-WORDS + MLE" + "\n" + "=" * 80)

    # Convert training data with pseudo-words
    pseudo_converter = PseudoWordConverter(min_frequency=min_freq)
    word_freqs = compute_word_frequencies(train_sents)

    # Apply pseudo-words to training data
    train_pseudo_sents = []
    for sent in train_sents:
        pseudo_sent = []
        for word, tag in sent:
            freq = word_freqs.get(word, 0)
            if freq < min_freq:
                pseudo_word = pseudo_converter.get_pseudo_word(word)
                pseudo_sent.append((pseudo_word, tag))
            else:
                pseudo_sent.append((word, tag))
        train_pseudo_sents.append(pseudo_sent)

    # Apply pseudo-words to test data
    test_pseudo_sents = []
    for sent in test_word_sents:
        pseudo_sent = []
        for word in sent:
            freq = word_freqs.get(word, 0)
            if freq < min_freq or word not in set(w for sent in train_word_sents for w in sent):
                pseudo_word = pseudo_converter.get_pseudo_word(word)
                pseudo_sent.append(pseudo_word)
            else:
                pseudo_sent.append(word)
        test_pseudo_sents.append(pseudo_sent)

    # Train HMM on pseudo-word data
    hmm_pseudo = BigramHMMTagger(smoothing="none")
    hmm_pseudo.train(train_pseudo_sents)

    # Get predictions
    pred_sents = hmm_pseudo.tag_corpus(test_pseudo_sents)

    # Get known words (from original training, not pseudo)
    train_words = set()
    for sent in train_word_sents:
        train_words.update(sent)

    # Evaluate
    results = Evaluator.evaluate_with_known_unknown_split(
        test_tag_sents,
        pred_sents,
        test_word_sents,
        train_words,
        tagger_name="Pseudo-Words (MLE)",
    )

    return results


def run_pseudo_words_addone_tagger(
    train_sents: List[List[Tuple[str, str]]],
    test_word_sents: List[List[str]],
    test_tag_sents: List[List[str]],
    train_word_sents: List[List[str]],
    min_freq: int = 5,
) -> Tuple[Dict[str, float], Dict, List[Tuple[str, str, int]]]:
    """
    Train and evaluate HMM tagger with pseudo-words and Add-One smoothing.

    Also generates confusion matrix.

    Args:
        train_sents: Training sentences.
        test_word_sents: Test words.
        test_tag_sents: Test tags (gold standard).
        train_word_sents: Training words (for known/unknown split).
        min_freq: Minimum frequency threshold for pseudo-word replacement.

    Returns:
        Tuple of (results, confusion_matrix, top_errors).
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80 + "\n" + "PART (e.2): PSEUDO-WORDS + Add-One Smoothing (Final System)" + "\n" + "=" * 80)

    # Convert training data with pseudo-words
    pseudo_converter = PseudoWordConverter(min_frequency=min_freq)
    word_freqs = compute_word_frequencies(train_sents)

    # Apply pseudo-words to training data
    train_pseudo_sents = []
    for sent in train_sents:
        pseudo_sent = []
        for word, tag in sent:
            freq = word_freqs.get(word, 0)
            if freq < min_freq:
                pseudo_word = pseudo_converter.get_pseudo_word(word)
                pseudo_sent.append((pseudo_word, tag))
            else:
                pseudo_sent.append((word, tag))
        train_pseudo_sents.append(pseudo_sent)

    # Apply pseudo-words to test data
    test_pseudo_sents = []
    for sent in test_word_sents:
        pseudo_sent = []
        for word in sent:
            freq = word_freqs.get(word, 0)
            if freq < min_freq or word not in set(w for ws in train_word_sents for w in ws):
                pseudo_word = pseudo_converter.get_pseudo_word(word)
                pseudo_sent.append(pseudo_word)
            else:
                pseudo_sent.append(word)
        test_pseudo_sents.append(pseudo_sent)

    # Train HMM on pseudo-word data with add-one smoothing
    hmm_pseudo_smooth = BigramHMMTagger(smoothing="add-one", smoothing_param=1.0)
    hmm_pseudo_smooth.train(train_pseudo_sents)

    # Get predictions
    pred_sents = hmm_pseudo_smooth.tag_corpus(test_pseudo_sents)

    # Get known words
    train_words = set()
    for sent in train_word_sents:
        train_words.update(sent)

    # Evaluate
    results = Evaluator.evaluate_with_known_unknown_split(
        test_tag_sents,
        pred_sents,
        test_word_sents,
        train_words,
        tagger_name="Pseudo-Words (Add-One Smoothing)",
    )

    # Build confusion matrix
    tag_list = sorted(list(hmm_pseudo_smooth.tags))
    matrix, top_errors = Evaluator.build_and_analyze_confusion_matrix(
        test_tag_sents,
        pred_sents,
        tag_list,
        tagger_name="Pseudo-Words (Add-One Smoothing)",
        top_errors=10,
    )

    return results, matrix, top_errors


def print_comparison(
    baseline_res: Dict,
    hmm_mle_res: Dict,
    hmm_addone_res: Dict,
    pseudo_mle_res: Dict,
    pseudo_addone_res: Dict,
) -> None:
    """
    Print comprehensive comparison of all taggers.

    Args:
        baseline_res: Baseline tagger results.
        hmm_mle_res: HMM MLE results.
        hmm_addone_res: HMM Add-One results.
        pseudo_mle_res: Pseudo-words MLE results.
        pseudo_addone_res: Pseudo-words Add-One results.
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80 + "\n" + "COMPREHENSIVE PERFORMANCE COMPARISON" + "\n" + "=" * 80)

    results_list = [baseline_res, hmm_mle_res, hmm_addone_res, pseudo_mle_res, pseudo_addone_res]

    # Print table header
    logger.info(
        f"\n{'Tagger Name':<35} {'Known ER':<12} {'Unk ER':<12} {'Overall ER':<12}"
    )
    logger.info("-" * 80)

    # Print results
    for res in results_list:
        known_er = res.get("known_error_rate", res.get("overall_error_rate", 0.0))
        unknown_er = res.get("unknown_error_rate", 0.0)
        overall_er = res.get("overall_error_rate", 0.0)

        logger.info(
            f"{res['name']:<35} {known_er:<12.4f} {unknown_er:<12.4f} {overall_er:<12.4f}"
        )

    logger.info("-" * 80)
    logger.info("Legend: ER = Error Rate (lower is better)")


def main() -> None:
    """Main orchestration function for the POS tagging pipeline."""
    logger = logging.getLogger(__name__)

    # ===== PART (a): DATA LOADING =====
    logger.info("\n" + "=" * 80 + "\n" + "PART (a): DATA LOADING AND PARTITIONING" + "\n" + "=" * 80)

    train_sents, test_sents, tag_set, num_tags = DataLoader.prepare_data(
        category="news", train_split=0.9
    )
    logger.info(f"Training sentences: {len(train_sents)}")
    logger.info(f"Test sentences: {len(test_sents)}")
    logger.info(f"Unique tags in training: {num_tags}")

    # Extract words and tags
    train_word_sents, train_tag_sents = extract_words_and_tags(train_sents)
    test_word_sents, test_tag_sents = extract_words_and_tags(test_sents)

    logger.info(f"Total training words: {sum(len(s) for s in train_word_sents)}")
    logger.info(f"Total test words: {sum(len(s) for s in test_word_sents)}")

    # ===== PART (b): BASELINE TAGGER =====
    baseline_results = run_baseline_tagger(
        train_sents, test_word_sents, test_tag_sents
    )

    # ===== PART (c): HMM MLE =====
    hmm_mle_results = run_hmm_mle_tagger(
        train_sents, test_word_sents, test_tag_sents, train_word_sents
    )

    # ===== PART (d): HMM ADD-ONE =====
    hmm_addone_results = run_hmm_addone_tagger(
        train_sents, test_word_sents, test_tag_sents, train_word_sents
    )

    # ===== PART (e.1): PSEUDO-WORDS MLE =====
    pseudo_mle_results = run_pseudo_words_mle_tagger(
        train_sents, test_word_sents, test_tag_sents, train_word_sents, min_freq=5
    )

    # ===== PART (e.2): PSEUDO-WORDS ADD-ONE (FINAL SYSTEM) =====
    pseudo_addone_results, confusion_matrix, top_errors = run_pseudo_words_addone_tagger(
        train_sents, test_word_sents, test_tag_sents, train_word_sents, min_freq=5
    )

    # ===== COMPARISON =====
    print_comparison(
        baseline_results,
        hmm_mle_results,
        hmm_addone_results,
        pseudo_mle_results,
        pseudo_addone_results,
    )

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_logging()
    main()
