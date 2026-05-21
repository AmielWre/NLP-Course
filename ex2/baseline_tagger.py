"""
Baseline POS tagger using Maximum Likelihood Estimation.

This module implements a simple baseline tagger that selects the most likely tag
for each word based on the training set. Unknown words are tagged as "NN".
"""

import logging
from typing import List, Dict, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class BaselineTagger:
    """
    Baseline tagger using Maximum Likelihood Estimation (MLE).

    For each word in the training set, stores the most likely tag. Unknown words
    default to "NN".
    """

    def __init__(self, unknown_tag: str = "NN") -> None:
        """
        Initialize the baseline tagger.

        Args:
            unknown_tag: Tag to assign to unknown words (default: "NN").
        """
        self.unknown_tag = unknown_tag
        # For each word, store a dictionary of tag counts: word_tag_counts[word][tag] = count
        self.word_tag_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # For each word, store the most likely tag based on training counts
        self.word_best_tag: Dict[str, str] = {}
        logger.debug(f"Initialized BaselineTagger with unknown_tag='{unknown_tag}'.")

    def train(self, sentences: List[List[Tuple[str, str]]]) -> None:
        """
        Train the baseline tagger using MLE to compute p(tag|word).

        Args:
            sentences: List of training sentences, each with (word, tag) tuples.
        """
        logger.info("Training baseline tagger...")

        for sent in sentences:
            for word, tag in sent:
                self.word_tag_counts[word][tag] += 1

        # Select the most likely tag for each word
        for word, tag_counts in self.word_tag_counts.items():
            best_tag = max(tag_counts, key=tag_counts.get)
            self.word_best_tag[word] = best_tag

        num_words = len(self.word_best_tag)
        logger.info(f"Trained on {num_words} unique words.")

    def tag(self, word: str) -> str:
        """
        Get the most likely tag for a word.

        Args:
            word: A single word to tag.

        Returns:
            The most likely tag for the word, or unknown_tag if not in training set.
        """
        return self.word_best_tag.get(word, self.unknown_tag)

    def tag_sentence(self, sentence: List[str]) -> List[str]:
        """
        Tag all words in a sentence. Calls tag() for each word.

        Args:
            sentence: List of words.

        Returns:
            List of predicted tags, one per word.
            Each tag corresponds to the word in the same position in the sentence.
            For example, if sentence[0] = "The", then the returned list will have a
            corresponding entry like "DT" at index 0.
        """
        return [self.tag(word) for word in sentence]

    def tag_corpus(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Tag all sentences in a corpus. Calls tag_sentence() for each sentence.

        Args:
            sentences: List of sentences, each a list of words.

        Returns:
            List of tagged sentences, each with predicted tags.
            Each tag corresponds to the word in the same position in the sentence.
            For example, if sentences[0] = ["The", "cat"], then the returned list will have a
            corresponding entry like ["DT", "NN"].
        """
        return [self.tag_sentence(sent) for sent in sentences]

    def get_vocabulary(self) -> Set[str]:
        """
        Get the set of words in the training vocabulary.

        Returns:
            Set of known words.
        """
        return set(self.word_best_tag.keys())
