"""
Data loading and preprocessing module for POS tagging.

This module handles loading the NLTK Brown corpus and partitioning data into training/test sets.
"""

import logging
from typing import List, Tuple, Dict
from nltk.corpus import brown

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess POS-tagged sentences from NLTK Brown corpus."""

    @staticmethod
    def clean_tag(tag: str) -> str:
        """
        Strip complex tag suffixes, keeping only the prefix before '+' or '-'.

        For example, if a word is tagged as "NN+JJ" or "VB-VBZ", this method returns
        "NN" and "VB" respectively.

        Args:
            tag: The original POS tag (possibly complex).

        Returns:
            The cleaned tag prefix.
        """
        # Find the first occurrence of '+' or '-' and take the prefix
        for delimiter in ['+', '-']:
            idx = tag.find(delimiter)
            if idx != -1:
                return tag[:idx]
        return tag

    @staticmethod
    def load_brown_corpus(category: str = "news") -> List[List[Tuple[str, str]]]:
        """
        Load tagged sentences from the NLTK Brown corpus.

        Args:
            category: The category of Brown corpus to load (default: "news").

        Returns:
            List of sentences, where each sentence is a list of (word, tag) tuples.

        Raises:
            ValueError: If the corpus cannot be loaded.
        """
        try:
            sentences = brown.tagged_sents(categories=category)
            logger.info(f"Loaded {len(sentences)} sentences from Brown corpus ({category}).")
            return sentences
        except Exception as e:
            logger.error(f"Failed to load Brown corpus: {e}")
            raise ValueError(f"Could not load Brown corpus: {e}") from e

    @staticmethod
    def partition_data(
        sentences: List[List[Tuple[str, str]]], train_split: float = 0.9
    ) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
        """
        Partition sentences into training and test sets.

        Args:
            sentences: List of tagged sentences.
            train_split: Fraction of data for training (default: 0.9 for 90/10 split).

        Returns:
            Tuple of (training_sentences, test_sentences).
        """
        split_idx = int(len(sentences) * train_split)
        train_sents = sentences[:split_idx]
        test_sents = sentences[split_idx:]

        logger.info(f"Partitioned data: {len(train_sents)} training, {len(test_sents)} test.")
        return train_sents, test_sents

    @staticmethod
    def clean_corpus(
        sentences: List[List[Tuple[str, str]]]
    ) -> List[List[Tuple[str, str]]]:
        """
        Clean all tags in a corpus by stripping complex tag suffixes.

        Args:
            sentences: List of tagged sentences with potentially complex tags.
            Each sentence is a list of (word, cleaned_tag) tuples.

        Returns:
            List of sentences with cleaned tags. Same structure as input but with cleaned tags.
        """
        cleaned = []
        for sent in sentences:
            cleaned_sent = [(word, DataLoader.clean_tag(tag)) for word, tag in sent]
            cleaned.append(cleaned_sent)
        logger.debug(f"Cleaned {len(cleaned)} sentences.")
        return cleaned

    @staticmethod
    def prepare_data(
        category: str = "news", train_split: float = 0.9
    ) -> Tuple[
        List[List[Tuple[str, str]]], List[List[Tuple[str, str]]], Dict[str, int], int
    ]:
        """
        Load, clean, and partition Brown corpus data.

        Args:
            category: Brown corpus category.
            train_split: Training/test split ratio.

        Returns:
            Tuple of (training_sentences, test_sentences, tag_set_dict, num_unique_tags).
        """
        raw_sents = DataLoader.load_brown_corpus(category)
        clean_sents = DataLoader.clean_corpus(raw_sents)
        train_sents, test_sents = DataLoader.partition_data(clean_sents, train_split)

        # Collect unique tags
        tag_set: Dict[str, int] = {}
        for sent in train_sents:
            for word, tag in sent:
                tag_set[tag] = tag_set.get(tag, 0) + 1

        num_tags = len(tag_set)
        logger.info(f"Identified {num_tags} unique tags in training set.")
        return train_sents, test_sents, tag_set, num_tags
