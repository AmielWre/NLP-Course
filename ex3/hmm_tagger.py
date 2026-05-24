"""
Bigram HMM POS tagger with Viterbi algorithm in log10 space.

This module implements a bigram Hidden Markov Model for POS tagging with the Viterbi
algorithm running in log-space (base-10) to prevent numerical underflow.
"""

import logging
import math
from typing import List, Dict, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class BigramHMMTagger:
    """
    Bigram HMM POS tagger with Viterbi algorithm in log10 space.

    Attributes:
        transitions: Dict[prev_tag][current_tag] = transition probability = p(current_tag | prev_tag)
        emissions: Dict[tag][word] = emission probability = p(word | tag)
        log_transitions: Log10 version of transitions
        log_emissions: Log10 version of emissions
        log_initial: Log10 initial state probabilities (probability of tag at position 0)
        tags: Set of all POS tags
        words: Set of all words in vocabulary
        smoothing: Type of smoothing ("none" or "add-one")
        smoothing_param: Smoothing parameter (used for add-one smoothing)
    """

    def __init__(self, smoothing: str = "none", smoothing_param: float = 1.0) -> None:
        """
        Initialize the HMM tagger.

        Args:
            smoothing: Type of smoothing ("none" or "add-one").
            smoothing_param: Smoothing parameter (used for add-one smoothing).
        """
        self.transitions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.emissions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.log_transitions: Dict[str, Dict[str, float]] = {}
        self.log_emissions: Dict[str, Dict[str, float]] = {}
        self.log_initial: Dict[str, float] = {}

        self.tags: Set[str] = set()
        self.words: Set[str] = set()
        self.smoothing = smoothing
        self.smoothing_param = smoothing_param
        logger.debug(f"Initialized BigramHMMTagger with smoothing={smoothing}.")

    def _compute_mle_probs(
        self,
        trans_counts: Dict[str, Dict[str, int]],
        emit_counts: Dict[str, Dict[str, int]],
        initial_counts: Dict[str, int],
    ) -> None:
        """
        Compute transition and emission probabilities using MLE from pre-computed counts.

        Args:
            trans_counts: Transition counts.
            emit_counts: Emission counts.
            initial_counts: Initial state counts.
        """
        # Initial probabilities
        total_initial = sum(initial_counts.values())
        for tag in self.tags:
            if total_initial > 0:
                prob = initial_counts.get(tag, 0) / total_initial
            else:
                prob = 1.0 / len(self.tags) if self.tags else 1.0
            self.log_initial[tag] = math.log10(prob) if prob > 0 else -float('inf')

        # Transition probabilities
        for prev_tag in self.tags:
            denom = sum(trans_counts[prev_tag].values())  # Total count of transitions from prev_tag
            if denom == 0:
                # Use uniform distribution for all tags if no transitions observed from prev_tag -> p(tagA | prev_tag) = p(tagB | prev_tag) = 1 / num_tags
                for curr_tag in self.tags:
                    self.transitions[prev_tag][curr_tag] = 1.0 / len(self.tags)
            else:
                for curr_tag in self.tags:
                    count = trans_counts[prev_tag].get(curr_tag, 0)
                    prob = count / denom
                    self.transitions[prev_tag][curr_tag] = prob

        # Emission probabilities
        for tag in self.tags:
            denom = sum(emit_counts[tag].values())  # Total count of emissions from tag
            if denom == 0:
                # Use uniform distribution for all words if no emissions observed from tag -> p(wordA | tag) = p(wordB | tag) = 1 / num_words
                for word in self.words:
                    self.emissions[tag][word] = 1.0 / len(self.words)
            else:
                for word in self.words:
                    count = emit_counts[tag].get(word, 0)
                    prob = count / denom
                    self.emissions[tag][word] = prob

    def _compute_add_one_probs(
        self,
        trans_counts: Dict[str, Dict[str, int]],
        emit_counts: Dict[str, Dict[str, int]],
        initial_counts: Dict[str, int],
        tag_counts: Dict[str, int],
    ) -> None:
        """
        Compute probabilities using Add-One (Laplace) smoothing.

        Args:
            trans_counts: Transition counts.
            emit_counts: Emission counts.
            initial_counts: Initial state counts.
            tag_counts: Total count of each tag.
        """
        logger.debug("Computing probabilities with Add-One smoothing.")

        # Initial probabilities (no smoothing for initial)
        total_initial = sum(initial_counts.values())
        for tag in self.tags:
            if total_initial > 0:
                prob = initial_counts.get(tag, 0) / total_initial
            else:
                prob = 1.0 / len(self.tags) if self.tags else 1.0
            self.log_initial[tag] = math.log10(prob) if prob > 0 else -float('inf')

        # Transition probabilities with add-one smoothing
        num_tags = len(self.tags)
        for prev_tag in self.tags:
            denom = sum(trans_counts[prev_tag].values()) + num_tags  # Note the +num_tags for add-one smoothing so the probabillity sums to 1
            for curr_tag in self.tags:
                count = trans_counts[prev_tag].get(curr_tag, 0)
                prob = (count + 1.0) / denom
                self.transitions[prev_tag][curr_tag] = prob

        # Emission probabilities with add-one smoothing
        vocab_size = len(self.words)
        for tag in self.tags:
            denom = sum(emit_counts[tag].values()) + vocab_size  # Note the +vocab_size for add-one smoothing so the probabillity sums to 1
            for word in self.words:
                count = emit_counts[tag].get(word, 0)
                prob = (count + 1.0) / denom
                self.emissions[tag][word] = prob

    def train(self, sentences: List[List[Tuple[str, str]]]) -> None:
        """
        Train the HMM tagger.

        Args:
            sentences: List of training sentences with (word, tag) tuples.
        """
        logger.info(f"Training BigramHMMTagger (smoothing={self.smoothing})...")

        # Collect counts first
        # 
        trans_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # trans_counts[from_tag][to_tag] = raw count of appearances 'from_tag' followed by 'to_tag'
        emit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))  # emit_counts[tag][word] = raw count of appearances of 'word' emitted by 'tag'
        initial_counts: Dict[str, int] = defaultdict(int)  # Count of tags at position 0
        tag_counts: Dict[str, int] = defaultdict(int)  # Total count of each tag (for add-one smoothing)

        for sent in sentences:
            if not sent:
                continue
            first_word, first_tag = sent[0]
            initial_counts[first_tag] += 1
            emit_counts[first_tag][first_word] += 1
            tag_counts[first_tag] += 1
            self.words.add(first_word)
            self.tags.add(first_tag)

            for i in range(1, len(sent)):
                prev_word, prev_tag = sent[i - 1]
                curr_word, curr_tag = sent[i]
                trans_counts[prev_tag][curr_tag] += 1
                emit_counts[curr_tag][curr_word] += 1
                tag_counts[curr_tag] += 1
                self.words.add(curr_word)
                self.tags.add(curr_tag)

        # Compute probabilities based on smoothing method
        if self.smoothing == "add-one":
            self._compute_add_one_probs(
                trans_counts, emit_counts, initial_counts, tag_counts
            )
        else:
            self._compute_mle_probs(
                trans_counts, emit_counts, initial_counts
            )

        # Convert to log10 space
        self._to_log_space()

        logger.info(f"Training complete. {len(self.tags)} tags, {len(self.words)} words.")

    def _to_log_space(self) -> None:
        """Convert all probabilities to log10 space, handling zero probabilities."""
        logger.debug("Converting probabilities to log10 space.")

        # Transitions to log10
        for prev_tag in self.tags:
            self.log_transitions[prev_tag] = {}
            for curr_tag in self.tags:
                prob = self.transitions[prev_tag].get(curr_tag, 1e-10)
                self.log_transitions[prev_tag][curr_tag] = (
                    math.log10(prob) if prob > 0 else -float('inf')
                )

        # Emissions to log10
        for tag in self.tags:
            self.log_emissions[tag] = {}
            for word in self.words:
                prob = self.emissions[tag].get(word, 1e-10)
                self.log_emissions[tag][word] = (
                    math.log10(prob) if prob > 0 else -float('inf')
                )

    def viterbi(self, sentence: List[str]) -> List[str]:
        """
        Run Viterbi algorithm to find the best tag sequence in log10 space.

        Args:
            sentence: List of words in the sentence.

        Returns:
            List of predicted tags for each word.
        """
        if not sentence:
            return []

        n = len(sentence)

        # Viterbi tables: trellis[t][tag] = best log10 probability up to time t
        trellis: List[Dict[str, float]] = [{} for _ in range(n)]  # trellis[t][tag] = best log10 probability of any path that ends in 'tag' at time t
        backpointers: List[Dict[str, str]] = [{} for _ in range(n)]  # backpointers[t][curr_tag] = best previous tag at time t

        # ===== Step 1: Initialization =====
        word_0 = sentence[0]
        for tag in self.tags:
            # Get emission probability, defaulting to small value for unknown words
            log_emit = self.log_emissions.get(tag, {}).get(word_0, -3.0)  # log10(0.001)
            trellis[0][tag] = self.log_initial.get(tag, -float('inf')) + log_emit

        # ===== Step 2: Recursion =====
        for t in range(1, n):
            word_t = sentence[t]
            for curr_tag in self.tags:
                # Get emission probability, defaulting to small value for unknown words
                log_emit_t = self.log_emissions.get(curr_tag, {}).get(word_t, -3.0)

                # Find the best previous tag
                best_prob = -float('inf')
                best_prev_tag = None

                for prev_tag in self.tags:
                    if prev_tag not in trellis[t - 1]:
                        continue

                    prev_score = trellis[t - 1][prev_tag]
                    if prev_score == -float('inf'):
                        continue

                    log_trans = self.log_transitions.get(prev_tag, {}).get(curr_tag, -float('inf'))
                    if log_trans == -float('inf'):
                        continue

                    score = prev_score + log_trans + log_emit_t

                    if score > best_prob:
                        best_prob = score
                        best_prev_tag = prev_tag

                if best_prob > -float('inf'):
                    trellis[t][curr_tag] = best_prob
                    if best_prev_tag is not None:
                        backpointers[t][curr_tag] = best_prev_tag

        # ===== Step 3: Backtracking =====
        # Find the best final tag
        if not trellis[n - 1]:
            # If no valid path, return uniform tags
            return [list(self.tags)[0]] * n if self.tags else ["NN"] * n

        best_last_tag = max(
            trellis[n - 1].keys(),
            key=lambda tag: trellis[n - 1][tag]
        )

        # Backtrack to recover the sequence
        path = [best_last_tag]
        for t in range(n - 1, 0, -1):
            prev_tag = backpointers[t].get(path[-1], list(self.tags)[0] if self.tags else "NN")
            path.append(prev_tag)

        path.reverse()
        return path

    def tag_corpus(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Tag all sentences in a corpus using Viterbi.

        Args:
            sentences: List of sentences, each a list of words.

        Returns:
            List of tagged sentences. Each sentence is a list of predicted tags corresponding to the words in the same position.
        """
        return [self.viterbi(sent) for sent in sentences]
