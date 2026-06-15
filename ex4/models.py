"""Model components for perceptron and attention-based dependency parsing.

Function Call Tree:
Perceptron:
    AveragedPerceptronParser.fit()
        -> dataset.shuffled_sentences()
        -> predict()
            -> score_arcs()
                -> edge_features()
                -> _dot()
            -> utils.predict_min_arborescence()
        -> _update_sentence()
            -> edge_features()
            -> _update_feature()
        -> _average_weights()
    AveragedPerceptronParser.evaluate()
        -> predict()
        -> utils.mean_uas()
    load_or_train_perceptron()
        -> utils.load_checkpoint_if_valid()
        -> AveragedPerceptronParser.fit()
        -> utils.save_checkpoint()
=================================================
Attention-based parser:
    AttentionParserAdapter.evaluate_layers()
        -> predict()
            -> get_attention_matrix()
            -> attn_to_arc_scores()
            -> utils.predict_min_arborescence()
        -> utils.mean_uas()
    load_or_evaluate_attention()
        -> utils.load_checkpoint_if_valid()
        -> AttentionParserAdapter.evaluate_layers()
        -> utils.save_checkpoint()

Cross-module Communication:
    dataset.py provides normalized native sentence dictionaries. utils.py
    provides CLE decoding, UAS metrics, checkpointing, and plotting support.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

from ex4 import config
from ex4.dataset import Sentence, shuffled_sentences
from ex4.utils import (
    ArcCosts,
    HeadMap,
    get_logger,
    load_checkpoint_if_valid,
    mean_uas,
    predict_min_arborescence,
    save_checkpoint,
)


Weights = Dict[str, float]
LOGGER = get_logger(__name__)


def edge_features(sentence: Sentence, head: int, dependent: int) -> List[str]:
    """Construct sparse Boolean feature keys for one directed dependency edge.

    Args:
        sentence: Native sentence dictionary with words and POS tags.
        head: Candidate head node index, including ROOT at index 0.
        dependent: Candidate dependent node index, excluding ROOT during use.

    Returns:
        Sparse feature keys active for edge head -> dependent with those features:
            WORD_BIGRAM::head_word::dependent_word
            POS_BIGRAM::head_pos::dependent_pos

    Raises:
        IndexError: If head or dependent is outside the sentence bounds.

    Notes:
        Implements the assignment templates f(u, v, s): lexical word bigram and
        POS bigram. Each active Boolean feature contributes value 1.

    Example:
        >>> sentence = {"words": ["ROOT", "I", "saw", "her"], "pos": ["ROOT", "PRON", "VERB", "PRON"]}
        >>> edge_features(sentence, 2, 3)
        ['WORD_BIGRAM::saw::her', 'POS_BIGRAM::VERB::PRON']
        >>> edge_features(sentence, 0, 1)
        ['WORD_BIGRAM::ROOT::I', 'POS_BIGRAM::ROOT::PRON']
    """
    words: List[str] = sentence["words"]
    pos_tags: List[str] = sentence["pos"]
    return [
        f"WORD_BIGRAM::{words[head]}::{words[dependent]}",
        f"POS_BIGRAM::{pos_tags[head]}::{pos_tags[dependent]}",
    ]


def score_arcs(sentence: Sentence, weights: Mapping[str, float]) -> ArcCosts:
    """Score every legal directed arc in a sentence and convert scores into CLE (minimum spanning tree) costs.

    Args:
        sentence: Native sentence dictionary.
        weights: Sparse model parameters.

    Returns:
        A mapping from (head, dependent) to negative perceptron score.

    Notes:
        The perceptron maximizes theta dot phi(head, dep, sentence), while
        cle_min minimizes. Therefore each dot product is multiplied by -1.
    """
    token_count: int = len(sentence["tokens"])
    costs: ArcCosts = {}
    for head in range(token_count):
        for dependent in range(1, token_count):  # start at 1 to exclude ROOT as a dependent
            if head == dependent:
                continue
            score: float = _dot(weights, edge_features(sentence, head, dependent))
            costs[(head, dependent)] = -score
    return costs


def load_or_train_perceptron(train_sentences: Sequence[Sentence]) -> "AveragedPerceptronParser":
    """Load a valid perceptron checkpoint or train a fresh averaged model.

    Args:
        train_sentences: Training sentences from dataset.prepare_dependency_data().

    Returns:
        A fitted AveragedPerceptronParser instance.

    Raises:
        OSError: If checkpoint IO fails.

    Notes:
        The checkpoint signature depends on perceptron hyperparameters and the
        number of training examples, so changing config constants retrains.
    """
    signature: Dict[str, Any] = {  # used for checkpoint validation and logging, not model state
        "epochs": config.PERCEPTRON_EPOCHS,
        "learning_rate": config.PERCEPTRON_LEARNING_RATE,
        "train_size": len(train_sentences),
        "test_ratio": config.TEST_SPLIT_RATIO,
        "root_word": config.ROOT_WORD,
        "root_pos": config.ROOT_POS,
    }
    payload: Any | None = load_checkpoint_if_valid(config.PERCEPTRON_CHECKPOINT, signature)
    parser: AveragedPerceptronParser = AveragedPerceptronParser()
    if payload is not None:
        LOGGER.info("Loaded perceptron checkpoint from %s.", config.PERCEPTRON_CHECKPOINT)
        parser.weights = payload["weights"]
        return parser

    LOGGER.info("Training averaged perceptron parser...")
    parser.fit(train_sentences)
    save_checkpoint(config.PERCEPTRON_CHECKPOINT, {"weights": parser.weights}, signature)
    LOGGER.info("Saved perceptron checkpoint to %s.", config.PERCEPTRON_CHECKPOINT)
    return parser


def load_or_evaluate_attention(test_sentences: Sequence[Sentence]) -> Dict[str, float]:
    """Load valid attention metrics or run the BERT attention parser.

    Args:
        test_sentences: Static test sentences from dataset.prepare_dependency_data().

    Returns:
        Mapping from layer label to mean UAS.

    Raises:
        ImportError: If torch or transformers are unavailable and no checkpoint exists.

    Notes:
        This caches final metrics rather than tensors, avoiding repeated BERT
        downloads and inference when the relevant constants are unchanged.
    """
    signature: Dict[str, Any] = {
        "model": config.BERT_MODEL_NAME,
        "settings": list(config.ATTENTION_LAYER_SETTINGS),
        "test_size": len(test_sentences),
        "test_ratio": config.TEST_SPLIT_RATIO,
        "max_wordpieces": config.MAX_BERT_WORDPIECE_LENGTH,
    }
    payload: Any | None = load_checkpoint_if_valid(config.ATTENTION_CHECKPOINT, signature)
    if payload is not None:
        LOGGER.info("Loaded attention metrics checkpoint from %s.", config.ATTENTION_CHECKPOINT)
        return payload["metrics"]

    LOGGER.info("Running BERT attention-based parsing evaluation...")
    adapter: AttentionParserAdapter = AttentionParserAdapter()
    metrics: Dict[str, float] = adapter.evaluate_layers(test_sentences)
    save_checkpoint(config.ATTENTION_CHECKPOINT, {"metrics": metrics}, signature)
    LOGGER.info("Saved attention metrics checkpoint to %s.", config.ATTENTION_CHECKPOINT)
    return metrics


class AveragedPerceptronParser:
    """Structured averaged perceptron parser for MST dependency decoding.

    Attributes:
        weights: Sparse feature weight dictionary.
            Each key is a string like "WORD_BIGRAM::saw::her" and each value is a real-valued weight.
            If a feature is not in the dictionary, its weight is treated as 0.0.
            Example: {"WORD_BIGRAM::saw::her": 0.5, "POS_BIGRAM::VERB::PRON": -1.2, ...}
        _totals: Accumulator for historical weight values used in averaging.
        _timestamps: Tracking the last update step for each feature for averaging.
        _step: Global training step counter for averaging calculations.

    Notes:
        Training performs two shuffled corpus traversals by default. For each
        sentence, the update is theta <- theta + eta(Phi_gold - Phi_pred).
        Lazy totals maintain the global running average over all observed
        sentence-level iterations.
    """

    def __init__(self) -> None:
        """Initialize sparse weights and lazy averaging state.

        Notes:
            totals and timestamps implement the standard averaged perceptron
            trick without materializing all historical parameter vectors.
        """
        self.weights: Weights = {}
        self._totals: DefaultDict[str, float] = defaultdict(float)
        self._timestamps: DefaultDict[str, int] = defaultdict(int)
        self._step: int = 0

    def fit(self, train_sentences: Sequence[Sentence]) -> None:
        """Train the structured perceptron and replace weights by averages.

        Args:
            train_sentences: Training sentence dictionaries.

        Raises:
            ValueError: If train_sentences is empty.

        Notes:
            Each epoch uses a deterministic shuffle. Inference inside training
            decodes the highest-scoring arborescence via CLE over negated costs.
        """
        if not train_sentences:
            raise ValueError("Cannot train perceptron with no training sentences.")

        for epoch in range(config.PERCEPTRON_EPOCHS):
            LOGGER.debug("Perceptron epoch %d/%d started.", epoch + 1, config.PERCEPTRON_EPOCHS)
            shuffled: List[Sentence] = shuffled_sentences(train_sentences, epoch)
            for index, sentence in enumerate(shuffled, start=1):
                self._step += 1
                predicted: HeadMap = self.predict(sentence)
                gold: HeadMap = sentence["gold_heads"]  # type: ignore[assignment]
                if predicted != gold:
                    self._update_sentence(sentence, gold, predicted)
                if index % config.LOG_EVERY_SENTENCES == 0 or index == len(shuffled):
                    LOGGER.debug(
                        "Perceptron epoch %d/%d: %d/%d sentences processed.",
                        epoch + 1,
                        config.PERCEPTRON_EPOCHS,
                        index,
                        len(shuffled),
                    )
        self.weights = self._average_weights()
        LOGGER.info("Perceptron averaging complete.")

    def predict(self, sentence: Sentence) -> HeadMap:
        """Predict a dependency tree for one sentence.

        Args:
            sentence: Native sentence dictionary.

        Returns:
            Predicted dependent -> head mapping.

        Raises:
            ValueError: If CLE decoding cannot produce a valid tree.

        Notes:
            ROOT is allowed as a head but never as a dependent. Self-loops are
            excluded before optimization.
        """
        weights: Mapping[str, float] = self.weights
        costs: ArcCosts = score_arcs(sentence, weights)
        n: int = len(sentence["tokens"]) - 1  # type: ignore[arg-type]
        return predict_min_arborescence(costs, n)

    def evaluate(self, test_sentences: Sequence[Sentence]) -> float:
        """Evaluate the parser on the static test subset.

        Args:
            test_sentences: Test sentence dictionaries.

        Returns:
            Mean UAS over the test set.

        Raises:
            ValueError: If no test sentences are provided.

        Notes:
            UAS is computed by utils.mean_uas(), shared with the attention
            parser for consistent reporting.
        """
        pairs: List[Tuple[Mapping[int, int], Mapping[int, int]]] = []
        LOGGER.debug("Evaluating perceptron parser on the test set...")
        for index, sentence in enumerate(test_sentences, start=1):
            gold: HeadMap = sentence["gold_heads"]  # type: ignore[assignment]
            pairs.append((gold, self.predict(sentence)))
            if index % config.LOG_EVERY_SENTENCES == 0 or index == len(test_sentences):
                LOGGER.debug("Perceptron evaluation: %d/%d sentences processed.", index, len(test_sentences))
        return mean_uas(pairs)

    def _update_sentence(self, sentence: Sentence, gold: HeadMap, predicted: HeadMap) -> None:
        """Apply the structured perceptron update for one sentence.

        Args:
            sentence: Sentence dictionary.
            gold: Gold dependent -> head mapping.
            predicted: Predicted dependent -> head mapping.

        Notes:
            The sparse difference vector is accumulated feature-by-feature:
            +eta for each gold edge feature and -eta for each predicted edge
            feature.
        """
        eta: float = config.PERCEPTRON_LEARNING_RATE
        for dependent, gold_head in gold.items():
            predicted_head: int | None = predicted.get(dependent)
            if predicted_head == gold_head:
                continue
            for feature in edge_features(sentence, gold_head, dependent):
                self._update_feature(feature, eta)
            if predicted_head is not None:
                for feature in edge_features(sentence, predicted_head, dependent):
                    self._update_feature(feature, -eta)

    def _update_feature(self, feature: str, delta: float) -> None:
        """Update one sparse feature weight and its averaging accumulator.

        Args:
            feature: Sparse feature key.
            delta: Signed update value.

        Notes:
            Before changing theta_f, the previous value is credited for the
            number of sentence steps since the feature was last touched.
        """
        elapsed: int = self._step - self._timestamps[feature]
        self._totals[feature] += elapsed * self.weights.get(feature, 0.0)
        self._timestamps[feature] = self._step
        self.weights[feature] = self.weights.get(feature, 0.0) + delta
        if self.weights[feature] == 0.0:
            del self.weights[feature]

    def _average_weights(self) -> Weights:
        """Finalize and return averaged perceptron weights.

        Returns:
            Sparse averaged weight dictionary.

        Raises:
            ValueError: If called before any training step.

        Notes:
            For feature f, avg_f = total_historical_theta_f / number_of_updates.
        """
        if self._step == 0:
            raise ValueError("Cannot average perceptron weights before training.")

        averaged: Weights = {}
        features: set[str] = set(self.weights.keys()) | set(self._totals.keys())
        for feature in features:
            elapsed: int = self._step - self._timestamps[feature]
            total: float = self._totals[feature] + elapsed * self.weights.get(feature, 0.0)
            value: float = total / self._step
            if value != 0.0:
                averaged[feature] = value
        return averaged


class AttentionParserAdapter:
    """Adapter that converts BERT self-attention into dependency arc scores.

    Raises:
        ImportError: If torch or transformers cannot be imported.

    Notes:
        BERT attentions are not trained dependency scores. The adapter treats
        attention strength from candidate head token to dependent token as an
        arc preference and decodes a maximum-attention arborescence by passing
        negated strengths to the minimum CLE solver.
    """

    def __init__(self) -> None:
        """Instantiate tokenizer and BERT model exactly once.

        Raises:
            ImportError: If required Hugging Face dependencies are unavailable.

        Notes:
            The model is configured with output_attentions=True and switched to
            eval mode so no gradients or training behavior are triggered.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            from ex4.transformer_parser_utils import get_attention_matrix
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for Part 2 attention parsing."
            ) from exc

        LOGGER.info("Loading Hugging Face model %s...", config.BERT_MODEL_NAME)
        self.attention_matrix_fn = get_attention_matrix
        self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(config.BERT_MODEL_NAME, output_attentions=True)
        self.model.eval()
        LOGGER.info("Loaded %s in eval mode.", config.BERT_MODEL_NAME)

    def evaluate_layers(self, test_sentences: Sequence[Sentence]) -> Dict[str, float]:
        """Evaluate configured BERT layers on the shared test set.

        Args:
            test_sentences: Test sentence dictionaries.

        Returns:
            Mapping from layer/head-mode label to mean UAS.

        Raises:
            ValueError: If no test sentences are provided.

        Notes:
            The exact settings are read from config.ATTENTION_LAYER_SETTINGS and
            evaluated independently: layer 0, 5, and 11 with mean head mode.
        """
        metrics: Dict[str, float] = {}
        for layer, head_mode in config.ATTENTION_LAYER_SETTINGS:
            LOGGER.debug("Attention evaluation layer=%d, head_mode=%s started.", layer, head_mode)
            pairs: List[Tuple[Mapping[int, int], Mapping[int, int]]] = []
            for index, sentence in enumerate(test_sentences, start=1):
                gold: HeadMap = sentence["gold_heads"]  # type: ignore[assignment]
                predicted: HeadMap = self.predict(sentence, layer, head_mode)
                pairs.append((gold, predicted))
                if index % config.LOG_EVERY_SENTENCES == 0 or index == len(test_sentences):
                    LOGGER.debug(
                        "Attention layer=%d, head_mode=%s: %d/%d sentences processed.",
                        layer,
                        head_mode,
                        index,
                        len(test_sentences),
                    )
            metrics[f"attention_layer_{layer}_{head_mode}"] = mean_uas(pairs)
            LOGGER.info(
                "Attention layer=%d, head_mode=%s UAS: %.4f",
                layer,
                head_mode,
                metrics[f"attention_layer_{layer}_{head_mode}"],
            )
        return metrics

    def predict(self, sentence: Sentence, layer: int, head_mode: str) -> HeadMap:
        """Predict one dependency tree from a BERT attention layer.

        Args:
            sentence: Native sentence dictionary.
            layer: Zero-based BERT layer index.
            head_mode: Attention-head aggregation mode; currently "mean".

        Returns:
            Predicted dependent -> head mapping.

        Raises:
            ValueError: If head_mode is unsupported.

        Notes:
            Attention scores are converted into CLE costs by attn_to_arc_scores.
        """
        matrix: List[List[float]] = self.get_attention_matrix(sentence, layer, head_mode)
        costs: ArcCosts = attn_to_arc_scores(matrix)
        return predict_min_arborescence(costs, len(matrix) - 1)

    def get_attention_matrix(self, sentence: Sentence, layer: int, head_mode: str) -> List[List[float]]:
        """Extract an aligned ROOT-aware word-level attention matrix.

        Args:
            sentence: Native sentence dictionary.
            layer: Zero-based BERT layer index.
            head_mode: Attention-head aggregation mode; currently "mean".

        Returns:
            A square (n + 1) x (n + 1) matrix indexed by ROOT plus words.

        Raises:
            ValueError: If head_mode is unsupported or a sentence is too long.

        Notes:
            BERT lacks an explicit ROOT token, so the [CLS] position is used as
            ROOT. WordPiece subtokens are aligned back to original words by
            averaging attention over all subtoken pairs for each word pair.
        """
        if head_mode != "mean":
            raise ValueError(f"Unsupported attention head mode: {head_mode}")

        words: List[str] = sentence["words"][1:]  # type: ignore[index]
        if len(words) > config.MAX_BERT_WORDPIECE_LENGTH:
            raise ValueError("Sentence exceeds the configured BERT WordPiece limit.")

        matrix: Any = self.attention_matrix_fn(words, self.tokenizer, self.model, layer=layer, head_mode=head_mode)
        return matrix.tolist() if hasattr(matrix, "tolist") else matrix


def attn_to_arc_scores(attn_matrix: Sequence[Sequence[float]]) -> ArcCosts:
    """Convert a ROOT-aware attention matrix into CLE-compatible arc costs.

    Args:
        attn_matrix: Square (n + 1) x (n + 1) attention matrix.

    Returns:
        Directed edge cost mapping for all legal head -> dependent arcs.

    Raises:
        ValueError: If the matrix is not square.

    Notes:
        ROOT index 0 can be a head but can never be evaluated as a dependent.
        Values are negated so minimizing cost maximizes attention strength.
    """
    size: int = len(attn_matrix)
    if any(len(row) != size for row in attn_matrix):
        raise ValueError("Attention matrix must be square.")

    costs: ArcCosts = {}
    for head in range(size):
        for dependent in range(1, size):
            if head == dependent:
                continue
            costs[(head, dependent)] = -float(attn_matrix[head][dependent])
    return costs


def _dot(weights: Mapping[str, float], features: Iterable[str]) -> float:
    """Compute a sparse dot product for Boolean active features.

    Args:
        weights: Sparse feature weights.
        features: Active Boolean feature keys.

    Returns:
        Sum of weights for active features.

    Notes:
        Since each feature value is 1, theta dot phi is simply a key lookup sum.
    """
    return sum(weights.get(feature, 0.0) for feature in features)
