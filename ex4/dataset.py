"""Dataset preparation utilities for Exercise 4.

Function Call Tree:
    prepare_dependency_data()
        -> ensure_runtime_directories()
        -> load_dependency_treebank()
        -> split_corpus()
        -> graph_to_sentence()
            -> _node_word()
            -> _node_pos()
            -> _gold_heads_from_graph()

Cross-module Communication:
    main.run_part_1() and main.run_part_2() call prepare_dependency_data().
    models.AveragedPerceptronParser consumes the returned native dictionaries.
    models.AttentionParserAdapter consumes the same test partition.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

from ex4 import config
from ex4.utils import get_logger


Token = Dict[str, str]
Sentence = Dict[str, object]
LOGGER = get_logger(__name__)


def ensure_runtime_directories() -> None:
    """Create output directories declared in the centralized configuration.

    Raises:
        OSError: If a configured directory cannot be created.

    Notes:
        The function intentionally reads all paths from config.py so downstream
        modules never need hardcoded output paths.
    """
    for path in (config.DATA_DIR, config.CHECKPOINT_DIR, config.RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def prepare_dependency_data() -> Tuple[List[Sentence], List[Sentence]]:
    """Load, normalize, and split the NLTK dependency treebank.

    Returns:
        A tuple containing the training sentences and static test sentences.
        The portions are determined in config.py and applied in split_corpus().

    Raises:
        LookupError: If the NLTK corpus cannot be found.

    Notes:
        The final ceil-free 10% slice is used as the evaluation set, matching
        the assignment requirement that the last corpus segment remain static.
    """
    ensure_runtime_directories()
    LOGGER.debug("Preparing NLTK dependency treebank...")
    dependency_graphs: List[Any] = load_dependency_treebank()
    train_graphs, test_graphs = split_corpus(dependency_graphs)
    train_sentences: List[Sentence] = [graph_to_sentence(graph) for graph in train_graphs]
    test_sentences: List[Sentence] = [graph_to_sentence(graph) for graph in test_graphs]
    LOGGER.info("Prepared %d train sentences and %d test sentences.", len(train_sentences), len(test_sentences))
    return train_sentences, test_sentences


def load_dependency_treebank() -> List[Any]:
    """Load parsed dependency graphs from NLTK.

    Returns:
        A list of NLTK DependencyGraph objects.

    Raises:
        ImportError: If NLTK is not installed.
        LookupError: If the dependency treebank resource is unavailable.

    Notes:
        The corpus is requested through dependency_treebank.parsed_sents(),
        which yields dependency structures with token indices and gold heads.
    """
    try:
        from nltk.corpus import dependency_treebank
    except ImportError as exc:
        raise ImportError(
            "NLTK is required for Part 1 and Part 2 data preparation. "
            "Install project dependencies before running ex4/main.py."
        ) from exc

    return list(dependency_treebank.parsed_sents())


def split_corpus(graphs: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Split graphs into the first (100 - config.TEST_SPLIT_RATIO * 100)% train and final (config.TEST_SPLIT_RATIO * 100)% test partitions.

    Args:
        graphs: Sequence of parsed dependency graphs.

    Returns:
        A tuple of train graphs and test graphs.

    Raises:
        ValueError: If the corpus is empty or too small for a nonempty test set.

    Notes:
        For N graphs and ratio r, the split point is int(N * (1 - r)). This
        makes the test set the dynamic final corpus slice required by the PDF.
    """
    if len(graphs) < 2:
        raise ValueError("The dependency treebank must contain at least two sentences.")

    split_index: int = int(len(graphs) * (1.0 - config.TEST_SPLIT_RATIO))
    split_index = min(max(split_index, 1), len(graphs) - 1)
    return list(graphs[:split_index]), list(graphs[split_index:])


def graph_to_sentence(graph: Any) -> Sentence:
    """Convert an NLTK DependencyGraph into a lightweight native dictionary.

    Args:
        graph: NLTK DependencyGraph  for one sentence containing node metadata and gold heads.

    Returns:
        A sentence dictionary with tokens, words, POS tags, and gold head map.

    Raises:
        KeyError: If an expected dependency node field is missing.

    Notes:
        Index 0 is always the artificial ROOT token. Gold heads are represented
        as dependent-index -> head-index, excluding ROOT as a dependent.

    Example:
        >>> from nltk.corpus import dependency_treebank
        >>> graph = dependency_treebank.parsed_sents()[0]
        >>> sentence = graph_to_sentence(graph)
        >>> sentence.keys()
        dict_keys(['tokens', 'words', 'pos', 'gold_heads'])
        >>> sentence['tokens'][0]
        {'word': 'ROOT', 'pos': 'ROOT'}
        >>> sentence['words'][1:7]
        ['Pierre', 'Vinken', ',', '61', 'years', 'old']
        >>> sentence['pos'][1:7]
        ['NNP', 'NNP', ',', 'CD', 'NNS', 'JJ']
        >>> sentence['gold_heads']
        {1: 2, 2: 0, 3: 2, 4: 2, 5: 4, 6: 5, 7: 2, 8: 9, 9: 5}
    """
    node_count: int = max(int(index) for index in graph.nodes.keys())
    tokens: List[Token] = [
        {"word": config.ROOT_WORD, "pos": config.ROOT_POS},
    ]

    for index in range(1, node_count + 1):
        node: Dict[str, Any] = graph.nodes[index]
        tokens.append({"word": _node_word(node), "pos": _node_pos(node)})

    sentence: Sentence = {
        "tokens": tokens,
        "words": [token["word"] for token in tokens],
        "pos": [token["pos"] for token in tokens],
        "gold_heads": _gold_heads_from_graph(graph, node_count),
    }
    return sentence


def shuffled_sentences(sentences: Sequence[Sentence], epoch: int) -> List[Sentence]:
    """Return an epoch-specific shuffled copy of training sentences.

    Args:
        sentences: Training sentences represented as native dictionaries.
        epoch: Zero-based epoch index.

    Returns:
        A shuffled list containing the same sentence dictionaries.

    Notes:
        Shuffling uses RANDOM_SEED + epoch, making the perceptron traversal
        deterministic while still breaking corpus-order artifacts each epoch.
    """
    shuffled: List[Sentence] = list(sentences)
    rng: random.Random = random.Random(config.RANDOM_SEED + epoch)
    rng.shuffle(shuffled)
    return shuffled


def _node_word(node: Dict[str, Any]) -> str:
    """Read and normalize a token word from an NLTK dependency node.

    Args:
        node: Dependency node metadata dictionary.

    Returns:
        A lowercase lexical token, or ROOT for the artificial root.

    Notes:
        Lowercasing lexical forms reduces sparsity for the Boolean word-bigram
        feature template f_word(u, v, s).
    """
    word: str | None = node.get("word")
    if not word:
        return config.ROOT_WORD
    return word.lower()


def _node_pos(node: Dict[str, Any]) -> str:
    """Read and normalize a POS tag from an NLTK dependency node.

    Args:
        node: Dependency node metadata dictionary.

    Returns:
        A POS tag string with ROOT assigned to the artificial root.

    Notes:
        The virtual ROOT token must explicitly receive POS tag ROOT so POS
        bigram features distinguish root arcs from ordinary token arcs.
    """
    tag: str | None = node.get("tag")
    return tag if tag else config.ROOT_POS


def _gold_heads_from_graph(graph: Any, node_count: int) -> Dict[int, int]:
    """Extract gold head assignments from an NLTK dependency graph.

    Args:
        graph: NLTK DependencyGraph with head metadata for one sentence.
        node_count: Number of non-root tokens in the sentence.

    Returns:
        A mapping from dependent indices 1..n to head indices 0..n.

    Raises:
        KeyError: If a non-root node lacks head metadata.

    Notes:
        UAS evaluation and structured perceptron updates compare predicted
        arborescence parents against this gold map.
    """
    gold_heads: Dict[int, int] = {}
    for dependent in range(1, node_count + 1):
        gold_heads[dependent] = int(graph.nodes[dependent]["head"] or 0)
    return gold_heads
