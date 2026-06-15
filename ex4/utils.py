"""Shared utilities for parsing, evaluation, checkpoints, and plotting.

Function Call Tree:
    get_logger()
        -> setup_logging()
            -> _ColoredFormatter.format()
    load_checkpoint_if_valid()
        -> _checkpoint_signature()
    save_checkpoint()
        -> _checkpoint_signature()
    predict_min_arborescence()
        -> _call_external_cle_min()
        -> _normalize_cle_output()
    compute_uas()
    mean_uas()
        -> compute_uas()
    plot_uas_results()
        -> _format_metric_label()

Cross-module Communication:
    models.py uses predict_min_arborescence(), compute_uas(), and checkpoint
    helpers. main.py uses mean_uas() and plot_uas_results().
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ex4 import config


ArcCosts = Dict[Tuple[int, int], float]
HeadMap = Dict[int, int]
LOGGER = logging.getLogger("ex4.utils")


class _ColoredFormatter(logging.Formatter):
    """Formatter that colors log level names for console readability."""

    _COLORS: Dict[int, str] = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    _RESET: str = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with optional ANSI colors.

        Args:
            record: Log record emitted by the Python logging module.

        Returns:
            Formatted log message.
        """
        if config.LOG_USE_COLORS:
            color: str = self._COLORS.get(record.levelno, "")
            original_levelname: str = record.levelname
            record.levelname = f"{color}{record.levelname}{self._RESET}" if color else record.levelname
            try:
                return super().format(record)
            finally:
                record.levelname = original_levelname
        return super().format(record)


def setup_logging() -> None:
    """Configure the shared Exercise 4 logger.

    Notes:
        DEBUG is used for step-by-step progress, INFO for summaries, and ERROR
        for failures. The stream handler flushes normally after each record.
    """
    logger: logging.Logger = logging.getLogger("ex4")
    if logger.handlers:
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        return

    handler: logging.StreamHandler[str] = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColoredFormatter("%(asctime)s | %(levelname)s | %(name)s line %(lineno)d | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured child logger for an Exercise 4 module.

    Args:
        name: Module name requesting a logger.

    Returns:
        Configured logger instance.
    """
    setup_logging()
    if name == "ex4" or name.startswith("ex4."):
        return logging.getLogger(name)
    return logging.getLogger(f"ex4.{name}")


def load_checkpoint_if_valid(path: Path, signature_payload: Mapping[str, Any]) -> Any | None:
    """Load a checkpoint only when its configuration signature matches and when FORCE_RETRAIN is False.

    Args:
        path: Checkpoint file path from config.py.
        signature_payload: Serializable values that define the experiment.

    Returns:
        The checkpoint payload when valid, otherwise None.

    Raises:
        OSError: If the checkpoint exists but cannot be read.

    Notes:
        The signature prevents stale model reuse when constants such as epochs,
        model name, split ratio, or layer settings change between runs.
    """
    if config.FORCE_RETRAIN or not path.exists():
        return None
    with path.open("rb") as handle:
        checkpoint: Dict[str, Any] = pickle.load(handle)
    current_signature: str = _checkpoint_signature(signature_payload)
    checkpoint_signature: str | None = checkpoint.get("signature")
    if checkpoint_signature != current_signature:
        LOGGER.debug(
            "Checkpoint signature mismatch for %s. checkpoint_signature=%s current_signature=%s",
            path,
            checkpoint_signature,
            current_signature,
        )
        return None
    return checkpoint.get("payload")


def save_checkpoint(path: Path, payload: Any, signature_payload: Mapping[str, Any]) -> None:
    """Persist a payload together with a deterministic configuration signature.

    Args:
        path: Destination checkpoint path from config.py.
        payload: Python object to serialize.
        signature_payload: Serializable values that define the experiment.

    Raises:
        OSError: If the checkpoint cannot be written.

    Notes:
        Checkpoints are plain pickle files because perceptron weights are sparse
        dictionaries and attention results are simple metrics dictionaries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "signature": _checkpoint_signature(signature_payload),
        "payload": payload,
    }
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)


def predict_min_arborescence(costs: ArcCosts, n: int) -> HeadMap:
    """Find a minimum rooted arborescence using external cle_min when available.

    Args:
        costs: Mapping from directed edge (head, dependent) to optimization cost.
        n: Number of non-root tokens, so nodes are indexed 0..n.

    Returns:
        A mapping from dependent indices 1..n to predicted head indices.

    Raises:
        ValueError: If the solver output cannot be normalized.

    Notes:
        cle_min is a minimum solver. Upstream scoring functions therefore
        negate model scores or attention strengths before calling this utility.
    """
    external_output: Any = _call_external_cle_min(costs, n)
    return _normalize_cle_output(external_output, n)


def compute_uas(gold_heads: Mapping[int, int], predicted_heads: Mapping[int, int]) -> float:
    """Compute Unlabeled Attachment Score for one sentence.

    Args:
        gold_heads: Gold dependent -> head mapping, excluding ROOT as dependent.
        predicted_heads: Predicted dependent -> head mapping.

    Returns:
        Fraction of dependents whose predicted head equals the gold head.

    Raises:
        ValueError: If the gold head map is empty.

    Notes:
        UAS = correct_head_assignments / number_of_non_root_tokens. Dependency
        labels are intentionally ignored.
    """
    if not gold_heads:
        raise ValueError("Cannot compute UAS for an empty gold head map.")
    correct: int = sum(1 for dep, head in gold_heads.items() if predicted_heads.get(dep) == head)
    return correct / len(gold_heads)


def mean_uas(gold_and_predicted: Iterable[Tuple[Mapping[int, int], Mapping[int, int]]]) -> float:
    """Compute mean UAS over a collection of sentence predictions.

    Args:
        gold_and_predicted: Iterable of (gold_heads, predicted_heads) pairs.

    Returns:
        Arithmetic mean UAS across all provided sentences.

    Raises:
        ValueError: If no sentence pairs are provided.

    Notes:
        This shared metric function is used by both the perceptron and attention
        parsers so reporting remains identical across parts.
    """
    scores: List[float] = [compute_uas(gold, predicted) for gold, predicted in gold_and_predicted]
    if not scores:
        raise ValueError("Cannot compute mean UAS with no evaluated sentences.")
    return sum(scores) / len(scores)


def plot_uas_results(metrics: Mapping[str, float]) -> None:
    """Plot side-by-side UAS metrics for enabled parsing experiments.

    Args:
        metrics: Mapping from experiment label to mean UAS.

    Raises:
        ImportError: If Plotly is not installed and plotting is enabled.

    Notes:
        The plot is saved as an interactive HTML file at config.METRICS_PLOT_PATH.
    """
    if not config.PLOT_RESULTS or not metrics:
        return
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required when config.PLOT_RESULTS is True.") from exc

    config.METRICS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_labels: List[str] = list(metrics.keys())
    labels: List[str] = [_format_metric_label(label) for label in raw_labels]
    values: List[float] = [metrics[label] for label in raw_labels]
    colors: List[str] = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c"]
    figure = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker={
                    "color": [colors[index % len(colors)] for index in range(len(values))],
                    "line": {"color": "#111827", "width": 1},
                },
                text=[f"{value:.3f}" for value in values],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Mean UAS: %{x:.4f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title={
            "text": "Exercise 4 Parser Comparison",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24, "color": "#111827"},
        },
        xaxis={
            "title": "Mean UAS",
            "range": [0.0, 1.0],
            "tickformat": ".0%",
            "gridcolor": "#e5e7eb",
            "zerolinecolor": "#9ca3af",
        },
        yaxis={
            "title": "",
            "autorange": "reversed",
            "tickfont": {"size": 15, "color": "#111827"},
        },
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 14, "color": "#111827"},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#f9fafb",
        bargap=0.35,
        height=max(420, 110 * len(labels)),
        margin={"l": 190, "r": 70, "t": 90, "b": 70},
    )
    figure.write_html(config.METRICS_PLOT_PATH, include_plotlyjs="cdn")


def _format_metric_label(label: str) -> str:
    """Convert internal metric keys into readable plot labels.

    Args:
        label: Raw metric key used by the experiment code.

    Returns:
        Human-friendly label for charts.
    """
    if label == "perceptron_mst":
        return "Perceptron MST"
    if label.startswith("attention_layer_"):
        parts: List[str] = label.split("_")
        if len(parts) >= 4:
            return f"BERT attention - layer {parts[2]} ({parts[3]})"
    return label.replace("_", " ").title()


def _checkpoint_signature(payload: Mapping[str, Any]) -> str:
    """Build a stable SHA256 signature for checkpoint compatibility.

    Args:
        payload: JSON-serializable configuration payload.

    Returns:
        A hexadecimal SHA256 digest.

    Raises:
        TypeError: If payload contains non-serializable values.

    Notes:
        Paths are converted by callers before reaching this function; keeping
        the signature JSON-based makes compatibility checks deterministic.
    """
    encoded: bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _call_external_cle_min(costs: ArcCosts, n: int) -> Any:
    """Call a provided chu_liu_edmonds.cle_min implementation if importable.

    Args:
        costs: Mapping from directed edge to cost.
        n: Number of non-root tokens.

    Returns:
        Raw external solver output.

    Raises:
        ImportError: If chu_liu_edmonds.py is unavailable.
        RuntimeError: If an importable helper raises during optimization.

    Notes:
        The assignment helper expects the total number of nodes, so this adapter
        passes n + 1 for a sentence with n non-root tokens.
    """
    try:
        from ex4.chu_liu_edmonds import cle_min
    except ImportError as exc:
        raise ImportError("chu_liu_edmonds.py with cle_min is required for decoding.") from exc

    try:
        return cle_min(costs, n + 1)
    except Exception as exc:
        raise RuntimeError("External cle_min failed during arborescence decoding.") from exc


def _normalize_cle_output(output: Any, n: int) -> HeadMap:
    """Normalize common cle_min return formats into dependent -> head maps.

    Args:
        output: Raw output returned by cle_min.
        n: Number of non-root tokens.

    Returns:
        Normalized dependent -> head mapping.

    Raises:
        ValueError: If the format is unsupported or incomplete.

    Notes:
        Course-provided CLE helpers often return either edge lists, parent maps,
        or (cost, structure) tuples. This adapter accepts those common shapes.
    """
    if isinstance(output, tuple) and len(output) == 2:
        first, second = output
        output = second if not isinstance(first, (dict, list, set)) else first

    heads: HeadMap = {}
    if isinstance(output, dict):
        keys_are_dependents: bool = all(isinstance(key, int) for key in output.keys())
        if keys_are_dependents:
            heads = {int(dep): int(head) for dep, head in output.items() if int(dep) != 0}
        else:
            heads = {int(dep): int(head) for head, dep in output.keys() if int(dep) != 0}
    elif isinstance(output, (list, set, tuple)):
        for item in output:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                head, dep = int(item[0]), int(item[1])
                if dep != 0:
                    heads[dep] = head

    expected: set[int] = set(range(1, n + 1))
    if set(heads.keys()) != expected:
        raise ValueError(f"Unsupported or incomplete cle_min output: {output!r}")
    return heads

