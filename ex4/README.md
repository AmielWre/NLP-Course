# Exercise 4 - Dependency Parsing

Amiel Wreschner, cs user: amiel  
Noam Azulay, cs user: noam.azulay

## Overview

This directory includes two independent dependency parsing pipelines:

1. Perceptron-Based MST Parsing
2. Attention-Based Parsing with BERT attentions

Both parts are controlled from `config.py`, so they can be enabled or disabled independently without changing the execution code.

## Files

`config.py`  
Stores all configuration values: paths, execution flags, hyperparameters, checkpoint paths, logging settings, BERT settings, and plot output settings.

`dataset.py`  
Loads the NLTK `dependency_treebank`, converts dependency graphs into native Python dictionaries, and splits the corpus into train and test partitions.

`models.py`  
Contains the averaged perceptron parser, sparse feature extraction, arc scoring, BERT attention parsing adapter, and model-level checkpoint usage.

`utils.py`  
Contains shared utilities: logger setup, checkpoint save/load helpers, CLE decoding adapter, UAS computation, metric averaging, and Plotly result plotting.

`main.py`  
Unified entry point. It reads config flags, prepares data, runs the enabled pipeline or pipelines, plots results, and logs final metrics.

`chu_liu_edmonds.py`  
Provided helper implementing `cle_min`, the minimum spanning arborescence decoder used by both parsing methods.

`transformer_parser_utils.py`  
Provided helper for extracting word-level BERT attention matrices.

## How To Run

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m ex4.main
```

Or, if your active Python environment is already the project environment:

```powershell
python -m ex4.main
```

Running as a module is important because `ex4` is a package and the code uses package imports such as:

```python
from ex4.models import load_or_train_perceptron
```

## Configuration

Edit `config.py` to control execution.

Important flags:

```python
RUN_PART_1_PERCEPTRON = True
RUN_PART_2_ATTENTION = True
FORCE_RETRAIN = True
```

Set `RUN_PART_1_PERCEPTRON` to `False` to skip perceptron training and evaluation.

Set `RUN_PART_2_ATTENTION` to `False` to skip BERT loading and attention evaluation.

Set `FORCE_RETRAIN` to `False` to reuse valid checkpoints when the configuration signature matches.

Training and model settings:

```python
PERCEPTRON_EPOCHS = 2
PERCEPTRON_LEARNING_RATE = 1.0
BERT_MODEL_NAME = "bert-base-uncased"
ATTENTION_LAYER_SETTINGS = ((0, "mean"), (5, "mean"), (11, "mean"))
```

Logging settings:

```python
LOG_LEVEL = "DEBUG"
LOG_USE_COLORS = True
LOG_EVERY_SENTENCES = 150
```

Plot output:

```python
PLOT_RESULTS = True
METRICS_PLOT_PATH = RESULTS_DIR / "ex4_uas_comparison.html"
```

## Function Call Flow

Main flow:

```text
main.main()
    -> prepare_dependency_data()
    -> run_part_1() if RUN_PART_1_PERCEPTRON
    -> run_part_2() if RUN_PART_2_ATTENTION
    -> plot_uas_results()
    -> _print_metrics()
```

Data flow:

```text
prepare_dependency_data()
    -> ensure_runtime_directories()
    -> load_dependency_treebank()
    -> split_corpus()
    -> graph_to_sentence()
        -> _node_word()
        -> _node_pos()
        -> _gold_heads_from_graph()
```

Perceptron flow:

```text
run_part_1()
    -> load_or_train_perceptron()
        -> load_checkpoint_if_valid()
        -> AveragedPerceptronParser.fit() if no valid checkpoint
            -> shuffled_sentences()
            -> AveragedPerceptronParser.predict()
                -> score_arcs()
                    -> edge_features()
                    -> _dot()
                -> predict_min_arborescence()
            -> _update_sentence()
                -> _update_feature()
            -> _average_weights()
        -> save_checkpoint()
    -> AveragedPerceptronParser.evaluate()
        -> AveragedPerceptronParser.predict()
        -> mean_uas()
```

Attention flow:

```text
run_part_2()
    -> load_or_evaluate_attention()
        -> load_checkpoint_if_valid()
        -> AttentionParserAdapter.evaluate_layers() if no valid checkpoint
            -> AttentionParserAdapter.predict()
                -> get_attention_matrix()
                -> attn_to_arc_scores()
                -> predict_min_arborescence()
            -> mean_uas()
        -> save_checkpoint()
```

Utility flow:

```text
predict_min_arborescence()
    -> _call_external_cle_min()
    -> _normalize_cle_output()

plot_uas_results()
    -> _format_metric_label()
    -> write Plotly HTML file
```

## Outputs

Checkpoints are written under:

```text
ex4/checkpoints/
```

The comparison plot is written to:

```text
ex4/results/ex4_uas_comparison.html
```

Final UAS metrics are logged to the terminal.
