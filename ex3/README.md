# Ex2 Module Overview

## Core Modules

### `data_loader.py`
Handles data loading and preprocessing from NLTK Brown corpus ("news" category).
- `clean_tag()`: Strips complex tag suffixes (e.g., "NN+JJ" → "NN")
- `load_brown_corpus()`: Loads tagged sentences from corpus
- `partition_data()`: Splits data into 90% training / 10% test
- `prepare_data()`: Main orchestrator for data pipeline

**Output**: Training/test sentences, tag set, unique tag count

---

### `baseline_tagger.py`
Implements a baseline POS tagger using Maximum Likelihood Estimation.
- `train()`: Learns most likely tag for each word from training data
- `tag()`: Tags a single word (defaults to "NN" for unknowns)
- `tag_corpus()`: Batch tagging
- `get_vocabulary()`: Returns known words

**Performance**: Simple baseline for comparison (Part b)

---

### `hmm_tagger.py`
Implements a Bigram HMM with Viterbi algorithm in log₁₀ space.
- `train()`: Learns transition and emission probabilities
- `_compute_mle_probs()`: MLE probability computation
- `_compute_add_one_probs()`: Laplace smoothing (Add-One)
- `_to_log_space()`: Converts probabilities to log₁₀ for numerical stability
- `viterbi()`: Finds best tag sequence using Viterbi algorithm
- `tag_corpus()`: Batch Viterbi tagging

**Features**: Supports both MLE (Part c) and Add-One smoothing (Part d)

---

### `pseudo_words.py`
Converts out-of-vocabulary (OOV) and low-frequency words to pseudo-word categories.
- `get_pseudo_word()`: Classifies word into 17 pseudo-word categories (e.g., "UNK-ED", "UNK-INITCAP", "UNK-DIGIT-ALL")
- Categories based on: capitalization, digit patterns, morphological suffixes, punctuation

**Usage**: Part e (pseudo-words + MLE/Add-One smoothing)

---

### `evaluator.py`
Computes evaluation metrics and generates visualization plots.
- `compute_accuracy()`: Accuracy metric
- `compute_error_rate()`: Error rate (1 - accuracy)
- `separate_known_unknown()`: Splits predictions by vocabulary membership
- `evaluate_with_known_unknown_split()`: Reports known/unknown/overall error rates
- `build_confusion_matrix()`: Creates confusion matrix for all tag pairs
- `build_and_analyze_confusion_matrix()`: **Generates confusion matrix heatmap plot** (saved to `results/`)
- `analyze_confusion_matrix()`: Top N error pairs

**Outputs**: Metrics logged to console; heatmap PNG saved to `results/`

---

### `main.py`
Orchestrates the full POS tagging pipeline with 5 tagger variants.

**Pipeline Stages**:
1. **(a) Data Loading**: Brown corpus → training/test split
2. **(b) Baseline**: Most-likely-tag tagger (MLE)
3. **(c) HMM-MLE**: Bigram HMM without smoothing
4. **(d) HMM-AddOne**: Bigram HMM with Laplace smoothing
5. **(e.1) Pseudo-Words + MLE**: Low-freq words → pseudo-words, HMM-MLE
6. **(e.2) Pseudo-Words + AddOne**: Low-freq words → pseudo-words, HMM-AddOne (generates confusion matrix)

**Functions**:
- `extract_words_and_tags()`: Separates words from (word, tag) tuples
- `compute_word_frequencies()`: Word frequency counter
- `run_*_tagger()`: Individual tagger training/evaluation functions
- `print_comparison()`: **Generates error rate comparison bar plot** (saved to `results/`)

**Outputs**: All logs to stdout; plots saved to `results/` directory

---

## Execution

Run the complete pipeline:
```bash
python main.py
```

Outputs are saved to:
- Console logs (INFO level)
- `results/Pseudo-Words_(Add-One_Smoothing)_confusion_matrix.png` - confusion matrix heatmap
- `results/tagger_comparison_error_rates.png` - error rate comparison chart
