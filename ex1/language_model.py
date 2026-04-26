import pickle
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Union
import spacy
from datasets import load_dataset

class UnigramModel:
    """
    Unigram Language Model using Maximum Likelihood Estimation (MLE).
    
    Attributes:
        unigram_counts (Counter): Frequency of each unique word lemma.
        total_tokens (int): Total number of token occurrences in the training set.
    """
    def __init__(self):
        self.unigram_counts = Counter()
        self.total_tokens = 0
    
    def add_tokens(self, tokens: List[str]) -> None:
        """Updates frequencies with a list of tokens."""
        for token in tokens:
            self.unigram_counts[token] += 1
            self.total_tokens += 1
    
    def get_unigram_probability(self, token: str) -> float:
        """Calculates log P(w). Returns log(1e-10) for OOV."""
        count = self.unigram_counts.get(token, 0)
        if count == 0:
            return -float('inf')  # OOV token has zero probability
        return math.log(count / self.total_tokens)

class BigramModel:
    """Bigram Language Model using MLE with <START> token support."""
    START_TOKEN = "<START>"
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
    
    def add_document(self, tokens: List[str]) -> None:
        """Prepends START and updates bigram frequencies."""
        document = [self.START_TOKEN] + tokens
        for i in range(len(document) - 1):
            w1, w2 = document[i], document[i + 1]
            self.unigram_counts[w1] += 1
            self.bigram_counts[w1][w2] += 1
            
    def get_bigram_probability(self, w1: str, w2: str) -> float:
        """Calculates log P(w2|w1). Returns log(1e-10) for unseen pairs."""
        if w1 not in self.unigram_counts:
            return -float('inf')  # Unseen w1 means zero probability
        
        count_w1_w2 = self.bigram_counts[w1].get(w2, 0)
        count_w1 = self.unigram_counts[w1]
        
        if count_w1_w2 == 0:
            return -float('inf')  # Unseen bigram means zero probability
        return math.log(count_w1_w2 / count_w1)

class LanguageModels:
    """Orchestrator for preprocessing and training on WikiText-2."""
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)
        self.unigram_model = UnigramModel()
        self.bigram_model = BigramModel()
    
    def preprocess_text(self, text: str) -> List[str]:
        """Lemmatizes and filters alphabetic tokens only."""
        doc = self.nlp(text)
        return [t.lemma_.lower() for t in doc if t.is_alpha]
    
    def train(self, max_samples: int = None) -> None:
        """Loads and trains on the dataset."""
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        limit = max_samples if max_samples else len(dataset)
        for i in range(limit):
            text = dataset[i]['text']
            if text.strip():
                tokens = self.preprocess_text(text)
                if tokens:
                    self.unigram_model.add_tokens(tokens)
                    self.bigram_model.add_document(tokens)

    def save(self, folder: str = "models") -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        with open(f"{folder}/unigram.pkl", 'wb') as f: pickle.dump(self.unigram_model, f)
        with open(f"{folder}/bigram.pkl", 'wb') as f: pickle.dump(self.bigram_model, f)

    def load(self, folder: str = "models") -> bool:
        try:
            with open(f"{folder}/unigram.pkl", 'rb') as f: self.unigram_model = pickle.load(f)
            with open(f"{folder}/bigram.pkl", 'rb') as f: self.bigram_model = pickle.load(f)
            return True
        except FileNotFoundError: return False

class LanguageModelEvaluator:
    """Tool for predicting words and scoring sentences."""
    def __init__(self, manager: LanguageModels):
        """
        Initializes with the manager to ensure consistent preprocessing.
        Args:
            manager (LanguageModels): The trained manager object.
        """
        self.manager = manager
        self.uni = manager.unigram_model
        self.bi = manager.bigram_model
    
    def get_token_prob(self, w1: str, w2: str, mode: str) -> float:
        """Calculates log probability based on the mode."""
        if mode == 'unigram':
            return self.uni.get_unigram_probability(w2)
        elif mode == 'bigram':
            return self.bi.get_bigram_probability(w1, w2)
        else: # Smoothed Task 4
            p_bi = math.exp(self.bi.get_bigram_probability(w1, w2))
            p_uni = math.exp(self.uni.get_unigram_probability(w2))
            if p_bi == 0 and p_uni == 0:
                return -float('inf')  # Both models give zero probability
            return math.log((2/3) * p_bi + (1/3) * p_uni)

    def predict_next_word(self, context: str, mode: str) -> Tuple[str, float]:
        """Predicts the best next word after cleaning the context."""
        tokens = self.manager.preprocess_text(context)
        last_word = tokens[-1] if tokens else BigramModel.START_TOKEN
        
        best_word, max_prob = None, -float('inf')
        candidates = self.bi.bigram_counts.get(last_word, self.uni.unigram_counts)
        
        for word in candidates:
            prob = self.get_token_prob(last_word, word, mode)
            if prob > max_prob:
                max_prob, best_word = prob, word
        return best_word, max_prob

    def evaluate_sentence(self, sentence: str, mode: str) -> Tuple[float, float]:
        """
        Calculates Log-Prob and Perplexity using consistent preprocessing.
        Returns (log_prob, perplexity).
        """
        tokens = self.manager.preprocess_text(sentence)
        if not tokens:
            return -float('inf'), float('inf')
        
        log_prob = 0.0
        prev = BigramModel.START_TOKEN
        for token in tokens:
            log_prob += self.get_token_prob(prev, token, mode)
            prev = token
            
        perplexity = math.exp(-log_prob / len(tokens))
        return log_prob, perplexity