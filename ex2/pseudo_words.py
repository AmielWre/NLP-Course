"""
Pseudo-word strategy for handling out-of-vocabulary and low-frequency words.

This module implements a set of pseudo-words to replace OOV terms and low-frequency words,
based on their syntactic features.
"""

import logging
import re
from typing import Dict, Set, List

logger = logging.getLogger(__name__)


class PseudoWordConverter:
    """
    Convert OOV and low-frequency words to pseudo-word categories.

    The pseudo-word categories are based on syntactic features such as:
    - Digit patterns (all digits, contains digits, no digits)
    - Capitalization (initial capital, all capital, all lowercase)
    - Common suffixes (-ed, -ing, -ly, -er, -tion, -ness, etc.)
    - Punctuation
    - Length
    """


    def __init__(self, min_frequency: int = 5) -> None:
        """
        Initialize the pseudo-word converter.

        Args:
            min_frequency: Words appearing fewer than this many times are considered low-frequency.
        """
        self.min_frequency = min_frequency
        logger.debug(f"Initialized PseudoWordConverter with min_frequency={min_frequency}.")

    @staticmethod
    def get_pseudo_word(word: str) -> str:
        """
        Classify a word into a pseudo-word category.

        Args:
            word: The word to classify.

        Returns:
            The pseudo-word category string.
        """
        if not word:
            return "UNK-OTHER"

        # 1. Shape and Capitalization Features First
        # Check for all uppercase (but not single letter abbreviations)
        if word.isupper() and len(word) > 1:
            return "UNK-ALLUPPER"

        # Check for initial capitalization (first letter upper, rest lower)
        if word[0].isupper() and word[1:].islower():
            return "UNK-INITCAP"
        
        # Check for all digits
        if word.isdigit():
            return "UNK-DIGIT-ALL"

        # Check for mixed digits and letters
        if any(c.isdigit() for c in word):
            return "UNK-DIGIT-MIXED"

        # 2. Specific Morphological Suffixes (Evaluated BEFORE generic case fallbacks)
        # Check for plural (ends with 's' but not 'ss' and longer than 2 characters)
        if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
            return "UNK-PLURAL-S"
        
        # Check for adjectives with common suffixes
        if any(word.endswith(suff) for suff in ["able", "ful", "ous", "al", "ive", "ic", "ish", "ary", "ical", "less", "ent", "ant"]):
            return "UNK-ADJECTIVE-SUFFIX"

        # Check for verbal/participle suffixes
        if word.endswith("ed"):
            return "UNK-ED"
        if word.endswith("ing"):
            return "UNK-ING"
        if word.endswith("er"):
            return "UNK-ER"
        if word.endswith("est"):
            return "UNK-EST"
        if any(word.endswith(suff) for suff in ["ize", "ise", "ate"]):
            return "UNK-VERB-SUFFIX"

        # Check for common nominal suffixes
        if any(word.endswith(suff) for suff in ["tion", "ness", "ment", "ity", "ship"]):
            return "UNK-TION"  # Grouping under your existing nominal category

        # Check for adverbs
        if word.endswith("ly"):
            return "UNK-LY"

        # 3. Structural/Punctuation Fallbacks
        # Check for hyphen
        if "-" in word:
            return "UNK-ADJ-HYPHEN"

        # Check for punctuation
        if re.search(r'[^\w\s-]', word):
            return "UNK-PUNCT"

        # 4. Generic Case Fallbacks (Executed ONLY if no suffix matched)
        if word.islower():
            return "UNK-LOWERCASE"

        return "UNK-OTHER"
    
