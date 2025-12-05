"""
🕉️ DharmaMind Custom Tokenizer
==============================

A tokenizer designed specifically for dharmic texts, with special handling for:
- Sanskrit/Pali terms
- Diacritical marks
- Sacred mantras
- Spiritual concepts

This is OUR tokenizer, not a wrapper.

Author: DharmaMind Team
"""

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class DharmaTokenizer:
    """
    Custom tokenizer for dharmic texts.

    Features:
    - BPE (Byte Pair Encoding) tokenization
    - Special tokens for dharmic concepts
    - Sanskrit/Pali term preservation
    - Mantra recognition
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SEP_TOKEN = "<sep>"

    # Dharmic special tokens
    OM_TOKEN = "<om>"
    MANTRA_START = "<mantra>"
    MANTRA_END = "</mantra>"
    SANSKRIT_START = "<sanskrit>"
    SANSKRIT_END = "</sanskrit>"
    WISDOM_TOKEN = "<wisdom>"

    SPECIAL_TOKENS = [
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        SEP_TOKEN,
        OM_TOKEN,
        MANTRA_START,
        MANTRA_END,
        SANSKRIT_START,
        SANSKRIT_END,
        WISDOM_TOKEN,
    ]

    # Common Sanskrit/Pali terms to preserve as single tokens
    DHARMIC_TERMS = [
        # Core concepts
        "dharma",
        "karma",
        "moksha",
        "nirvana",
        "samsara",
        "maya",
        "atman",
        "brahman",
        "prakriti",
        "purusha",
        "shakti",
        "kundalini",
        # Yoga terms
        "yoga",
        "asana",
        "pranayama",
        "pratyahara",
        "dharana",
        "dhyana",
        "samadhi",
        "chakra",
        "nadi",
        "prana",
        "apana",
        "udana",
        "vyana",
        "samana",
        # Practices
        "japa",
        "mantra",
        "mudra",
        "bandha",
        "kriya",
        "tapas",
        "svadhyaya",
        # States
        "ananda",
        "shanti",
        "prema",
        "karuna",
        "metta",
        "mudita",
        "upeksha",
        # Sacred texts
        "veda",
        "upanishad",
        "gita",
        "sutra",
        "purana",
        "tantra",
        "agama",
        # Mantras
        "namaste",
        "namaskar",
        "swaha",
        "namah",
        # Titles
        "guru",
        "swami",
        "rishi",
        "muni",
        "acharya",
        "maharishi",
        # Buddhist terms
        "buddha",
        "sangha",
        "bodhisattva",
        "sunyata",
        "prajna",
        "sila",
        "dukkha",
        "anicca",
        "anatta",
        "vipassana",
        "shamatha",
        # Philosophical schools
        "vedanta",
        "advaita",
        "dvaita",
        "samkhya",
        "nyaya",
        "vaisheshika",
        # Chakras
        "muladhara",
        "svadhisthana",
        "manipura",
        "anahata",
        "vishuddha",
        "ajna",
        "sahasrara",
    ]

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ):
        """
        Initialize the DharmaTokenizer.

        Args:
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum frequency for a token to be included
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # BPE merges
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Initialize with special tokens
        self._init_special_tokens()

        # Compile regex patterns
        self._compile_patterns()

        self.is_trained = False

    def _init_special_tokens(self):
        """Initialize special tokens"""
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        # Add dharmic terms as special tokens
        next_id = len(self.SPECIAL_TOKENS)
        for term in self.DHARMIC_TERMS:
            if term not in self.token_to_id:
                self.token_to_id[term] = next_id
                self.id_to_token[next_id] = term
                next_id += 1

    def _compile_patterns(self):
        """Compile regex patterns for tokenization"""
        # Pattern for splitting text (Python's re compatible version)
        # Matches contractions, letters, numbers, punctuation, and whitespace
        self.split_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-ZāīūṛṝḷḹēōṃḥñṅṭḍṇśṣĀĪŪṚṜḶḸĒŌṂḤÑṄṬḌṆŚṢ]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
            re.UNICODE,
        )

        # Pattern for dharmic terms (case insensitive)
        dharmic_pattern = "|".join(re.escape(term) for term in self.DHARMIC_TERMS)
        self.dharmic_pattern = re.compile(f"({dharmic_pattern})", re.IGNORECASE)

        # Pattern for Sanskrit with diacritics
        self.sanskrit_pattern = re.compile(r"[āīūṛṝḷḹēōṃḥñṅṭḍṇśṣ]+", re.IGNORECASE)

    @property
    def vocab_len(self) -> int:
        """Get current vocabulary size"""
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def _get_pairs(self, word: List[str]) -> set:
        """Get all adjacent pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _apply_bpe(self, token: str) -> List[str]:
        """Apply BPE to a single token"""
        if not self.merges:
            return list(token)

        word = list(token)

        while len(word) >= 2:
            pairs = self._get_pairs(word)

            # Find the pair with lowest rank
            min_pair = None
            min_rank = float("inf")

            for pair in pairs:
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair

            if min_pair is None:
                break

            # Merge the pair
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == min_pair[0]
                    and word[i + 1] == min_pair[1]
                ):
                    new_word.append(min_pair[0] + min_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return word

    def train(self, texts: List[str], verbose: bool = True):
        """
        Train the tokenizer on a corpus of texts.

        Args:
            texts: List of training texts
            verbose: Whether to print progress
        """
        if verbose:
            logger.info("🕉️ Training DharmaTokenizer...")

        # Count word frequencies
        word_freqs = Counter()

        for text in texts:
            # Preprocess
            text = text.lower()

            # Extract words
            words = re.findall(r"\b\w+\b", text)
            word_freqs.update(words)

        # Initialize vocabulary with characters
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)

        # Add characters to vocabulary
        next_id = len(self.token_to_id)
        for char in sorted(vocab):
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        # Build word representations
        splits = {word: list(word) for word in word_freqs}

        # BPE training
        num_merges = self.vocab_size - len(self.token_to_id)

        for i in range(num_merges):
            # Count pairs
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                symbols = splits[word]
                if len(symbols) >= 2:
                    for j in range(len(symbols) - 1):
                        pair = (symbols[j], symbols[j + 1])
                        pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = pair_freqs.most_common(1)[0][0]

            # Merge pair
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = next_id
                self.id_to_token[next_id] = new_token
                next_id += 1

            # Store merge
            self.merges[best_pair] = new_token
            self.merge_ranks[best_pair] = i

            # Update splits
            for word in splits:
                symbols = splits[word]
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if (
                        j < len(symbols) - 1
                        and symbols[j] == best_pair[0]
                        and symbols[j + 1] == best_pair[1]
                    ):
                        new_symbols.append(new_token)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                splits[word] = new_symbols

            if verbose and (i + 1) % 1000 == 0:
                logger.info(f"   Completed {i+1}/{num_merges} merges")

        self.is_trained = True

        if verbose:
            logger.info(
                f"✅ Tokenizer trained with vocabulary size: {len(self.token_to_id)}"
            )

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            max_length: Maximum length (for truncation/padding)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length

        Returns:
            List of token IDs
        """
        # Lowercase
        text = text.lower()

        # Tokenize
        tokens = []

        # Find dharmic terms first
        parts = self.dharmic_pattern.split(text)

        for part in parts:
            if not part:
                continue

            # Check if it's a dharmic term
            if part.lower() in [t.lower() for t in self.DHARMIC_TERMS]:
                tokens.append(part.lower())
            else:
                # Apply BPE to other parts
                words = re.findall(r"\S+|\s+", part)
                for word in words:
                    if word.isspace():
                        tokens.append(word)
                    else:
                        bpe_tokens = self._apply_bpe(word)
                        tokens.extend(bpe_tokens)

        # Convert to IDs
        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.unk_token_id)

        if add_special_tokens:
            ids.append(self.eos_token_id)

        # Truncation
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_special_tokens:
                ids[-1] = self.eos_token_id

        # Padding
        if padding and max_length is not None and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        tokens = []

        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]

                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue

                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self.UNK_TOKEN)

        # Join tokens
        text = "".join(tokens)

        return text

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Tokenize text(s).

        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            max_length: Maximum length
            padding: Whether to pad
            truncation: Whether to truncate
            return_tensors: "pt" for PyTorch tensors

        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Encode all texts
        all_ids = []
        for t in texts:
            ids = self.encode(
                t,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )
            all_ids.append(ids)

        # Pad to same length if needed
        if padding and max_length is None:
            max_len = max(len(ids) for ids in all_ids)
            all_ids = [
                ids + [self.pad_token_id] * (max_len - len(ids)) for ids in all_ids
            ]

        # Create attention masks
        attention_masks = [
            [1 if id != self.pad_token_id else 0 for id in ids] for ids in all_ids
        ]

        result = {
            "input_ids": all_ids,
            "attention_mask": attention_masks,
        }

        # Convert to tensors
        if return_tensors == "pt":
            import torch

            result["input_ids"] = torch.tensor(all_ids)
            result["attention_mask"] = torch.tensor(attention_masks)

        return result

    def save(self, path: str):
        """Save tokenizer to directory"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocabulary
        vocab_path = path / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        # Save merges
        merges_path = path / "merges.json"
        merges_serializable = {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()}
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(merges_serializable, f, ensure_ascii=False, indent=2)

        # Save config
        config_path = path / "tokenizer_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "min_frequency": self.min_frequency,
                },
                f,
                indent=2,
            )

        logger.info(f"🕉️ Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DharmaTokenizer":
        """Load tokenizer from directory"""
        path = Path(path)

        # Load config
        with open(path / "tokenizer_config.json", "r") as f:
            config = json.load(f)

        tokenizer = cls(**config)

        # Load vocabulary
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            tokenizer.token_to_id = json.load(f)
            tokenizer.id_to_token = {
                int(v): k for k, v in tokenizer.token_to_id.items()
            }

        # Load merges
        with open(path / "merges.json", "r", encoding="utf-8") as f:
            merges_serializable = json.load(f)
            tokenizer.merges = {}
            tokenizer.merge_ranks = {}
            for i, (k, v) in enumerate(merges_serializable.items()):
                parts = k.split("|||")
                pair = (parts[0], parts[1])
                tokenizer.merges[pair] = v
                tokenizer.merge_ranks[pair] = i

        tokenizer.is_trained = True

        logger.info(f"🕉️ Tokenizer loaded from {path}")

        return tokenizer


def test_tokenizer():
    """Test the DharmaTokenizer"""
    print("=" * 60)
    print("🕉️ Testing DharmaTokenizer")
    print("=" * 60)

    # Create tokenizer
    print("\n1. Creating tokenizer...")
    tokenizer = DharmaTokenizer(vocab_size=1000)

    # Training corpus
    training_texts = [
        "Dharma is the cosmic law and order. Karma is the law of cause and effect.",
        "Yoga helps unite body, mind and spirit. Practice pranayama daily.",
        "The Bhagavad Gita teaches us about dharma and moksha.",
        "Meditation brings peace and awareness. Om shanti shanti shanti.",
        "The guru teaches the path to nirvana through bhakti and jnana.",
        "Atman is the individual soul, Brahman is the universal consciousness.",
        "Practice ahimsa - non-violence in thought, word and deed.",
        "The chakras are energy centers. Kundalini awakens spiritual power.",
    ]

    # Train
    print("\n2. Training tokenizer...")
    tokenizer.train(training_texts)
    print(f"   Vocabulary size: {tokenizer.vocab_len}")

    # Test encoding
    print("\n3. Testing encoding...")
    test_text = "What is dharma and karma?"
    encoded = tokenizer.encode(test_text)
    print(f"   Text: '{test_text}'")
    print(f"   Encoded: {encoded}")

    # Test decoding
    print("\n4. Testing decoding...")
    decoded = tokenizer.decode(encoded)
    print(f"   Decoded: '{decoded}'")

    # Test special tokens
    print("\n5. Dharmic terms as special tokens:")
    for term in ["dharma", "karma", "yoga", "meditation"]:
        if term in tokenizer.token_to_id:
            print(f"   {term}: ID {tokenizer.token_to_id[term]}")

    # Test batch encoding
    print("\n6. Testing batch encoding...")
    batch = tokenizer(
        ["What is dharma?", "How to meditate?"], padding=True, return_tensors="pt"
    )
    print(f"   Input IDs shape: {batch['input_ids'].shape}")
    print(f"   Attention mask shape: {batch['attention_mask'].shape}")

    # Test save/load
    print("\n7. Testing save/load...")
    save_path = "./cache/dharma_tokenizer_test"
    tokenizer.save(save_path)
    loaded = DharmaTokenizer.load(save_path)
    print(f"   Loaded vocabulary size: {loaded.vocab_len}")

    # Cleanup
    import shutil

    shutil.rmtree(save_path)

    print("\n" + "=" * 60)
    print("✅ DharmaTokenizer test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_tokenizer()
