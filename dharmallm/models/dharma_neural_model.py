"""
🕉️ DharmaMind Custom Neural Network Model
==========================================

Our OWN LLM - A custom transformer model designed and trained specifically
for dharmic wisdom, spiritual guidance, and philosophical understanding.

This is NOT a wrapper around existing models - this IS our model.

Architecture:
- Custom Transformer decoder (GPT-style)
- Dharmic embedding layer with spiritual concept encoding
- Multi-head self-attention with dharmic attention bias
- Feed-forward networks with wisdom projection

Features:
- Trained from scratch on dharmic texts
- Custom vocabulary with Sanskrit/Pali terms
- Spiritual concept embeddings
- Dharmic response generation

Author: DharmaMind Team
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DharmaModelConfig:
    """Configuration for DharmaMind's custom LLM"""

    # Model architecture
    vocab_size: int = 32000  # Vocabulary size (includes dharmic terms)
    hidden_size: int = 768  # Hidden dimension
    num_layers: int = 12  # Number of transformer layers
    num_heads: int = 12  # Number of attention heads
    intermediate_size: int = 3072  # FFN intermediate size
    max_position: int = 1024  # Maximum sequence length

    # Dharmic-specific
    num_dharmic_concepts: int = 108  # Sacred number of dharmic concepts
    dharmic_embedding_dim: int = 64  # Dimension for dharmic concept embeddings

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Training
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6

    # Model identification
    model_type: str = "dharmallm"
    model_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "max_position": self.max_position,
            "num_dharmic_concepts": self.num_dharmic_concepts,
            "dharmic_embedding_dim": self.dharmic_embedding_dim,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "model_type": self.model_type,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DharmaModelConfig":
        return cls(
            **{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DharmaModelConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# Pre-defined model sizes
DHARMA_MODEL_CONFIGS = {
    "dharma-tiny": DharmaModelConfig(
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=1024,
        vocab_size=16000,
    ),
    "dharma-small": DharmaModelConfig(
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        intermediate_size=2048,
        vocab_size=24000,
    ),
    "dharma-base": DharmaModelConfig(
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072,
        vocab_size=32000,
    ),
    "dharma-large": DharmaModelConfig(
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_size=4096,
        vocab_size=50000,
    ),
}


class DharmicConceptEmbedding(nn.Module):
    """
    Special embeddings for dharmic concepts.

    Encodes 108 fundamental dharmic concepts that influence
    the model's understanding and response generation.
    """

    # The 108 dharmic concepts (sacred number in Hinduism/Buddhism)
    DHARMIC_CONCEPTS = [
        # Core Principles (1-10)
        "dharma",
        "karma",
        "moksha",
        "ahimsa",
        "satya",
        "asteya",
        "brahmacharya",
        "aparigraha",
        "saucha",
        "santosha",
        # Yoga Paths (11-20)
        "jnana",
        "bhakti",
        "raja",
        "karma_yoga",
        "hatha",
        "kundalini",
        "tantra",
        "mantra",
        "laya",
        "kriya",
        # States of Being (21-30)
        "samadhi",
        "nirvana",
        "satori",
        "turiya",
        "ananda",
        "shanti",
        "prema",
        "karuna",
        "mudita",
        "upeksha",
        # Philosophical Concepts (31-40)
        "atman",
        "brahman",
        "maya",
        "avidya",
        "prakriti",
        "purusha",
        "gunas",
        "sattva",
        "rajas",
        "tamas",
        # Chakras (41-47)
        "muladhara",
        "svadhisthana",
        "manipura",
        "anahata",
        "vishuddha",
        "ajna",
        "sahasrara",
        # Koshas (48-52)
        "annamaya",
        "pranamaya",
        "manomaya",
        "vijnanamaya",
        "anandamaya",
        # Buddhist Concepts (53-62)
        "dukkha",
        "anicca",
        "anatta",
        "sila",
        "samadhi_buddhist",
        "panna",
        "metta",
        "sati",
        "viriya",
        "piti",
        # Vedantic Terms (63-72)
        "viveka",
        "vairagya",
        "shatsampat",
        "mumukshutva",
        "shraddha",
        "sadhana",
        "abhyasa",
        "vichara",
        "nididhyasana",
        "aparoksha",
        # Sacred Texts (73-82)
        "veda",
        "upanishad",
        "gita",
        "sutra",
        "shastra",
        "purana",
        "tantra_text",
        "agama",
        "smriti",
        "shruti",
        # Spiritual Practices (83-92)
        "japa",
        "dhyana",
        "pranayama",
        "asana",
        "pratyahara",
        "dharana",
        "tapas",
        "svadhyaya",
        "ishvara_pranidhana",
        "seva",
        # Teachers and Lineages (93-100)
        "guru",
        "acharya",
        "rishi",
        "muni",
        "swami",
        "yogi",
        "siddha",
        "avatar",
        # Additional Sacred Concepts (101-108)
        "om",
        "namaste",
        "puja",
        "yajna",
        "tirtha",
        "darshan",
        "prasad",
        "diksha",
    ]

    def __init__(self, config: DharmaModelConfig):
        super().__init__()
        self.config = config

        # Embedding for each dharmic concept
        self.concept_embeddings = nn.Embedding(
            config.num_dharmic_concepts, config.dharmic_embedding_dim
        )

        # Projection to hidden size
        self.projection = nn.Linear(config.dharmic_embedding_dim, config.hidden_size)

        # Concept detection weights (learned)
        self.concept_detector = nn.Linear(
            config.hidden_size, config.num_dharmic_concepts
        )

    def detect_concepts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Detect which dharmic concepts are relevant to the input"""
        # Shape: (batch, seq_len, num_concepts)
        concept_scores = torch.sigmoid(self.concept_detector(hidden_states))
        return concept_scores

    def get_concept_bias(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get dharmic concept bias to add to hidden states"""
        # Detect relevant concepts
        concept_scores = self.detect_concepts(hidden_states)

        # Get weighted combination of concept embeddings
        # Shape: (batch, seq_len, dharmic_embedding_dim)
        concept_ids = torch.arange(
            self.config.num_dharmic_concepts, device=hidden_states.device
        )
        all_concepts = self.concept_embeddings(concept_ids)  # (num_concepts, embed_dim)

        # Weighted sum: (batch, seq, num_concepts) @ (num_concepts, embed_dim)
        weighted_concepts = torch.matmul(concept_scores, all_concepts)

        # Project to hidden size
        concept_bias = self.projection(weighted_concepts)

        return concept_bias


class DharmaAttention(nn.Module):
    """
    Multi-head self-attention with dharmic awareness.

    Includes special attention patterns that emphasize
    dharmic concepts and spiritual coherence.
    """

    def __init__(self, config: DharmaModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5

        # QKV projection
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dharmic_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1
        ).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # Padding mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # Add dharmic bias if provided
        if dharmic_bias is not None:
            attn_output = attn_output + 0.1 * dharmic_bias

        return attn_output


class DharmaFeedForward(nn.Module):
    """Feed-forward network with GELU activation"""

    def __init__(self, config: DharmaModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DharmaTransformerBlock(nn.Module):
    """Single transformer block with dharmic awareness"""

    def __init__(self, config: DharmaModelConfig):
        super().__init__()
        self.attention = DharmaAttention(config)
        self.feed_forward = DharmaFeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dharmic_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, dharmic_bias)
        hidden_states = residual + hidden_states

        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DharmaLLM(nn.Module):
    """
    🕉️ DharmaMind's Own Language Model

    A custom transformer-based language model designed specifically
    for generating dharmic wisdom, spiritual guidance, and
    philosophical insights.

    This is OUR model, trained from scratch on dharmic texts.
    """

    def __init__(self, config: DharmaModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Position embeddings
        self.position_embedding = nn.Embedding(config.max_position, config.hidden_size)

        # Dharmic concept embeddings
        self.dharmic_concepts = DharmicConceptEmbedding(config)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [DharmaTransformerBlock(config) for _ in range(config.num_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Output head (tied with token embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        # Log model info
        self._log_model_info()

    def _init_weights(self, module):
        """Initialize weights with dharmic precision"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _log_model_info(self):
        """Log model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"🕉️ DharmaLLM initialized:")
        logger.info(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Layers: {self.config.num_layers}")
        logger.info(f"   Hidden size: {self.config.hidden_size}")
        logger.info(f"   Attention heads: {self.config.num_heads}")
        logger.info(f"   Vocab size: {self.config.vocab_size}")

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DharmaLLM.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Labels for language modeling loss (batch, seq_len)

        Returns:
            Dictionary with logits and optional loss
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position embeddings
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        position_embeds = self.position_embedding(positions)

        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Get dharmic concept bias
        dharmic_bias = self.dharmic_concepts.get_concept_bias(hidden_states)

        # Process attention mask
        if attention_mask is not None:
            # Convert to attention bias
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, dharmic_bias)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or use greedy
            eos_token_id: End of sequence token

        Returns:
            Generated token IDs
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Truncate if too long
            idx_cond = (
                input_ids
                if input_ids.size(1) <= self.config.max_position
                else input_ids[:, -self.config.max_position :]
            )

            # Forward pass
            outputs = self.forward(idx_cond)
            logits = outputs["logits"][:, -1, :]  # Last position

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

    def save_pretrained(self, save_path: str):
        """Save model and config"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(str(save_path / "config.json"))

        # Save model weights
        torch.save(self.state_dict(), str(save_path / "model.pt"))

        logger.info(f"🕉️ DharmaLLM saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cpu") -> "DharmaLLM":
        """Load model from saved weights"""
        load_path = Path(load_path)

        # Load config
        config = DharmaModelConfig.load(str(load_path / "config.json"))

        # Create model
        model = cls(config)

        # Load weights
        state_dict = torch.load(str(load_path / "model.pt"), map_location=device)
        model.load_state_dict(state_dict)

        logger.info(f"🕉️ DharmaLLM loaded from {load_path}")

        return model


def create_dharma_model(
    size: str = "dharma-small",
    custom_config: Optional[DharmaModelConfig] = None,
) -> DharmaLLM:
    """
    Create a DharmaLLM model.

    Args:
        size: Model size ("dharma-tiny", "dharma-small", "dharma-base", "dharma-large")
        custom_config: Optional custom configuration

    Returns:
        DharmaLLM model instance
    """
    if custom_config is not None:
        config = custom_config
    elif size in DHARMA_MODEL_CONFIGS:
        config = DHARMA_MODEL_CONFIGS[size]
    else:
        raise ValueError(
            f"Unknown model size: {size}. Choose from {list(DHARMA_MODEL_CONFIGS.keys())}"
        )

    return DharmaLLM(config)


def test_dharma_model():
    """Test the DharmaLLM model"""
    print("=" * 60)
    print("🕉️ Testing DharmaMind's Own LLM")
    print("=" * 60)

    # Create model
    print("\n1. Creating DharmaLLM (dharma-tiny)...")
    model = create_dharma_model("dharma-tiny")

    # Model info
    print(f"\n2. Model Statistics:")
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   Layers: {model.config.num_layers}")
    print(f"   Hidden size: {model.config.hidden_size}")
    print(f"   Vocab size: {model.config.vocab_size}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)
    print(f"   Logits shape: {outputs['logits'].shape}")

    # Test with labels
    print("\n4. Testing with labels (loss calculation)...")
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    outputs = model(input_ids, labels=labels)
    print(f"   Loss: {outputs['loss'].item():.4f}")

    # Test generation
    print("\n5. Testing generation...")
    start_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    generated = model.generate(start_ids, max_new_tokens=20)
    print(f"   Input length: 5")
    print(f"   Generated length: {generated.shape[1]}")

    # Test save/load
    print("\n6. Testing save/load...")
    save_path = "./cache/dharma_model_test"
    model.save_pretrained(save_path)
    loaded_model = DharmaLLM.from_pretrained(save_path)
    print(f"   Saved and loaded successfully!")

    # Cleanup
    import shutil

    shutil.rmtree(save_path)

    print("\n" + "=" * 60)
    print("✅ DharmaLLM test complete!")
    print("=" * 60)

    # Print model sizes
    print("\n📊 Available DharmaLLM sizes:")
    for name, config in DHARMA_MODEL_CONFIGS.items():
        temp_model = create_dharma_model(name)
        params = temp_model.get_num_params()
        print(f"   {name}: {params:,} parameters ({params/1e6:.1f}M)")
        del temp_model


if __name__ == "__main__":
    test_dharma_model()
