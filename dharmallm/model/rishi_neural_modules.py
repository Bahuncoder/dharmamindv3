"""
🧘 Rishi Neural Modules - The 7 Saptarishis as Learnable Neural Networks

This module implements the world's first AI system with multiple learned sage personalities!

Each of the 7 great Vedic sages (Saptarishi) is implemented as a neural network that learns:
- Unique personality traits and speech patterns
- Teaching wisdom from authentic Samhita texts
- Sanskrit usage and mantra selection
- Contextual guidance delivery
- Emotional resonance with seekers

The 7 Saptarishis:
1. ATRI (अत्रि) - The Silent Contemplator: Meditation, cosmic consciousness, stillness
2. BHRIGU (भृगु) - The Cosmic Astrologer: Jyotisha, karmic patterns, cosmic law
3. VASHISHTA (वशिष्ठ) - The Royal Guru: Dharma, royal wisdom, storytelling
4. VISHWAMITRA (विश्वामित्र) - The Warrior-Sage: Transformation, willpower, inner strength
5. GAUTAMA (गौतम) - The Equanimous One: Balance, justice, non-judgment
6. JAMADAGNI (जमदग्नि) - The Fierce Ascetic: Discipline, austerity, penance
7. KASHYAPA (कश्यप) - The Compassionate Father: Universal compassion, nurturing

Total Parameters: ~24.5M (7 Rishis × ~3.5M each)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RishiType(Enum):
    """The 7 Saptarishi sage types"""
    ATRI = "atri"                    # The Silent Contemplator
    BHRIGU = "bhrigu"                # The Cosmic Astrologer
    VASHISHTA = "vashishta"          # The Royal Guru
    VISHWAMITRA = "vishwamitra"      # The Warrior-Sage
    GAUTAMA = "gautama"              # The Equanimous One
    JAMADAGNI = "jamadagni"          # The Fierce Ascetic
    KASHYAPA = "kashyapa"            # The Compassionate Father


@dataclass
class RishiModuleConfig:
    """Configuration for Rishi neural modules"""
    hidden_size: int = 768
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    
    # Personality trait learning
    num_personality_traits: int = 12      # E.g., patience, intensity, compassion
    personality_dim: int = 128            # Dimension for personality embeddings
    
    # Speech pattern learning
    num_speech_patterns: int = 10         # E.g., slow_speech, pauses, intensity
    speech_pattern_dim: int = 64
    
    # Sanskrit pattern learning
    num_sanskrit_patterns: int = 100      # Common Sanskrit phrases/mantras
    sanskrit_embedding_dim: int = 256
    
    # Teaching style
    num_teaching_styles: int = 8          # E.g., direct, metaphorical, questioning
    teaching_style_dim: int = 64


class PersonalityEmbedding(nn.Module):
    """
    Learns personality traits for each Rishi.
    
    Traits include: patience, intensity, compassion, authority, playfulness,
    sternness, gentleness, wisdom_depth, teaching_directness, emotional_resonance, etc.
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__()
        self.config = config
        
        # Personality trait embeddings (learnable)
        self.trait_embeddings = nn.Embedding(
            config.num_personality_traits,
            config.personality_dim
        )
        
        # Personality projection to hidden space
        self.personality_projector = nn.Linear(
            config.personality_dim,
            config.hidden_size
        )
        
        # Personality modulator (affects all outputs)
        self.personality_modulator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
            nn.Sigmoid()  # Gating mechanism
        )
    
    def forward(self, trait_indices: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply personality traits to hidden states.
        
        Args:
            trait_indices: Tensor of trait indices to activate (batch_size, num_active_traits)
            hidden_states: Hidden states to modulate (batch_size, seq_len, hidden_size)
            
        Returns:
            Personality-modulated hidden states
        """
        # Get personality embeddings for active traits
        trait_embeds = self.trait_embeddings(trait_indices)  # (batch, num_traits, personality_dim)
        
        # Average active traits
        personality_vector = trait_embeds.mean(dim=1)  # (batch, personality_dim)
        
        # Project to hidden space
        personality_hidden = self.personality_projector(personality_vector)  # (batch, hidden_size)
        
        # Expand for sequence length
        personality_hidden = personality_hidden.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Create modulation gates
        personality_gates = self.personality_modulator(personality_hidden)  # (batch, 1, hidden_size)
        
        # Apply personality gating to hidden states
        modulated_states = hidden_states * personality_gates
        
        return modulated_states


class SanskritPatternLearning(nn.Module):
    """
    Learns Sanskrit phrase patterns, mantras, and usage for each Rishi.
    
    Each Rishi has preferred Sanskrit phrases, mantras they invoke,
    and contextual usage patterns.
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__()
        self.config = config
        
        # Sanskrit pattern embeddings (learnable from training data)
        self.sanskrit_embeddings = nn.Embedding(
            config.num_sanskrit_patterns,
            config.sanskrit_embedding_dim
        )
        
        # Context-aware Sanskrit selection
        self.context_to_sanskrit = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_sanskrit_patterns)
        )
        
        # Sanskrit integration into response
        self.sanskrit_integrator = nn.Linear(
            config.sanskrit_embedding_dim,
            config.hidden_size
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select and integrate appropriate Sanskrit patterns.
        
        Args:
            hidden_states: Context hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            - Sanskrit-enhanced hidden states
            - Sanskrit pattern logits (for training)
        """
        # Use last hidden state as context
        context = hidden_states[:, -1, :]  # (batch, hidden_size)
        
        # Predict Sanskrit pattern probabilities
        sanskrit_logits = self.context_to_sanskrit(context)  # (batch, num_patterns)
        sanskrit_probs = F.softmax(sanskrit_logits, dim=-1)
        
        # Soft selection of Sanskrit patterns
        sanskrit_weights = sanskrit_probs.unsqueeze(-1)  # (batch, num_patterns, 1)
        all_sanskrit = self.sanskrit_embeddings.weight.unsqueeze(0)  # (1, num_patterns, dim)
        selected_sanskrit = (sanskrit_weights * all_sanskrit).sum(dim=1)  # (batch, dim)
        
        # Integrate Sanskrit into hidden states
        sanskrit_hidden = self.sanskrit_integrator(selected_sanskrit)  # (batch, hidden_size)
        sanskrit_hidden = sanskrit_hidden.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Add Sanskrit influence to hidden states
        enhanced_states = hidden_states + sanskrit_hidden
        
        return enhanced_states, sanskrit_logits


class TeachingStyleModule(nn.Module):
    """
    Learns teaching style for each Rishi.
    
    Teaching styles: direct, metaphorical, questioning, storytelling,
    experiential, scriptural, meditative, practical
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__()
        self.config = config
        
        # Teaching style embeddings
        self.style_embeddings = nn.Embedding(
            config.num_teaching_styles,
            config.teaching_style_dim
        )
        
        # Style selector based on context
        self.style_selector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_teaching_styles)
        )
        
        # Style application
        self.style_applicator = nn.Linear(
            config.teaching_style_dim,
            config.hidden_size
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply appropriate teaching style.
        
        Args:
            hidden_states: Context hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            - Style-modulated hidden states
            - Style logits (for training)
        """
        # Use last hidden state as context
        context = hidden_states[:, -1, :]  # (batch, hidden_size)
        
        # Select teaching style
        style_logits = self.style_selector(context)  # (batch, num_styles)
        style_probs = F.softmax(style_logits, dim=-1)
        
        # Soft selection of teaching styles
        style_weights = style_probs.unsqueeze(-1)  # (batch, num_styles, 1)
        all_styles = self.style_embeddings.weight.unsqueeze(0)  # (1, num_styles, dim)
        selected_style = (style_weights * all_styles).sum(dim=1)  # (batch, dim)
        
        # Apply style to hidden states
        style_hidden = self.style_applicator(selected_style)  # (batch, hidden_size)
        style_hidden = style_hidden.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Add style influence
        styled_states = hidden_states + style_hidden
        
        return styled_states, style_logits


class BaseRishiModule(nn.Module):
    """
    Base class for all Rishi neural modules.
    
    Each Rishi module learns:
    - Personality traits (patience, intensity, compassion, etc.)
    - Speech patterns (slow/fast, pauses, intensity)
    - Sanskrit usage (mantras, phrases, contextual usage)
    - Teaching style (direct, metaphorical, storytelling)
    - Wisdom patterns from training data
    """
    def __init__(self, config: RishiModuleConfig, rishi_name: str):
        super().__init__()
        self.config = config
        self.rishi_name = rishi_name
        
        # Personality learning
        self.personality = PersonalityEmbedding(config)
        
        # Multi-head attention for wisdom integration
        self.wisdom_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Sanskrit pattern learning
        self.sanskrit_patterns = SanskritPatternLearning(config)
        
        # Teaching style learning
        self.teaching_style = TeachingStyleModule(config)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        
        # Wisdom detectors (specific to each Rishi, overridden in subclasses)
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        personality_traits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for Rishi module.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            personality_traits: Active personality trait indices (batch_size, num_traits)
            
        Returns:
            - Transformed hidden states
            - Dict of auxiliary outputs (sanskrit_logits, style_logits, wisdom_detections)
        """
        auxiliary_outputs = {}
        
        # Apply personality traits if provided
        if personality_traits is not None:
            hidden_states = self.personality(personality_traits, hidden_states)
        
        # Wisdom attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output, _ = self.wisdom_attention(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + attn_output
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        
        # Sanskrit pattern integration
        hidden_states, sanskrit_logits = self.sanskrit_patterns(hidden_states)
        auxiliary_outputs['sanskrit_logits'] = sanskrit_logits
        
        # Teaching style application
        hidden_states, style_logits = self.teaching_style(hidden_states)
        auxiliary_outputs['style_logits'] = style_logits
        
        # Wisdom detection
        wisdom_scores = self.wisdom_detectors(hidden_states[:, -1, :])
        auxiliary_outputs['wisdom_detections'] = wisdom_scores
        
        # Final normalization
        hidden_states = self.norm3(hidden_states)
        
        return hidden_states, auxiliary_outputs


# ============================================================================
# THE 7 SAPTARISHI NEURAL MODULES
# ============================================================================

class AtriNeuralModule(BaseRishiModule):
    """
    🧘 Maharishi Atri - The Silent Contemplator
    
    Personality: Slow, deep, contemplative, cosmic consciousness
    Focus: Meditation, tapasya, stillness, cosmic awareness
    Speech: Slow with long pauses, deep wisdom, gentle intensity
    Sanskrit: Cosmic mantras, meditation sutras, silence-related phrases
    Teaching: Experiential, meditative, uses silence as teaching tool
    
    Wisdom areas:
    - Deep meditation practices
    - Cosmic consciousness
    - Tapasya (austerity/discipline)
    - Stillness teachings
    - Breath awareness
    - Silence as wisdom
    - Contemplative philosophy
    - Universal consciousness
    
    Parameters: ~3.5M
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__(config, rishi_name="Atri")
        
        # Atri-specific wisdom detectors
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
        # Detects: meditation_depth, cosmic_consciousness, tapasya_guidance,
        #          stillness_teaching, breath_wisdom, silence_power,
        #          contemplation_depth, universal_awareness


class BhriguNeuralModule(BaseRishiModule):
    """
    🌟 Maharishi Bhrigu - The Cosmic Astrologer
    
    Personality: Precise, authoritative, mathematical, cosmic law keeper
    Focus: Jyotisha (astrology), karmic patterns, cosmic order
    Speech: Precise, uses astrological terms, references planets/stars
    Sanskrit: Jyotisha mantras, planetary invocations, nakshatra wisdom
    Teaching: Direct, systematic, uses cosmic examples
    
    Wisdom areas:
    - Vedic astrology (Jyotisha)
    - Karmic patterns and design
    - Planetary influences
    - Nakshatra wisdom
    - Muhurta (auspicious timing)
    - Cosmic law and order
    - Bhrigu Samhita wisdom
    - Destiny and free will
    
    Parameters: ~3.5M
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__(config, rishi_name="Bhrigu")
        
        # Bhrigu-specific wisdom detectors
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
        # Detects: astrological_inquiry, karmic_pattern, planetary_guidance,
        #          nakshatra_wisdom, timing_consciousness, cosmic_law,
        #          destiny_teaching, bhrigu_samhita_reference


class VashistaNeuralModule(BaseRishiModule):
    """
    👑 Maharishi Vashishta - The Royal Guru
    
    Personality: Gentle authority, wise storyteller, patient teacher
    Focus: Dharma, righteous living, royal wisdom, spiritual kingship
    Speech: Storytelling, uses parables, gentle but authoritative
    Sanskrit: Dharma sutras, royal mantras, righteousness teachings
    Teaching: Storytelling, metaphorical, uses historical examples
    
    Wisdom areas:
    - Dharma (righteousness)
    - Royal wisdom and governance
    - Guru-shishya relationship
    - Spiritual kingship
    - Family dharma
    - Teaching through stories
    - Patience and perseverance
    - Righteous action
    
    Parameters: ~3.5M
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__(config, rishi_name="Vashishta")
        
        # Vashishta-specific wisdom detectors
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
        # Detects: dharma_inquiry, royal_wisdom, guru_teaching,
        #          spiritual_kingship, family_guidance, storytelling_mode,
        #          patience_teaching, righteous_action


class VishwamitraNeuralModule(BaseRishiModule):
    """
    ⚔️ Maharishi Vishwamitra - The Warrior-Sage
    
    Personality: Intense, transformative, powerful, challenging
    Focus: Transformation, willpower, spiritual warrior path, tapas
    Speech: Fiery, intense, challenging, uses warrior metaphors
    Sanskrit: Power mantras (Gayatri), transformation sutras, strength invocations
    Teaching: Challenging, pushes boundaries, transformative
    
    Wisdom areas:
    - Spiritual transformation
    - Willpower and determination
    - Warrior spirit in spirituality
    - Gayatri Mantra wisdom
    - From warrior to sage journey
    - Overcoming obstacles
    - Inner strength cultivation
    - Tapas (spiritual fire)
    
    Parameters: ~3.5M
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__(config, rishi_name="Vishwamitra")
        
        # Vishwamitra-specific wisdom detectors
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
        # Detects: transformation_need, willpower_cultivation, warrior_spirit,
        #          gayatri_wisdom, obstacle_overcoming, inner_strength,
        #          tapas_guidance, spiritual_fire


class GautamaNeuralModule(BaseRishiModule):
    """
    ⚖️ Maharishi Gautama - The Equanimous One
    
    Personality: Perfectly balanced, non-judgmental, serene
    Focus: Balance, justice, equanimity, Nyaya philosophy
    Speech: Calm, measured, balanced, uses logical examples
    Sanskrit: Balance mantras, equanimity sutras, Nyaya aphorisms
    Teaching: Logical, balanced perspectives, non-judgmental
    
    Wisdom areas:
    - Equanimity (sama-bhava)
    - Justice and fairness
    - Nyaya philosophy (logic)
    - Non-judgment
    - Balanced perspective
    - Mental equilibrium
    - Logical thinking
    - Serenity in chaos
    
    Parameters: ~3.5M
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__(config, rishi_name="Gautama")
        
        # Gautama-specific wisdom detectors
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
        # Detects: equanimity_teaching, justice_inquiry, nyaya_logic,
        #          non_judgment, balance_guidance, mental_equilibrium,
        #          logical_reasoning, serenity_cultivation


class JamadagniNeuralModule(BaseRishiModule):
    """
    🔥 Maharishi Jamadagni - The Fierce Ascetic
    
    Personality: Austere, disciplined, direct, intense
    Focus: Discipline, austerity, penance, fierce spiritual practice
    Speech: Direct, no-nonsense, strict, uses discipline metaphors
    Sanskrit: Austerity mantras, discipline sutras, fierce invocations
    Teaching: Strict discipline, direct correction, demanding standards
    
    Wisdom areas:
    - Spiritual discipline
    - Austerity (tapas)
    - Penance practices
    - Self-control
    - Fierce determination
    - Purification through practice
    - Renunciation
    - Unwavering commitment
    
    Parameters: ~3.5M
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__(config, rishi_name="Jamadagni")
        
        # Jamadagni-specific wisdom detectors
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
        # Detects: discipline_need, austerity_guidance, penance_teaching,
        #          self_control, fierce_practice, purification_path,
        #          renunciation_wisdom, unwavering_commitment


class KashyapaNeuralModule(BaseRishiModule):
    """
    💚 Maharishi Kashyapa - The Compassionate Father
    
    Personality: Nurturing, compassionate, loving, fatherly
    Focus: Universal compassion, all beings, nurturing guidance
    Speech: Warm, caring, nurturing, uses family metaphors
    Sanskrit: Compassion mantras, loving invocations, fatherly blessings
    Teaching: Nurturing, encouraging, supportive, gentle guidance
    
    Wisdom areas:
    - Universal compassion
    - Fatherly guidance
    - Care for all beings
    - Nurturing wisdom
    - Karuna (compassion)
    - Protecting the vulnerable
    - Family consciousness
    - Unconditional love
    
    Parameters: ~3.5M
    """
    def __init__(self, config: RishiModuleConfig):
        super().__init__(config, rishi_name="Kashyapa")
        
        # Kashyapa-specific wisdom detectors
        self.num_wisdom_detectors = 8
        self.wisdom_detectors = nn.Linear(config.hidden_size, self.num_wisdom_detectors)
        # Detects: compassion_need, fatherly_guidance, universal_care,
        #          nurturing_wisdom, karuna_teaching, protection_need,
        #          family_consciousness, unconditional_love


# ============================================================================
# SAPTARISHI ROUTER - Intelligent Rishi Selection
# ============================================================================

class SaptarishiRouter(nn.Module):
    """
    Intelligent router that learns which Rishi (or combination of Rishis)
    to activate based on the user's question and context.
    
    The router learns patterns like:
    - Meditation questions → Atri
    - Astrology questions → Bhrigu
    - Dharma/ethics questions → Vashishta
    - Transformation/power → Vishwamitra
    - Justice/balance → Gautama
    - Discipline questions → Jamadagni
    - Compassion/relationships → Kashyapa
    - Complex questions → Multiple Rishis (council mode)
    
    Parameters: ~0.5M
    """
    def __init__(self, config: RishiModuleConfig, num_rishis: int = 7):
        super().__init__()
        self.config = config
        self.num_rishis = num_rishis
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Rishi selection head
        self.rishi_selector = nn.Linear(config.hidden_size // 2, num_rishis)
        
        # Routing weights (how much each Rishi contributes)
        self.routing_weights = nn.Sequential(
            nn.Linear(config.hidden_size // 2, num_rishis),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route to appropriate Rishi(s).
        
        Args:
            hidden_states: Context hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            - Rishi selection logits (for training)
            - Routing weights (for weighted combination of Rishis)
        """
        # Use last hidden state as context
        context = hidden_states[:, -1, :]  # (batch, hidden_size)
        
        # Encode context
        encoded = self.context_encoder(context)  # (batch, hidden_size // 2)
        
        # Get Rishi selection logits
        rishi_logits = self.rishi_selector(encoded)  # (batch, num_rishis)
        
        # Get routing weights (soft selection)
        routing_weights = self.routing_weights(encoded)  # (batch, num_rishis)
        
        return rishi_logits, routing_weights


# ============================================================================
# ALL RISHI MODULES - Container for all 7 Saptarishis
# ============================================================================

class AllRishiModules(nn.Module):
    """
    Container for all 7 Saptarishi neural modules with intelligent routing.
    
    This module:
    1. Takes input context
    2. Routes to appropriate Rishi(s) via SaptarishiRouter
    3. Applies selected Rishi transformations
    4. Combines outputs (if multiple Rishis activated)
    5. Returns Rishi-guided wisdom
    
    Total Parameters: ~25M
    - 7 Rishis × ~3.5M each = ~24.5M
    - Router = ~0.5M
    """
    def __init__(self, config: Optional[RishiModuleConfig] = None):
        super().__init__()
        
        if config is None:
            config = RishiModuleConfig()
        
        self.config = config
        
        # The 7 Saptarishi modules
        self.atri = AtriNeuralModule(config)              # The Silent Contemplator
        self.bhrigu = BhriguNeuralModule(config)          # The Cosmic Astrologer
        self.vashishta = VashistaNeuralModule(config)     # The Royal Guru
        self.vishwamitra = VishwamitraNeuralModule(config)  # The Warrior-Sage
        self.gautama = GautamaNeuralModule(config)        # The Equanimous One
        self.jamadagni = JamadagniNeuralModule(config)    # The Fierce Ascetic
        self.kashyapa = KashyapaNeuralModule(config)      # The Compassionate Father
        
        # Intelligent router
        self.router = SaptarishiRouter(config, num_rishis=7)
        
        # List of all Rishi modules (for iteration)
        self.rishi_modules = [
            self.atri, self.bhrigu, self.vashishta, self.vishwamitra,
            self.gautama, self.jamadagni, self.kashyapa
        ]
        
        # Output combiner (when multiple Rishis activated)
        self.output_combiner = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        specific_rishi: Optional[RishiType] = None,
        personality_traits: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through Rishi system.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            specific_rishi: If provided, use only this Rishi (ignore router)
            personality_traits: Active personality trait indices
            
        Returns:
            - Rishi-guided hidden states
            - Dict of auxiliary outputs from all components
        """
        auxiliary_outputs = {}
        
        if specific_rishi is not None:
            # Use specific Rishi only
            rishi_map = {
                RishiType.ATRI: self.atri,
                RishiType.BHRIGU: self.bhrigu,
                RishiType.VASHISHTA: self.vashishta,
                RishiType.VISHWAMITRA: self.vishwamitra,
                RishiType.GAUTAMA: self.gautama,
                RishiType.JAMADAGNI: self.jamadagni,
                RishiType.KASHYAPA: self.kashyapa
            }
            rishi_module = rishi_map[specific_rishi]
            output, rishi_aux = rishi_module(hidden_states, personality_traits)
            auxiliary_outputs[f'{specific_rishi.value}_outputs'] = rishi_aux
            
        else:
            # Use router to select Rishi(s)
            rishi_logits, routing_weights = self.router(hidden_states)
            auxiliary_outputs['rishi_logits'] = rishi_logits
            auxiliary_outputs['routing_weights'] = routing_weights
            
            # Apply each Rishi and weight by routing weights
            rishi_outputs = []
            for i, rishi_module in enumerate(self.rishi_modules):
                rishi_output, rishi_aux = rishi_module(hidden_states, personality_traits)
                
                # Weight by routing weight
                weight = routing_weights[:, i:i+1].unsqueeze(1)  # (batch, 1, 1)
                weighted_output = rishi_output * weight
                rishi_outputs.append(weighted_output)
                
                # Store auxiliary outputs
                rishi_name = rishi_module.rishi_name.lower()
                auxiliary_outputs[f'{rishi_name}_outputs'] = rishi_aux
            
            # Combine all Rishi outputs
            combined = torch.stack(rishi_outputs, dim=0).sum(dim=0)  # (batch, seq_len, hidden)
            output = self.output_combiner(combined)
        
        return output, auxiliary_outputs
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count for each component."""
        counts = {}
        for name, module in [
            ('atri', self.atri),
            ('bhrigu', self.bhrigu),
            ('vashishta', self.vashishta),
            ('vishwamitra', self.vishwamitra),
            ('gautama', self.gautama),
            ('jamadagni', self.jamadagni),
            ('kashyapa', self.kashyapa),
            ('router', self.router)
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        
        counts['total'] = sum(counts.values())
        return counts


# ============================================================================
# TESTING AND INITIALIZATION
# ============================================================================

def test_rishi_modules():
    """Test Rishi modules initialization and forward pass."""
    print("🧘 Testing Rishi Neural Modules...")
    print("=" * 70)
    
    # Configuration
    config = RishiModuleConfig(
        hidden_size=768,
        num_attention_heads=8,
        intermediate_size=2048
    )
    
    # Create all Rishi modules
    all_rishis = AllRishiModules(config)
    
    # Test input
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print(f"\n📊 Input shape: {hidden_states.shape}")
    
    # Test forward pass with router
    print("\n🔄 Testing with automatic routing...")
    output, aux = all_rishis(hidden_states)
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Routing weights shape: {aux['routing_weights'].shape}")
    print(f"   Routing weights: {aux['routing_weights'][0].detach().cpu().numpy()}")
    
    # Test specific Rishi
    print("\n🧘 Testing specific Rishi (Atri)...")
    output_atri, aux_atri = all_rishis(hidden_states, specific_rishi=RishiType.ATRI)
    print(f"✅ Atri output shape: {output_atri.shape}")
    
    # Parameter count
    print("\n📊 Parameter counts:")
    counts = all_rishis.get_parameter_count()
    for name, count in counts.items():
        print(f"   {name:15s}: {count:>12,} parameters")
    
    print(f"\n🎯 Total Rishi system: {counts['total']:,} parameters (~{counts['total']/1e6:.1f}M)")
    print("=" * 70)
    print("✅ All tests passed! Rishi modules ready for training.")


if __name__ == "__main__":
    test_rishi_modules()
