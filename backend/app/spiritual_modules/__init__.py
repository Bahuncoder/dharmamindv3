"""
Spiritual Modules Package
========================

This package contains all the spiritual modules for the DharmaMind system.
Each module rep        "artha": get_artha_module(),
        "kama": get_kama_module(),
        "varna": get_varna_module(),
    }sents a different aspect of spiritual wisdom and guidance.

Available Modules:
- SpiritualRouter: Central routing system for spiritual paths
- DharmaModule: Righteous path and moral guidance
- KarmaModule: Action and consequence processing
- MokshaModule: Liberation and enlightenment guidance
- JnanaModule: Knowledge and wisdom processing
- SevaModule: Service and compassion practices
- BhaktiModule: Divine love and devotion practices
- YogaChakra: Complete eight-limbed yoga path system

Each module provides specialized guidance based on ancient Hindu and Buddhist
wisdom traditions, adapted for modern spiritual seekers.
"""

from .spiritual_router import SpiritualRouter, get_spiritual_router
from .dharma_module import DharmaModule, get_dharma_module
from .karma_module import KarmaModule, get_karma_module
from .moksha_module import MokshaModule, get_moksha_module
from .jnana_module import JnanaModule, get_jnana_module
from .seva_module import SevaModule, get_seva_module
from .bhakti_module import BhaktiModule, get_bhakti_module
from .yoga_module import YogaChakra, get_yoga_module
from .ahimsa_module import AhimsaModule, get_ahimsa_module
from .atman_module import AtmanModule, get_atman_module
from .shakti_module import ShaktiModule, get_shakti_module
from .guru_module import GuruModule, get_guru_module
from .satya_module import SatyaModule, get_satya_module
from .tapas_module import TapasModule, get_tapas_module
from .wellness_module import WellnessModule, get_wellness_module
from .leadership_module import LeadershipModule, get_leadership_module
from .clarity_module import ClarityModule, get_clarity_module
from .manas_module import ManasModule, get_manas_module
from .grihastha_module import GrihasthaModule, get_grihastha_module
from .satsang_module import SatsangModule, get_satsang_module
from .dhyana_module import DhyanaModule, get_dhyana_module
from .sankalpa_module import SankalpaModule, get_sankalpa_module
from .shraddha_module import ShraddhaModule, get_shraddha_module
from .ananda_module import AnandaModule, get_ananda_module
from .smarana_module import SmaranaModule, get_smarana_module
from .shanti_module import ShantiModule, get_shanti_module
from .ahamkara_module import AhamkaraModule, get_ahamkara_module
from .raksha_module import RakshaModule, get_raksha_module
from .artha_module import get_artha_module, create_artha_guidance, ArthaResponse
from .kama_module import get_kama_module, create_kama_guidance, KamaResponse
from .varna_module import get_varna_module, create_varna_guidance, VarnaResponse
from .raashi_module import RaashiModule, get_raashi_module
from .health_crisis_module import HealthCrisisModule, get_health_crisis_module
from .career_crisis_module import CareerCrisisModule, get_career_crisis_module
from .financial_crisis_module import (
    FinancialCrisisModule, 
    get_financial_crisis_module
)

__all__ = [
    "SpiritualRouter",
    "DharmaModule",
    "KarmaModule",
    "MokshaModule",
    "JnanaModule",
    "SevaModule",
    "BhaktiModule",
    "YogaChakra",
    "AhimsaModule",
    "AtmanModule",
    "ShaktiModule",
    "GuruModule",
    "SatyaModule",
    "TapasModule",
    "WellnessModule",
    "LeadershipModule",
    "ClarityModule",
    "ManasModule",
    "GrihasthaModule",
    "SatsangModule",
    "DhyanaModule",
    "SankalpaModule",
    "ShraddhaModule",
    "AnandaModule",
    "SmaranaModule",
    "ShantiModule",
    "AhamkaraModule",
    "RakshaModule",
    "RaashiModule",
    "HealthCrisisModule",
    "CareerCrisisModule",
    "FinancialCrisisModule",
    "get_spiritual_router",
    "get_dharma_module",
    "get_karma_module",
    "get_moksha_module",
    "get_jnana_module",
    "get_seva_module",
    "get_bhakti_module",
    "get_yoga_module",
    "get_ahimsa_module",
    "get_atman_module",
    "get_shakti_module",
    "get_guru_module",
    "get_satya_module",
    "get_tapas_module",
    "get_wellness_module",
    "get_leadership_module",
    "get_clarity_module",
    "get_manas_module",
    "get_grihastha_module",
    "get_satsang_module",
    "get_dhyana_module",
    "get_sankalpa_module",
    "get_shraddha_module",
    "get_ananda_module",
    "get_smarana_module",
    "get_shanti_module",
    "get_ahamkara_module",
    "get_raksha_module",
    "get_artha_module",
    "get_kama_module",
    "get_varna_module",
    "get_raashi_module",
    "get_health_crisis_module",
    "get_career_crisis_module",
    "get_financial_crisis_module"
]

# Module metadata
__version__ = "1.0.0"
__author__ = "DharmaMind Team"
__description__ = "Spiritual wisdom modules for conscious AI guidance"


def get_all_modules():
    """Get instances of all spiritual modules"""
    return {
        "spiritual_router": get_spiritual_router(),
        "dharma": get_dharma_module(),
        "karma": get_karma_module(),
        "moksha": get_moksha_module(),
        "jnana": get_jnana_module(),
        "seva": get_seva_module(),
        "bhakti": get_bhakti_module(),
        "yoga": get_yoga_module(),
        "ahimsa": get_ahimsa_module(),
        "atman": get_atman_module(),
        "shakti": get_shakti_module(),
        "guru": get_guru_module(),
        "satya": get_satya_module(),
        "tapas": get_tapas_module(),
        "wellness": get_wellness_module(),
        "leadership": get_leadership_module(),
        "clarity": get_clarity_module(),
        "manas": get_manas_module(),
        "grihastha": get_grihastha_module(),
        "satsang": get_satsang_module(),
        "dhyana": get_dhyana_module(),
        "sankalpa": get_sankalpa_module(),
        "shraddha": get_shraddha_module(),
        "ananda": get_ananda_module(),
        "smarana": get_smarana_module(),
        "shanti": get_shanti_module(),
        "ahamkara": get_ahamkara_module(),
        "raksha": get_raksha_module(),
        "artha": get_artha_module(),
        "kama": get_kama_module(),
        "varna": get_varna_module(),
        "raashi": get_raashi_module(),
        "health_crisis": get_health_crisis_module(),
        "career_crisis": get_career_crisis_module(),
        "financial_crisis": get_financial_crisis_module()
    }


def get_module_by_name(module_name: str):
    """Get a specific module by name"""
    modules = {
        "spiritual_router": get_spiritual_router,
        "dharma": get_dharma_module,
        "karma": get_karma_module,
        "moksha": get_moksha_module,
        "jnana": get_jnana_module,
        "seva": get_seva_module,
        "bhakti": get_bhakti_module,
        "yoga": get_yoga_module,
        "ahimsa": get_ahimsa_module,
        "atman": get_atman_module,
        "shakti": get_shakti_module,
        "guru": get_guru_module,
        "satya": get_satya_module,
        "tapas": get_tapas_module,
        "wellness": get_wellness_module,
        "leadership": get_leadership_module,
        "clarity": get_clarity_module,
        "manas": get_manas_module,
        "grihastha": get_grihastha_module,
        "satsang": get_satsang_module,
        "dhyana": get_dhyana_module,
        "sankalpa": get_sankalpa_module,
        "shraddha": get_shraddha_module,
        "ananda": get_ananda_module,
        "smarana": get_smarana_module,
        "shanti": get_shanti_module,
        "ahamkara": get_ahamkara_module,
        "raksha": get_raksha_module,
        "artha": get_artha_module,
        "kama": get_kama_module,
        "varna": get_varna_module,
        "raashi": get_raashi_module,
        "health_crisis": get_health_crisis_module,
        "career_crisis": get_career_crisis_module,
        "financial_crisis": get_financial_crisis_module
    }
    
    getter = modules.get(module_name.lower())
    if getter:
        return getter()
    else:
        raise ValueError(f"Unknown module: {module_name}")
