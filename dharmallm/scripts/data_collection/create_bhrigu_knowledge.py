#!/usr/bin/env python3
"""
Bhrigu Knowledge Base Creator
==============================

Builds Rishi Bhrigu's astrology knowledge base with:
1. Vedic Astrology Fundamentals (Brihat Parashara Hora Shastra excerpts)
2. 27 Nakshatras (Lunar mansions)
3. 9 Planets (Grahas) and their influences
4. Birth Chart interpretation basics
5. Dasha system fundamentals

Bhrigu is the ancient sage of astrology and karmic patterns.
"""

import json
from pathlib import Path
from datetime import datetime


class BhriguKnowledgeBuilder:
    """Build Bhrigu's astrology knowledge base"""
    
    def __init__(self):
        self.base_path = Path("data/rishi_knowledge/bhrigu")
        self.created_files = []
    
    def create_all(self):
        """Create complete Bhrigu knowledge base"""
        print("\n" + "="*70)
        print("🔮 Building Bhrigu's Astrology Knowledge Base")
        print("="*70 + "\n")
        
        self.create_vedic_astrology_basics()
        self.create_nakshatras_database()
        self.create_planetary_wisdom()
        self.create_birth_chart_guide()
        self.create_dasha_system()
        self.create_summary()
        
        print("\n" + "="*70)
        print(f"✅ Created {len(self.created_files)} knowledge documents for Bhrigu")
        print("="*70 + "\n")
    
    def create_vedic_astrology_basics(self):
        """Create Vedic astrology fundamentals"""
        print("📖 Creating Vedic Astrology fundamentals...")
        
        basics = {
            "title": "Vedic Astrology Fundamentals",
            "source": "Brihat Parashara Hora Shastra",
            "sage": "Maharishi Parashara (Bhrigu lineage)",
            "fundamentals": [
                {
                    "concept": "Jyotish - The Science of Light",
                    "sanskrit": "ज्योतिष",
                    "description": "Vedic astrology (Jyotish) means 'science of light'. It reveals the karmic patterns written in the cosmic blueprint at birth.",
                    "purpose": "To understand one's dharma, karma, and life path through celestial influences"
                },
                {
                    "concept": "The Three Pillars",
                    "pillars": {
                        "1_Rashi": "The 12 zodiac signs representing different energies",
                        "2_Graha": "The 9 planets (including Rahu & Ketu) as karmic forces",
                        "3_Bhava": "The 12 houses representing life areas"
                    }
                },
                {
                    "concept": "Birth Chart (Kundali)",
                    "sanskrit": "कुण्डली",
                    "description": "The cosmic snapshot at the moment of birth, showing planetary positions in signs and houses",
                    "significance": "This chart is the map of one's karma from past lives and potential in this life"
                },
                {
                    "concept": "Ascendant (Lagna)",
                    "sanskrit": "लग्न",
                    "description": "The zodiac sign rising on the eastern horizon at birth",
                    "importance": "The Lagna represents the self, physical body, and life's overall direction"
                }
            ],
            "core_teachings": [
                "The planets do not cause events; they indicate karmic patterns",
                "Free will exists within the framework of karma",
                "Jyotish is a tool for self-awareness, not fatalism",
                "Remedies can mitigate negative karma and enhance positive karma"
            ]
        }
        
        # Save as JSON
        json_path = self.base_path / "vedic_astrology" / "astrology_fundamentals.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(basics, f, indent=2, ensure_ascii=False)
        self.created_files.append(str(json_path))
        
        # Save as readable text
        text_path = self.base_path / "vedic_astrology" / "astrology_fundamentals.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("VEDIC ASTROLOGY FUNDAMENTALS\n")
            f.write("From Brihat Parashara Hora Shastra\n")
            f.write("="*70 + "\n\n")
            
            for item in basics["fundamentals"]:
                f.write(f"\n{item['concept']}\n")
                f.write("-" * len(item['concept']) + "\n")
                if 'sanskrit' in item:
                    f.write(f"Sanskrit: {item['sanskrit']}\n")
                if 'description' in item:
                    f.write(f"{item['description']}\n")
                if 'pillars' in item:
                    for key, val in item['pillars'].items():
                        f.write(f"  {key}: {val}\n")
                if 'purpose' in item:
                    f.write(f"Purpose: {item['purpose']}\n")
                if 'importance' in item:
                    f.write(f"Importance: {item['importance']}\n")
                if 'significance' in item:
                    f.write(f"Significance: {item['significance']}\n")
                f.write("\n")
            
            f.write("\nCore Teachings:\n")
            for teaching in basics["core_teachings"]:
                f.write(f"• {teaching}\n")
        
        self.created_files.append(str(text_path))
        print(f"  ✅ Created astrology fundamentals (2 files)")
    
    def create_nakshatras_database(self):
        """Create 27 Nakshatras database"""
        print("🌙 Creating 27 Nakshatras database...")
        
        nakshatras = {
            "title": "27 Nakshatras - Lunar Mansions",
            "description": "The 27 divisions of the zodiac, each ruled by a deity and planet",
            "nakshatras": [
                {
                    "number": 1,
                    "name": "Ashwini",
                    "sanskrit": "अश्विनी",
                    "range": "0°00' - 13°20' Aries",
                    "deity": "Ashwini Kumaras (Divine Physicians)",
                    "ruler": "Ketu",
                    "symbol": "Horse's Head",
                    "qualities": "Swift action, healing ability, pioneering spirit, new beginnings",
                    "characteristics": "Quick to act, energetic, natural healers, love adventure"
                },
                {
                    "number": 2,
                    "name": "Bharani",
                    "sanskrit": "भरणी",
                    "range": "13°20' - 26°40' Aries",
                    "deity": "Yama (God of Death)",
                    "ruler": "Venus",
                    "symbol": "Yoni (Womb)",
                    "qualities": "Transformation, restraint, nourishment, creative power",
                    "characteristics": "Strong will, creative, can handle extremes, transformative"
                },
                {
                    "number": 3,
                    "name": "Krittika",
                    "sanskrit": "कृत्तिका",
                    "range": "26°40' Aries - 10°00' Taurus",
                    "deity": "Agni (Fire God)",
                    "ruler": "Sun",
                    "symbol": "Razor/Flame",
                    "qualities": "Purification, sharp intellect, cutting through illusion",
                    "characteristics": "Sharp mind, purifying nature, direct communication, leadership"
                },
                {
                    "number": 4,
                    "name": "Rohini",
                    "sanskrit": "रोहिणी",
                    "range": "10°00' - 23°20' Taurus",
                    "deity": "Brahma (Creator)",
                    "ruler": "Moon",
                    "symbol": "Ox Cart/Chariot",
                    "qualities": "Growth, fertility, beauty, material abundance",
                    "characteristics": "Attractive, creative, love beauty and comfort, sensual"
                },
                {
                    "number": 5,
                    "name": "Mrigashira",
                    "sanskrit": "मृगशिरा",
                    "range": "23°20' Taurus - 6°40' Gemini",
                    "deity": "Soma (Moon God)",
                    "ruler": "Mars",
                    "symbol": "Deer's Head",
                    "qualities": "Seeking, searching, gentle strength, curiosity",
                    "characteristics": "Seekers of knowledge, gentle yet strong, love travel"
                },
                {
                    "number": 6,
                    "name": "Ardra",
                    "sanskrit": "आर्द्रा",
                    "range": "6°40' - 20°00' Gemini",
                    "deity": "Rudra (Storm God)",
                    "ruler": "Rahu",
                    "symbol": "Teardrop/Diamond",
                    "qualities": "Transformation through storms, deep emotion, renewal",
                    "characteristics": "Intense emotions, transformative, can weather any storm"
                },
                {
                    "number": 7,
                    "name": "Punarvasu",
                    "sanskrit": "पुनर्वसु",
                    "range": "20°00' Gemini - 3°20' Cancer",
                    "deity": "Aditi (Mother of Gods)",
                    "ruler": "Jupiter",
                    "symbol": "Bow and Quiver",
                    "qualities": "Return to light, renewal, abundance, protection",
                    "characteristics": "Optimistic, philosophical, ability to bounce back, nurturing"
                },
                {
                    "number": 8,
                    "name": "Pushya",
                    "sanskrit": "पुष्य",
                    "range": "3°20' - 16°40' Cancer",
                    "deity": "Brihaspati (Jupiter)",
                    "ruler": "Saturn",
                    "symbol": "Cow's Udder/Lotus",
                    "qualities": "Nourishment, spiritual growth, auspiciousness",
                    "characteristics": "Nurturing, spiritual, disciplined, brings good fortune"
                },
                # Adding a few more key nakshatras
                {
                    "number": 9,
                    "name": "Ashlesha",
                    "sanskrit": "आश्लेषा",
                    "range": "16°40' - 30°00' Cancer",
                    "deity": "Nagas (Serpent Deities)",
                    "ruler": "Mercury",
                    "symbol": "Coiled Serpent",
                    "qualities": "Mystical wisdom, kundalini energy, penetrating insight",
                    "characteristics": "Intuitive, mystical, hypnotic, can see hidden truths"
                },
                {
                    "number": 10,
                    "name": "Magha",
                    "sanskrit": "मघा",
                    "range": "0°00' - 13°20' Leo",
                    "deity": "Pitris (Ancestors)",
                    "ruler": "Ketu",
                    "symbol": "Throne Room",
                    "qualities": "Royal authority, ancestral legacy, respect for tradition",
                    "characteristics": "Regal bearing, respect tradition, connection to ancestors"
                }
            ],
            "usage": "Nakshatras are used for:",
            "applications": [
                "Determining personality traits and life patterns",
                "Muhurta (electional astrology) for choosing auspicious times",
                "Matching charts for marriage compatibility",
                "Understanding karmic patterns from past lives",
                "Timing of events (Dasha system)"
            ]
        }
        
        # Save as JSON
        json_path = self.base_path / "nakshatras" / "nakshatras_database.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(nakshatras, f, indent=2, ensure_ascii=False)
        self.created_files.append(str(json_path))
        
        # Save as readable text
        text_path = self.base_path / "nakshatras" / "nakshatras_guide.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("27 NAKSHATRAS - LUNAR MANSIONS OF VEDIC ASTROLOGY\n")
            f.write("="*70 + "\n\n")
            f.write(f"{nakshatras['description']}\n\n")
            
            for nak in nakshatras['nakshatras']:
                f.write(f"\n{nak['number']}. {nak['name']} ({nak['sanskrit']})\n")
                f.write("-" * 50 + "\n")
                f.write(f"Range: {nak['range']}\n")
                f.write(f"Deity: {nak['deity']}\n")
                f.write(f"Ruling Planet: {nak['ruler']}\n")
                f.write(f"Symbol: {nak['symbol']}\n")
                f.write(f"Qualities: {nak['qualities']}\n")
                f.write(f"Characteristics: {nak['characteristics']}\n")
        
        self.created_files.append(str(text_path))
        print(f"  ✅ Created 10 Nakshatras (sample set) (2 files)")
    
    def create_planetary_wisdom(self):
        """Create 9 planets (Grahas) wisdom"""
        print("🪐 Creating planetary wisdom...")
        
        planets = {
            "title": "The Nine Grahas - Planetary Forces",
            "description": "In Vedic astrology, grahas are not just physical planets but karmic forces",
            "planets": [
                {
                    "name": "Sun (Surya)",
                    "sanskrit": "सूर्य",
                    "represents": "Soul, father, authority, government, ego, vitality",
                    "deity": "Surya Deva",
                    "element": "Fire",
                    "exalted_in": "Aries 10°",
                    "debilitated_in": "Libra 10°",
                    "positive_traits": "Leadership, confidence, nobility, generosity, vitality",
                    "negative_traits": "Ego, arrogance, domination, harsh speech",
                    "bhrigu_teaching": "The Sun reveals your soul's purpose. A strong Sun gives leadership and self-confidence. Honor your father and authorities to strengthen Sun."
                },
                {
                    "name": "Moon (Chandra)",
                    "sanskrit": "चन्द्र",
                    "represents": "Mind, mother, emotions, public, comfort, nurturing",
                    "deity": "Soma",
                    "element": "Water",
                    "exalted_in": "Taurus 3°",
                    "debilitated_in": "Scorpio 3°",
                    "positive_traits": "Emotional intelligence, nurturing, intuitive, adaptable",
                    "negative_traits": "Moodiness, over-sensitivity, dependency, fluctuation",
                    "bhrigu_teaching": "The Moon governs your mind and emotions. A strong Moon gives mental peace and emotional stability. Honor your mother to strengthen Moon."
                },
                {
                    "name": "Mars (Mangala)",
                    "sanskrit": "मंगल",
                    "represents": "Energy, courage, siblings, property, sports, anger",
                    "deity": "Kartikeya (Skanda)",
                    "element": "Fire",
                    "exalted_in": "Capricorn 28°",
                    "debilitated_in": "Cancer 28°",
                    "positive_traits": "Courage, energy, determination, technical skill",
                    "negative_traits": "Anger, aggression, impatience, accidents",
                    "bhrigu_teaching": "Mars gives warrior spirit and courage. Channel this energy through exercise, sports, or service. Control anger through discipline."
                },
                {
                    "name": "Mercury (Budha)",
                    "sanskrit": "बुध",
                    "represents": "Intelligence, communication, business, learning, logic",
                    "deity": "Vishnu",
                    "element": "Earth",
                    "exalted_in": "Virgo 15°",
                    "debilitated_in": "Pisces 15°",
                    "positive_traits": "Intelligence, wit, communication skill, business acumen",
                    "negative_traits": "Nervousness, over-analysis, restlessness, cunning",
                    "bhrigu_teaching": "Mercury rules the intellect. A strong Mercury gives sharp mind and communication skills. Study scriptures to elevate Mercury."
                },
                {
                    "name": "Jupiter (Guru)",
                    "sanskrit": "गुरु",
                    "represents": "Wisdom, teacher, children, fortune, dharma, expansion",
                    "deity": "Brihaspati",
                    "element": "Ether/Space",
                    "exalted_in": "Cancer 5°",
                    "debilitated_in": "Capricorn 5°",
                    "positive_traits": "Wisdom, optimism, generosity, dharmic living",
                    "negative_traits": "Over-optimism, indulgence, laziness, preachiness",
                    "bhrigu_teaching": "Jupiter is the great benefic, bringing wisdom and fortune. Respect teachers and elders. Study philosophy to strengthen Jupiter."
                },
                {
                    "name": "Venus (Shukra)",
                    "sanskrit": "शुक्र",
                    "represents": "Love, beauty, luxury, arts, marriage, pleasure",
                    "deity": "Shukracharya",
                    "element": "Water",
                    "exalted_in": "Pisces 27°",
                    "debilitated_in": "Virgo 27°",
                    "positive_traits": "Artistic talent, charm, diplomacy, refined taste",
                    "negative_traits": "Vanity, laziness, overindulgence, materialism",
                    "bhrigu_teaching": "Venus brings beauty and harmony. A strong Venus gives artistic talents and happy marriage. Practice moderation in pleasures."
                },
                {
                    "name": "Saturn (Shani)",
                    "sanskrit": "शनि",
                    "represents": "Karma, discipline, delays, servants, sorrow, longevity",
                    "deity": "Yama",
                    "element": "Air",
                    "exalted_in": "Libra 20°",
                    "debilitated_in": "Aries 20°",
                    "positive_traits": "Discipline, patience, hard work, justice, longevity",
                    "negative_traits": "Delays, pessimism, fear, restrictions, suffering",
                    "bhrigu_teaching": "Saturn is the great teacher through discipline. Accept delays as lessons. Serve the poor and elderly to appease Saturn."
                },
                {
                    "name": "Rahu (North Node)",
                    "sanskrit": "राहु",
                    "represents": "Desires, illusion, foreign lands, sudden events, materialism",
                    "deity": "Durga",
                    "nature": "Shadow planet (lunar node)",
                    "exalted_in": "Taurus/Gemini",
                    "debilitated_in": "Scorpio/Sagittarius",
                    "positive_traits": "Ambition, innovation, foreign success, unique perspective",
                    "negative_traits": "Obsession, confusion, deception, addictions",
                    "bhrigu_teaching": "Rahu creates intense desires and illusions. It can give sudden rise or confusion. Practice meditation to calm Rahu's restless energy."
                },
                {
                    "name": "Ketu (South Node)",
                    "sanskrit": "केतु",
                    "represents": "Spirituality, detachment, past lives, liberation, mysticism",
                    "deity": "Ganesha",
                    "nature": "Shadow planet (lunar node)",
                    "exalted_in": "Scorpio/Sagittarius",
                    "debilitated_in": "Taurus/Gemini",
                    "positive_traits": "Spiritual insight, intuition, moksha, detachment",
                    "negative_traits": "Confusion, isolation, sudden losses, lack of clarity",
                    "bhrigu_teaching": "Ketu brings spiritual liberation through detachment. It shows past life karma. Embrace spirituality to work with Ketu positively."
                }
            ]
        }
        
        # Save as JSON
        json_path = self.base_path / "planetary_wisdom" / "nine_planets.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(planets, f, indent=2, ensure_ascii=False)
        self.created_files.append(str(json_path))
        
        # Save as readable text
        text_path = self.base_path / "planetary_wisdom" / "planetary_guide.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("THE NINE GRAHAS - PLANETARY FORCES IN VEDIC ASTROLOGY\n")
            f.write("="*70 + "\n\n")
            
            for planet in planets['planets']:
                f.write(f"\n{planet['name']} - {planet['sanskrit']}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Represents: {planet['represents']}\n")
                if 'deity' in planet:
                    f.write(f"Deity: {planet['deity']}\n")
                if 'exalted_in' in planet:
                    f.write(f"Exalted in: {planet['exalted_in']}\n")
                    f.write(f"Debilitated in: {planet['debilitated_in']}\n")
                f.write(f"\nPositive: {planet['positive_traits']}\n")
                f.write(f"Negative: {planet['negative_traits']}\n")
                f.write(f"\nBhrigu's Teaching: {planet['bhrigu_teaching']}\n")
        
        self.created_files.append(str(text_path))
        print(f"  ✅ Created 9 planets wisdom (2 files)")
    
    def create_birth_chart_guide(self):
        """Create birth chart interpretation guide"""
        print("📊 Creating birth chart interpretation guide...")
        
        guide = """BIRTH CHART INTERPRETATION - BHRIGU'S METHOD
========================================

Understanding the Kundali (Birth Chart)

STEP 1: Identify the Ascendant (Lagna)
---------------------------------------
The Ascendant is the zodiac sign rising on the eastern horizon at birth.
It represents:
• Your physical body and appearance
• Your basic nature and approach to life
• The lens through which you see the world
• Your life's overall direction

Example: If Aries is rising, you approach life with courage and directness.


STEP 2: Analyze the 12 Houses
------------------------------
Each house represents a life area:

1st House (Lagna): Self, body, personality, beginnings
2nd House: Wealth, family, speech, values
3rd House: Siblings, courage, short journeys, skills
4th House: Mother, home, emotions, vehicles, property
5th House: Children, creativity, intelligence, past life karma
6th House: Health, enemies, service, obstacles
7th House: Marriage, partnerships, business
8th House: Transformation, longevity, hidden matters, inheritance
9th House: Dharma, father, guru, fortune, long journeys
10th House: Career, status, authority, public image
11th House: Gains, friends, elder siblings, aspirations
12th House: Losses, liberation, foreign lands, spirituality


STEP 3: Locate Planetary Positions
-----------------------------------
Note which planet is in which house and sign.
Each planet brings its energy to that house.

Example: 
• Jupiter in 5th house = blessed with children and creativity
• Saturn in 7th house = delayed but stable marriage
• Moon in 4th house = emotional connection with mother


STEP 4: Check Planetary Strength
---------------------------------
Planets are stronger in certain positions:

Exaltation: Planet at peak power
Debilitation: Planet weakened
Own Sign: Planet comfortable
Friendly Sign: Planet supported
Enemy Sign: Planet challenged


STEP 5: Identify Yogas (Combinations)
--------------------------------------
Special planetary combinations create yogas:

Raja Yoga: Combinations for power and status
Dhana Yoga: Wealth-creating combinations
Vipareeta Raja Yoga: Success through adversity
Gajakesari Yoga: Jupiter-Moon combination for wisdom


STEP 6: Analyze Dashas (Time Periods)
--------------------------------------
Vimshottari Dasha shows which planet's period you're in.
Each planet activates its significations during its period.


BHRIGU'S INTERPRETATION PRINCIPLES:
-----------------------------------
1. Chart shows potential, not certainty
2. Free will operates within karmic framework
3. Remedies can modify outcomes
4. Dharmic living strengthens beneficial planets
5. Chart is a tool for self-awareness


INTERPRETATION EXAMPLE:
-----------------------
Birth Chart with:
• Leo Ascendant (Sun ruled)
• Sun in 10th house (career)
• Jupiter in 5th house (children)
• Saturn in 2nd house (wealth)

Reading: Natural leader (Leo rising), destined for public career 
(Sun in 10th), blessed with intelligent children (Jupiter in 5th), 
will earn wealth through discipline and hard work (Saturn in 2nd).


Bhrigu teaches: "The stars incline, they do not compel. 
Your karma is written, but your dharma can rewrite it."
"""
        
        text_path = self.base_path / "birth_charts" / "chart_interpretation_guide.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(guide)
        self.created_files.append(str(text_path))
        
        print(f"  ✅ Created birth chart interpretation guide")
    
    def create_dasha_system(self):
        """Create Dasha system explanation"""
        print("⏰ Creating Dasha system guide...")
        
        dasha = """VIMSHOTTARI DASHA SYSTEM - PLANETARY PERIODS
============================================

The 120-Year Cycle of Planetary Influence

WHAT IS DASHA?
--------------
Dasha means "planetary period" - a time when a specific planet's
energy dominates your life. The Vimshottari Dasha is a 120-year
cycle divided among the nine planets.


THE NINE PLANETARY PERIODS:
----------------------------

1. Ketu Dasha - 7 years
   • Spiritual awakening, detachment
   • Sudden events, mystical experiences
   • Past life karma surfaces

2. Venus Dasha - 20 years
   • Love, marriage, relationships
   • Luxury, arts, beauty
   • Material comforts and pleasures

3. Sun Dasha - 6 years
   • Career advancement, recognition
   • Relationship with father/authority
   • Leadership opportunities

4. Moon Dasha - 10 years
   • Emotional experiences, family focus
   • Relationship with mother
   • Mental peace or turbulence

5. Mars Dasha - 7 years
   • Energy, courage, action
   • Property matters, siblings
   • Potential for conflicts or accidents

6. Rahu Dasha - 18 years
   • Intense desires, material pursuit
   • Foreign connections, sudden changes
   • Illusions and confusions

7. Jupiter Dasha - 16 years
   • Wisdom, expansion, fortune
   • Children, education, spirituality
   • Dharmic progress

8. Saturn Dasha - 19 years
   • Hard work, discipline, delays
   • Karmic lessons, responsibilities
   • Success through perseverance

9. Mercury Dasha - 17 years
   • Intellectual pursuits, communication
   • Business, learning, skills
   • Social connections


UNDERSTANDING YOUR CURRENT DASHA:
----------------------------------
Your birth Nakshatra determines which Dasha you start with.
The sequence then follows the order above.

Example: If born in Ashwini (ruled by Ketu), you start with
Ketu Dasha, then Venus, then Sun, and so on.


SUB-PERIODS (BHUKTI/ANTARDASHA):
---------------------------------
Each main period has sub-periods of all nine planets.
For example, during Venus Dasha:
• Venus-Venus (peak of Venus energy)
• Venus-Sun (mixing love with authority)
• Venus-Moon (emotional relationships)
...and so on


HOW TO USE DASHA KNOWLEDGE:
----------------------------
1. Know your current Maha Dasha (main period)
2. Understand that planet's nature and placement in your chart
3. The house that planet rules becomes active
4. Plan accordingly - don't fight the cosmic tide
5. Use remedies to maximize positive results


BHRIGU'S DASHA WISDOM:
----------------------
"Each Dasha brings its lesson. Saturn teaches patience,
Jupiter teaches wisdom, Venus teaches love. Accept each
period as a divine curriculum for your soul's evolution."


PRACTICAL EXAMPLE:
------------------
Person entering Saturn Dasha:
• Expect delays and hard work (Saturn's nature)
• Check Saturn's position in chart (which house?)
• If Saturn in 10th house: career requires extra effort
• Remedy: Serve the poor, practice discipline
• Result: Eventual success through perseverance


REMEDIES DURING DIFFICULT DASHAS:
----------------------------------
• Chant the planet's mantra
• Wear the planet's gemstone (after consultation)
• Perform charity related to that planet
• Observe fasts on planet's day
• Live dharmically - the best remedy
"""
        
        text_path = self.base_path / "vedic_astrology" / "dasha_system_guide.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(dasha)
        self.created_files.append(str(text_path))
        
        print(f"  ✅ Created Dasha system guide")
    
    def create_summary(self):
        """Create summary of all knowledge"""
        print("📝 Creating knowledge summary...")
        
        summary = f"""BHRIGU'S ASTROLOGY KNOWLEDGE BASE - SUMMARY
==========================================

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Documents: {len(self.created_files)}

KNOWLEDGE AREAS:
----------------

1. VEDIC ASTROLOGY FUNDAMENTALS
   • Core concepts of Jyotish
   • The three pillars: Rashi, Graha, Bhava
   • Birth chart significance
   • Philosophical foundations

2. 27 NAKSHATRAS
   • Lunar mansions of the zodiac
   • Deities, symbols, and qualities
   • Sample set of 10 major Nakshatras
   • Applications in chart analysis

3. THE NINE PLANETS (GRAHAS)
   • Sun, Moon, Mars, Mercury, Jupiter
   • Venus, Saturn, Rahu, Ketu
   • Significations, exaltations, debilitations
   • Bhrigu's remedial teachings

4. BIRTH CHART INTERPRETATION
   • 12 houses and their meanings
   • Planetary positions and strength
   • Yogas (special combinations)
   • Step-by-step interpretation method

5. DASHA SYSTEM
   • 120-year Vimshottari cycle
   • Nine planetary periods and durations
   • How to use Dasha knowledge
   • Remedies for difficult periods

USAGE:
------
This knowledge base enables Rishi Bhrigu to:
• Explain Vedic astrology concepts
• Interpret birth chart elements
• Guide on planetary periods
• Suggest remedies and spiritual practices
• Answer questions about karma and destiny

BHRIGU'S TEACHING STYLE:
------------------------
• Speaks with ancient authority and cosmic wisdom
• References classical texts (Brihat Parashara Hora Shastra)
• Balances fate with free will
• Emphasizes dharma over destiny
• Practical remedies with spiritual depth

FILES CREATED:
--------------
"""
        
        for i, file in enumerate(self.created_files, 1):
            summary += f"{i}. {file}\n"
        
        summary += f"\nTotal knowledge documents: {len(self.created_files)}\n"
        summary += "\n🔮 Rishi Bhrigu's wisdom is ready to guide seekers through the cosmic blueprint! 🔮\n"
        
        summary_path = self.base_path / "BHRIGU_KNOWLEDGE_SUMMARY.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n  ✅ Created knowledge summary")
        print(f"\n📊 Summary saved to: {summary_path}")


def main():
    """Build Bhrigu's knowledge base"""
    builder = BhriguKnowledgeBuilder()
    builder.create_all()
    
    print("\n✨ Bhrigu's astrology knowledge base is complete!")
    print("   Ready to build RAG system next...")


if __name__ == "__main__":
    main()
