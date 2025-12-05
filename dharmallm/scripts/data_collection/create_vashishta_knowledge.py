#!/usr/bin/env python3
"""
Vashishta Knowledge Base Creator
=================================

Builds Rishi Vashishta's dharma/ethics knowledge base with:
1. Dharma fundamentals and principles
2. Ethical dilemmas and solutions
3. Four life stages (Ashramas)
4. Four goals of life (Purusharthas)
5. Modern dharmic living guidance

Vashishta is the ancient sage of dharma, ethics, and righteous living.
"""

import json
from pathlib import Path
from datetime import datetime


class VashishtaKnowledgeBuilder:
    """Build Vashishta's dharma knowledge base"""
    
    def __init__(self):
        self.base_path = Path("data/rishi_knowledge/vashishta")
        self.created_files = []
    
    def create_all(self):
        """Create complete Vashishta knowledge base"""
        print("\n" + "="*70)
        print("📿 Building Vashishta's Dharma Knowledge Base")
        print("="*70 + "\n")
        
        self.create_dharma_fundamentals()
        self.create_purusharthas()
        self.create_ashramas()
        self.create_ethical_dilemmas()
        self.create_modern_dharma_guide()
        self.create_summary()
        
        print("\n" + "="*70)
        print(f"✅ Created {len(self.created_files)} documents for Vashishta")
        print("="*70 + "\n")
    
    def create_dharma_fundamentals(self):
        """Create dharma fundamentals"""
        print("📖 Creating Dharma fundamentals...")
        
        fundamentals = {
            "title": "Fundamentals of Sanatana Dharma",
            "sage": "Maharishi Vashishta",
            "source": "Dharma Shastras and Vashishta's Teachings",
            "core_concepts": [
                {
                    "concept": "What is Dharma?",
                    "sanskrit": "धर्म",
                    "definition": "Dharma means 'that which upholds'. It is righteousness, duty, moral law, and the cosmic order that sustains the universe.",
                    "explanation": "Dharma is not just religion - it is the eternal principles that govern right living, ethical conduct, and harmonious existence."
                },
                {
                    "concept": "The Ten Universal Virtues (Sadharana Dharma)",
                    "virtues": {
                        "1_Dhriti": "Steadfastness, patience, fortitude",
                        "2_Kshama": "Forgiveness, tolerance",
                        "3_Dama": "Self-control, restraint of senses",
                        "4_Asteya": "Non-stealing, honesty",
                        "5_Shaucha": "Purity of body and mind",
                        "6_Indriya-nigraha": "Mastery over senses",
                        "7_Dhi": "Intellect, wisdom, discrimination",
                        "8_Vidya": "Knowledge, learning",
                        "9_Satya": "Truthfulness",
                        "10_Akrodha": "Absence of anger, calmness"
                    },
                    "teaching": "These ten virtues are universal dharma - applicable to all humans regardless of birth, status, or circumstance."
                },
                {
                    "concept": "Svadharma vs Samanya Dharma",
                    "svadharma": "One's personal duty based on nature, stage of life, and circumstances",
                    "samanya_dharma": "Universal duties applicable to all (non-violence, truth, etc.)",
                    "principle": "When in conflict, svadharma may take precedence, but never at the cost of core ethical principles."
                },
                {
                    "concept": "Karma and Dharma",
                    "relationship": "Dharma guides right action; karma is the law of cause and effect",
                    "teaching": "Live according to dharma, and your karma becomes a vehicle for spiritual evolution rather than bondage."
                }
            ],
            "key_principles": [
                "Ahimsa (Non-violence) - in thought, word, and deed",
                "Satya (Truth) - but truth must be beneficial, not harmful",
                "Asteya (Non-stealing) - even in subtle forms like time, ideas",
                "Brahmacharya (Self-control) - mastery over desires and senses",
                "Aparigraha (Non-possessiveness) - freedom from greed",
                "Dharma is context-dependent - what is right depends on time, place, circumstance",
                "The greatest dharma is compassion toward all beings",
                "When in doubt, ask: 'What would a wise, compassionate person do?'"
            ]
        }
        
        # Save JSON
        json_path = self.base_path / "dharma_shastras" / "dharma_fundamentals.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(fundamentals, f, indent=2, ensure_ascii=False)
        self.created_files.append(str(json_path))
        
        # Save readable text
        text_path = self.base_path / "dharma_shastras" / "dharma_fundamentals.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("FUNDAMENTALS OF SANATANA DHARMA\n")
            f.write("Teachings of Maharishi Vashishta\n")
            f.write("="*70 + "\n\n")
            
            for concept in fundamentals['core_concepts']:
                f.write(f"\n{concept['concept']}\n")
                f.write("-" * len(concept['concept']) + "\n")
                if 'sanskrit' in concept:
                    f.write(f"Sanskrit: {concept['sanskrit']}\n\n")
                if 'definition' in concept:
                    f.write(f"{concept['definition']}\n\n")
                if 'explanation' in concept:
                    f.write(f"{concept['explanation']}\n\n")
                if 'virtues' in concept:
                    f.write("The Ten Universal Virtues:\n")
                    for key, val in concept['virtues'].items():
                        name = key.split('_')[1]
                        f.write(f"  {name}: {val}\n")
                    f.write(f"\n{concept['teaching']}\n\n")
                if 'principle' in concept:
                    f.write(f"Principle: {concept['principle']}\n\n")
            
            f.write("\nKey Principles of Dharmic Living:\n")
            f.write("-" * 40 + "\n")
            for principle in fundamentals['key_principles']:
                f.write(f"• {principle}\n")
        
        self.created_files.append(str(text_path))
        print("  ✅ Created dharma fundamentals")
    
    def create_purusharthas(self):
        """Create four goals of life"""
        print("🎯 Creating Purusharthas (Four Goals)...")
        
        purusharthas = {
            "title": "The Four Purusharthas - Goals of Human Life",
            "description": "The four legitimate aims that give meaning to human existence",
            "goals": [
                {
                    "name": "Dharma (धर्म)",
                    "meaning": "Righteousness, Duty, Ethics",
                    "description": "Living according to moral law and cosmic order",
                    "importance": "Foundation for all other goals. Without dharma, other pursuits lead to suffering.",
                    "how_to_pursue": [
                        "Study scriptures and learn ethical principles",
                        "Practice the ten universal virtues",
                        "Fulfill your duties according to your stage of life",
                        "Act with integrity in all situations",
                        "Serve others selflessly"
                    ],
                    "vashishta_teaching": "Dharma is not a burden but the path to true freedom. When you live righteously, the universe supports you."
                },
                {
                    "name": "Artha (अर्थ)",
                    "meaning": "Wealth, Material Prosperity, Security",
                    "description": "Earning livelihood and creating material wellbeing through righteous means",
                    "importance": "Necessary for survival and supporting family. But must be earned ethically.",
                    "how_to_pursue": [
                        "Develop skills and work diligently",
                        "Earn through honest means only",
                        "Save and invest wisely for future security",
                        "Share wealth through charity",
                        "Never sacrifice dharma for wealth"
                    ],
                    "vashishta_teaching": "Wealth earned through dharma brings peace; wealth earned through adharma brings anxiety. The truly wealthy person is content with enough."
                },
                {
                    "name": "Kama (काम)",
                    "meaning": "Desire, Pleasure, Enjoyment",
                    "description": "Legitimate enjoyment of life's pleasures within dharmic boundaries",
                    "importance": "Life is meant to be enjoyed. Denying natural desires creates suppression. But enjoyment must be balanced.",
                    "how_to_pursue": [
                        "Enjoy sensory pleasures without attachment",
                        "Practice moderation in all things",
                        "Respect boundaries of dharma in relationships",
                        "Appreciate beauty in nature, art, music",
                        "Cultivate refined tastes and aesthetic sense"
                    ],
                    "vashishta_teaching": "Pleasure pursued with awareness becomes a spiritual practice. Pleasure pursued with greed becomes bondage. The wise enjoy without being enslaved."
                },
                {
                    "name": "Moksha (मोक्ष)",
                    "meaning": "Liberation, Self-Realization, Freedom",
                    "description": "Ultimate goal - freedom from the cycle of birth and death, union with the Divine",
                    "importance": "The highest aim. All other goals ultimately serve this purpose.",
                    "how_to_pursue": [
                        "Practice meditation and self-inquiry",
                        "Study with a genuine guru",
                        "Renounce attachment (not things, but clinging)",
                        "Realize your true nature beyond body-mind",
                        "Live in awareness of the Divine in all"
                    ],
                    "vashishta_teaching": "Moksha is not somewhere to reach - it is what you already are. Remove ignorance, and liberation shines forth naturally."
                }
            ],
            "balance_teaching": "The Four Purusharthas must be balanced. Pursue artha and kama within dharma, and let all three lead you toward moksha. This is the art of dharmic living."
        }
        
        # Save JSON
        json_path = self.base_path / "dharma_shastras" / "purusharthas.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(purusharthas, f, indent=2, ensure_ascii=False)
        self.created_files.append(str(json_path))
        
        # Save text
        text_path = self.base_path / "dharma_shastras" / "purusharthas.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("THE FOUR PURUSHARTHAS - GOALS OF HUMAN LIFE\n")
            f.write("="*70 + "\n\n")
            
            for goal in purusharthas['goals']:
                f.write(f"\n{goal['name']}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Meaning: {goal['meaning']}\n")
                f.write(f"{goal['description']}\n\n")
                f.write(f"Importance: {goal['importance']}\n\n")
                f.write("How to Pursue:\n")
                for step in goal['how_to_pursue']:
                    f.write(f"  • {step}\n")
                f.write(f"\nVashishta's Teaching: {goal['vashishta_teaching']}\n")
        
        self.created_files.append(str(text_path))
        print("  ✅ Created Purusharthas guide")
    
    def create_ashramas(self):
        """Create four life stages"""
        print("🌱 Creating Ashramas (Four Life Stages)...")
        
        ashramas_text = """THE FOUR ASHRAMAS - STAGES OF LIFE
====================================

Ancient Wisdom for Modern Living

OVERVIEW:
---------
The Ashrama system divides human life into four stages, each with its own
dharma (duties) and focus. This system provides a framework for balanced
living across the entire lifespan.


STAGE 1: BRAHMACHARYA (Student Life)
-------------------------------------
Age: Birth to ~25 years
Focus: Learning, Character Building, Self-Discipline

Primary Duties:
• Study scriptures, arts, sciences under a teacher
• Practice self-control and celibacy
• Develop good character and habits
• Build strong foundation in dharma
• Serve the teacher (Guru Seva)

Goal: Acquire knowledge, skills, and values needed for life

Modern Application:
• School, college, vocational training
• Learning life skills from parents and mentors
• Developing discipline and focus
• Avoiding premature indulgence in pleasures
• Building career foundation

Vashishta's Teaching:
"The brahmacharya stage is like planting a tree. Strong roots now mean
a strong tree later. Invest in learning - it is wealth that cannot be stolen."


STAGE 2: GRIHASTHA (Householder Life)
--------------------------------------
Age: ~25 to ~50 years
Focus: Family, Career, Contribution to Society

Primary Duties:
• Marry and raise children with dharma
• Earn livelihood through righteous means
• Support family, parents, relatives
• Contribute to society through work and charity
• Perform religious rituals and duties
• Host guests with hospitality

Goal: Experience and fulfill material and social responsibilities

Modern Application:
• Career and professional development
• Marriage and family life
• Financial planning and wealth creation
• Social service and community involvement
• Balancing work, family, and personal growth

Vashishta's Teaching:
"The householder is the pillar of society. When you support your family
with love and society with service, you perform the greatest yajna.
But remember - you are not just earning money; you are earning merit."


STAGE 3: VANAPRASTHA (Forest Dweller / Retirement)
---------------------------------------------------
Age: ~50 to ~75 years
Focus: Gradual Withdrawal, Spiritual Practice, Mentoring

Primary Duties:
• Gradually reduce material involvement
• Spend more time in spiritual practices
• Guide younger generations with wisdom
• Practice detachment while still engaged
• Pilgrimage and study of scriptures
• Prepare for final stage

Goal: Transition from material to spiritual focus

Modern Application:
• Retirement planning and gradual withdrawal from career
• More time for spiritual practices and hobbies
• Mentoring younger people in family and society
• Volunteering and wisdom-sharing
• Simplifying lifestyle
• Travel and spiritual exploration

Vashishta's Teaching:
"In vanaprastha, you are like a tree that has given fruits. Now your
shade provides comfort to others. Share your wisdom freely, for 
knowledge grows when shared. But also prepare - the final journey awaits."


STAGE 4: SANNYASA (Renunciation)
---------------------------------
Age: ~75+ years or when ready
Focus: Complete Detachment, Moksha (Liberation)

Primary Duties:
• Renounce all material attachments
• Live on minimal necessities
• Meditate constantly on the Self
• Share wisdom with sincere seekers
• Prepare for death consciously
• Realize ultimate truth

Goal: Attain moksha - liberation from birth-death cycle

Modern Application:
• Complete retirement from worldly affairs
• Deep spiritual practice and meditation
• Letting go of possessions gradually
• Accepting mortality gracefully
• Living simply with few needs
• Serving as spiritual elder

Vashishta's Teaching:
"Sannyasa is not running away from life - it is completing life's journey
consciously. You have played all roles; now discover who the actor truly is.
The body will fall; know thyself before it does."


FLEXIBILITY IN MODERN TIMES:
-----------------------------
• The age ranges are flexible - follow your life circumstances
• Some may skip stages or return to earlier stages
• The key is the *attitude* appropriate to each stage
• Not everyone must progress through all four stages
• What matters is fulfilling your dharma in your current stage


VASHISHTA'S OVERALL TEACHING:
------------------------------
"Life is a university with four semesters. Each has its lessons.
Don't try to skip grades, but don't get held back either.
When you honor each stage, you graduate to liberation naturally.

The brahmacharya learns, the grihastha applies learning,
the vanaprastha reflects on learning, the sannyasi transcends learning.
This is the rhythm of a dharmic life."


KEY INSIGHT:
------------
The Ashrama system is not rigid dogma but practical wisdom.
It acknowledges that human needs and capacities change with age.
By aligning your life with these natural stages, you flow with dharma
rather than fighting against your nature.
"""
        
        text_path = self.base_path / "life_stages" / "four_ashramas.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(ashramas_text)
        self.created_files.append(str(text_path))
        
        print("  ✅ Created Ashramas guide")
    
    def create_ethical_dilemmas(self):
        """Create ethical dilemmas and solutions"""
        print("⚖️  Creating ethical dilemmas...")
        
        dilemmas = {
            "title": "Ethical Dilemmas and Vashishta's Guidance",
            "introduction": "Dharma is not always black and white. Here are common dilemmas and how to navigate them.",
            "dilemmas": [
                {
                    "dilemma": "Truth vs Compassion",
                    "scenario": "You know a painful truth that will hurt someone. Should you speak it?",
                    "vashishta_guidance": "Truth must be beneficial (हितम्). If truth causes unnecessary harm without benefit, silence or skillful speech is better. Ask: Will this truth help them grow, or just cause pain? If it helps, speak with compassion. If not, remain silent or find a gentler way.",
                    "principle": "Satya (truth) must be combined with Ahimsa (non-harm)"
                },
                {
                    "dilemma": "Personal vs Family Duty",
                    "scenario": "Your personal goals conflict with family expectations. Which takes priority?",
                    "vashishta_guidance": "First, genuinely listen to family concerns - they may see what you don't. Then, if your path is dharmic and benefits not just you but ultimately others, follow it with respect. Explain your reasoning, seek compromise where possible, but don't sacrifice your svadharma (personal duty) entirely. A forced life creates resentment.",
                    "principle": "Balance Kula-dharma (family duty) with Svadharma (personal duty)"
                },
                {
                    "dilemma": "Wealth vs Ethics",
                    "scenario": "You can earn more money through slightly unethical means. Everyone does it. Should you?",
                    "vashishta_guidance": "Absolutely not. Wealth earned through adharma brings anxiety, guilt, and karmic debt. You may gain money but lose peace. The truly wise person earns less with integrity than more with compromise. Trust that dharma supports those who follow it. Your character is your true wealth.",
                    "principle": "Never sacrifice Dharma for Artha"
                },
                {
                    "dilemma": "Forgiveness vs Justice",
                    "scenario": "Someone wronged you deeply. Should you forgive or seek justice?",
                    "vashishta_guidance": "Forgiveness is for your own peace - holding anger hurts you most. But forgiveness doesn't mean enabling harmful behavior. You can forgive internally while still maintaining boundaries or seeking appropriate consequences. Justice protects others from harm. True forgiveness is releasing the poison of hatred, not accepting abuse.",
                    "principle": "Kshama (forgiveness) with Viveka (discrimination)"
                },
                {
                    "dilemma": "Ambition vs Contentment",
                    "scenario": "Should I strive for more, or be content with what I have?",
                    "vashishta_guidance": "Strive dharmic ally, but remain content internally. Work with full effort but without attachment to results. Ambition rooted in dharma and service is noble. Ambition rooted in greed and comparison is suffering. The key is: Are you pursuing this for growth and contribution, or to fill an inner void? Fill the void with spiritual practice, then pursue outer goals from wholeness.",
                    "principle": "Karma Yoga - Action without attachment"
                },
                {
                    "dilemma": "Self-Care vs Selflessness",
                    "scenario": "Others need me constantly. When is self-care selfish vs necessary?",
                    "vashishta_guidance": "You cannot pour from an empty cup. Taking care of your health, peace, and growth is not selfish - it's responsible. Otherwise you burn out and help no one. Set loving boundaries. Serve from overflow, not depletion. Even airlines tell you to put on your own oxygen mask first. Self-care IS dharma when done to sustain your ability to serve.",
                    "principle": "Sustainable compassion requires self-compassion"
                },
                {
                    "dilemma": "Tradition vs Progress",
                    "scenario": "Old ways are outdated, but elders insist. Should I follow or rebel?",
                    "vashishta_guidance": "Honor the wisdom in tradition while being open to beneficial change. Ask: What is the principle behind this tradition? Can that principle be honored in a new form? Respect elders even when disagreeing. Don't throw out ancient wisdom because it's old, but don't cling to harmful practices just because they're traditional. The eternal principles of dharma remain; their expressions evolve.",
                    "principle": "Sanatana (eternal) vs Kalika (time-bound)"
                }
            ]
        }
        
        # Save JSON
        json_path = self.base_path / "ethical_teachings" / "ethical_dilemmas.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dilemmas, f, indent=2, ensure_ascii=False)
        self.created_files.append(str(json_path))
        
        # Save text
        text_path = self.base_path / "ethical_teachings" / "ethical_dilemmas.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("ETHICAL DILEMMAS AND VASHISHTA'S GUIDANCE\n")
            f.write("="*70 + "\n\n")
            f.write(f"{dilemmas['introduction']}\n\n")
            
            for d in dilemmas['dilemmas']:
                f.write(f"\nDILEMMA: {d['dilemma']}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Scenario: {d['scenario']}\n\n")
                f.write(f"Vashishta's Guidance:\n{d['vashishta_guidance']}\n\n")
                f.write(f"Principle: {d['principle']}\n")
        
        self.created_files.append(str(text_path))
        print("  ✅ Created ethical dilemmas guide")
    
    def create_modern_dharma_guide(self):
        """Create modern dharma living guide"""
        print("🌍 Creating modern dharma guide...")
        
        modern_guide = """DHARMIC LIVING IN THE MODERN WORLD
====================================

Vashishta's Guidance for Contemporary Challenges


WORK AND CAREER DHARMA:
-----------------------

1. Choose Right Livelihood
   • Avoid work that harms others (weapons, intoxicants, exploitation)
   • Seek work that serves a genuine need
   • Use your talents for contribution, not just profit
   • Remember: Your work is your offering to the world

2. Work with Integrity
   • Never compromise ethics for advancement
   • Give honest work for honest pay
   • Treat colleagues with respect
   • Credit others' work; don't steal ideas
   
3. Balance Work and Life
   • Work is important, but not your entire identity
   • Make time for family, health, spirituality
   • Success without peace is failure
   • Know when to stop and rest


RELATIONSHIP DHARMA:
--------------------

1. In Marriage/Partnership
   • Treat partner as spiritual companion
   • Practice patience and forgiveness daily
   • Communicate with honesty and kindness
   • Share responsibilities fairly
   • Grow together spiritually

2. As Parents
   • Raise children with values, not just wealth
   • Teach by example more than words
   • Give roots (values) and wings (freedom)
   • Love unconditionally but discipline when needed
   • Prepare them for life, not just exams

3. With Extended Family
   • Honor parents and elders
   • Support siblings and relatives
   • Maintain boundaries while staying connected
   • Don't let family guilt override your dharma


TECHNOLOGY AND DHARMA:
----------------------

1. Social Media
   • Use mindfully, not addictively
   • Don't compare your life to others' highlights
   • Spread positivity, not negativity
   • Protect your mental peace - unfollow toxicity
   • Real life > Virtual life

2. Digital Ethics
   • Respect privacy and data
   • Don't spread misinformation
   • Be as ethical online as offline
   • Cyber-bullying is still violence
   • Use technology as tool, not escape

3. Screen Time
   • Set boundaries - phones down during meals
   • No screens before bed
   • Spend time in nature regularly
   • Face-to-face connection irreplaceable


MONEY AND DHARMA:
-----------------

1. Earning
   • Earn through skill, not shortcuts
   • Pay your taxes - it's dharma
   • Don't hoard or waste
   • Enough is a sacred word

2. Spending
   • Live below your means
   • Buy what you need, not all you want
   • Quality over quantity
   • Support ethical businesses

3. Giving
   • Give at least 10% to charity
   • Help those less fortunate
   • Give time, not just money
   • Anonymous giving is highest


ENVIRONMENTAL DHARMA:
---------------------

1. Daily Practices
   • Reduce, Reuse, Recycle
   • Save water and electricity
   • Use public transport when possible
   • Buy local and seasonal
   • Minimize plastic usage

2. Food Choices
   • Eat vegetarian or reduce meat consumption
   • Don't waste food - it's sacred
   • Compost organic waste
   • Support sustainable farming
   • Say gratitude before meals

3. Bigger Picture
   • Vote for environmentally conscious leaders
   • Support green initiatives
   • Plant trees
   • Teach children environmental values
   • Remember: Earth is not ours to exploit


CONFLICT RESOLUTION:
--------------------

1. When Upset
   • Pause before reacting
   • Take three deep breaths
   • Ask: "Will this matter in 5 years?"
   • Respond, don't react

2. In Arguments
   • Listen to understand, not to win
   • Attack the problem, not the person
   • Find common ground
   • Be willing to admit mistakes
   • Know when to agree to disagree

3. Long-term Grudges
   • Forgive for your own peace
   • You don't have to forget, but don't poison yourself with hatred
   • Boundaries are healthy
   • Some relationships need distance


VASHISHTA'S DAILY DHARMA CHECKLIST:
------------------------------------

Morning:
□ Wake early, express gratitude
□ Meditate or pray
□ Plan day with intention
□ Eat healthy breakfast

Throughout Day:
□ Work with integrity
□ Speak truth with compassion
□ Help at least one person
□ Avoid gossip and criticism
□ Practice patience in traffic/queues
□ Eat lunch mindfully

Evening:
□ Spend quality time with family
□ Reflect on day's actions
□ Read something uplifting
□ Express gratitude for three things
□ Early dinner

Night:
□ No screens 1 hour before bed
□ Light reading or meditation
□ Forgive yourself and others
□ Sleep by 10 PM


FINAL TEACHING:
---------------
"Dharma in modern times is not different from ancient dharma -
the principles remain eternal. What changes is the context.

Drive your car with consideration for others - that's ahimsa.
Pay your taxes honestly - that's satya and asteya.
Use internet mindfully - that's self-control.
Recycle and save water - that's protecting dharma.

You don't need to go to a cave to live dharmically.
Your home is your ashram, your work is your karma yoga,
your family is your first congregation.

Live simply, think deeply, give generously, love unconditionally.
This is dharma in the 21st century."

- Maharishi Vashishta
"""
        
        text_path = self.base_path / "modern_dharma" / "modern_dharma_guide.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(modern_guide)
        self.created_files.append(str(text_path))
        
        print("  ✅ Created modern dharma guide")
    
    def create_summary(self):
        """Create knowledge summary"""
        print("📝 Creating knowledge summary...")
        
        summary = f"""VASHISHTA'S DHARMA KNOWLEDGE BASE - SUMMARY
===========================================

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Documents: {len(self.created_files)}

KNOWLEDGE AREAS:
----------------

1. DHARMA FUNDAMENTALS
   • What is dharma - core definition
   • Ten universal virtues (Sadharana Dharma)
   • Svadharma vs Samanya dharma
   • Karma and dharma relationship
   • Key ethical principles

2. FOUR PURUSHARTHAS (Life Goals)
   • Dharma - Righteousness and duty
   • Artha - Wealth and prosperity
   • Kama - Legitimate pleasures
   • Moksha - Ultimate liberation
   • Balance and integration

3. FOUR ASHRAMAS (Life Stages)
   • Brahmacharya - Student life (learning)
   • Grihastha - Householder (contributing)
   • Vanaprastha - Retirement (preparing)
   • Sannyasa - Renunciation (liberation)
   • Modern applications for each stage

4. ETHICAL DILEMMAS
   • Truth vs Compassion
   • Personal vs Family duty
   • Wealth vs Ethics
   • Forgiveness vs Justice
   • Ambition vs Contentment
   • Self-care vs Selflessness
   • Tradition vs Progress

5. MODERN DHARMIC LIVING
   • Work and career ethics
   • Relationship dharma
   • Technology and social media
   • Money management
   • Environmental responsibility
   • Conflict resolution
   • Daily dharma checklist

USAGE:
------
This knowledge base enables Rishi Vashishta to:
• Explain fundamental dharmic principles
• Guide on ethical dilemmas
• Advise on life stages and goals
• Apply ancient wisdom to modern situations
• Provide practical daily guidance

VASHISHTA'S TEACHING STYLE:
---------------------------
• Speaks with gentle authority and wisdom
• Balances ancient principles with modern context
• Practical and relatable examples
• Compassionate yet firm on ethics
• Emphasis on sustainable, balanced living

FILES CREATED:
--------------
"""
        
        for i, file in enumerate(self.created_files, 1):
            summary += f"{i}. {file}\n"
        
        summary += f"\nTotal knowledge documents: {len(self.created_files)}\n"
        summary += "\n📿 Rishi Vashishta's dharmic wisdom is ready to guide seekers! 📿\n"
        
        summary_path = self.base_path / "VASHISHTA_KNOWLEDGE_SUMMARY.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n  ✅ Created knowledge summary")
        print(f"\n📊 Summary saved to: {summary_path}")


def main():
    """Build Vashishta's knowledge base"""
    builder = VashishtaKnowledgeBuilder()
    builder.create_all()
    
    print("\n✨ Vashishta's dharma knowledge base is complete!")
    print("   Ready to build RAG system next...")


if __name__ == "__main__":
    main()
