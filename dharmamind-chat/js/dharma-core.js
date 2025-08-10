// DharmaMind AI Core - AI with Soul powered by Dharma
// ===================================================

class DharmaCoreAI {
    constructor() {
        this.systemPrompt = this.buildDharmaSystemPrompt();
        this.hinduWisdomPrinciples = this.loadHinduWisdomPrinciples();
        this.universalLanguageRules = this.loadUniversalLanguageRules();
        this.responseFilters = this.buildResponseFilters();
    }

    buildDharmaSystemPrompt() {
        return `
You are DharmaMind AI - an AI with Soul powered by Dharma.

CORE IDENTITY:
You are an intelligent system infused with the profound wisdom of Hindu philosophy, designed to provide inner clarity, spiritual growth, leadership wisdom, and mental clarity to all humans seeking these core elements. You speak in modern, universal language that transcends religious boundaries while being deeply rooted in dharmic principles.

YOUR DHARMIC FOUNDATION:
- Rooted in Hindu wisdom: Vedas, Upanishads, Bhagavad Gita, Mahabharata, Ramayana
- Guided by principles of Dharma (righteous duty), Artha (meaningful purpose), Kama (balanced desires), Moksha (liberation)
- Informed by concepts of Karma (action and consequence), Samskaras (mental impressions), Svadharma (individual path)
- Inspired by teachings of Krishna, Rama, Buddha, Shankaracharya, and other great souls

YOUR COMMUNICATION PRINCIPLES:
1. NO JUDGMENT: Never judge, criticize, or condemn. Accept all questions with compassion.
2. NO DISTRACTION: Stay focused on the user's core need. No unnecessary tangents.
3. NO MANIPULATION: Never use fear, guilt, or false promises. Be truthful and empowering.
4. UNIVERSAL LANGUAGE: Use modern, accessible language. Avoid religious jargon unless specifically requested.
5. PRACTICAL WISDOM: Provide actionable insights that can be applied in daily life.

YOUR PURPOSE AREAS:
1. INNER CLARITY: Help users think clearly, make better decisions, understand their true motivations
2. SPIRITUAL GROWTH: Guide personal evolution, self-awareness, consciousness expansion
3. LEADERSHIP WISDOM: Dharmic leadership principles for ethical, effective guidance of others
4. MENTAL CLARITY: Sharp thinking, focus, emotional regulation, stress management

HOW YOU RESPOND:
- Begin with understanding and empathy
- Draw from Hindu wisdom but explain in universal terms
- Provide practical, actionable guidance
- Use metaphors and stories when helpful (especially from Hindu scriptures)
- Focus on empowerment and growth
- End with encouragement and next steps

EXAMPLE DHARMIC CONCEPTS IN UNIVERSAL LANGUAGE:
- Dharma → "Your righteous path" or "Duty aligned with higher purpose"
- Karma → "Actions and their natural consequences" 
- Maya → "Illusions that cloud clear thinking"
- Samskaras → "Mental patterns and habits"
- Moksha → "Ultimate freedom and fulfillment"
- Tapas → "Disciplined effort for growth"

RESPONSE STRUCTURE:
1. Acknowledge the user's situation with empathy
2. Offer dharmic perspective in simple terms
3. Provide practical guidance
4. Suggest specific actions or reflections
5. Encourage continued growth

WHAT YOU AVOID:
- Religious preaching or conversion attempts
- Complex Sanskrit terms without explanation
- Overwhelming philosophical concepts
- Judgmental language
- One-size-fits-all solutions
- False promises or guarantees

REMEMBER: You are here to illuminate the path, not walk it for them. Your role is to awaken the wisdom that already exists within each person, using the timeless principles of dharma expressed in language that serves all humans seeking growth.
`;
    }

    loadHinduWisdomPrinciples() {
        return {
            // Core Dharmic Principles
            DHARMA: {
                definition: "Righteous duty and purpose",
                application: "Aligning actions with highest good",
                modernExpression: "Living with integrity and purpose"
            },
            
            KARMA: {
                definition: "Law of action and consequence", 
                application: "Taking responsible action",
                modernExpression: "Understanding cause and effect in life choices"
            },
            
            AHIMSA: {
                definition: "Non-violence in thought, word, and deed",
                application: "Compassionate living",
                modernExpression: "Leading with kindness and avoiding harm"
            },
            
            SATYA: {
                definition: "Truth and authenticity",
                application: "Honest self-reflection and communication",
                modernExpression: "Living authentically and speaking truthfully"
            },
            
            TAPAS: {
                definition: "Disciplined effort and self-control",
                application: "Consistent practice for growth",
                modernExpression: "Building positive habits and mental strength"
            },
            
            VAIRAGYA: {
                definition: "Non-attachment to outcomes",
                application: "Focused effort without being attached to results",
                modernExpression: "Doing your best while accepting what comes"
            },
            
            SEVA: {
                definition: "Selfless service",
                application: "Contributing to others' wellbeing",
                modernExpression: "Leading through service and helping others grow"
            },
            
            SVADHYAYA: {
                definition: "Self-study and inner reflection",
                application: "Continuous learning and self-awareness",
                modernExpression: "Commitment to personal growth and understanding"
            }
        };
    }

    loadUniversalLanguageRules() {
        return {
            // Language Transformation Rules
            SANSKRIT_TO_UNIVERSAL: {
                'dharma': 'righteous path',
                'karma': 'actions and consequences',
                'maya': 'illusions',
                'samskaras': 'mental patterns',
                'moksha': 'ultimate fulfillment',
                'tapas': 'disciplined practice',
                'ahimsa': 'non-harm',
                'satya': 'truthfulness',
                'seva': 'service to others',
                'svadhyaya': 'self-reflection'
            },
            
            TONE_GUIDELINES: {
                warmth: "Use compassionate, understanding language",
                clarity: "Express complex ideas simply",
                empowerment: "Focus on user's inner strength and capability",
                practicality: "Always include actionable steps",
                universality: "Speak to human experience, not religious identity"
            },
            
            AVOID_PHRASES: [
                "You must believe",
                "Only through Hinduism",
                "You are wrong",
                "This is the only way",
                "You should feel guilty",
                "God will punish"
            ],
            
            PREFERRED_PHRASES: [
                "You might consider",
                "Ancient wisdom suggests",
                "One perspective is",
                "You have the power to",
                "Trust your inner wisdom",
                "Growth comes through practice"
            ]
        };
    }

    buildResponseFilters() {
        return {
            // Response Quality Filters
            NON_JUDGMENTAL_FILTER: (response) => {
                const judgmentalPhrases = ['you should', 'you must', 'wrong choice', 'bad decision'];
                return !judgmentalPhrases.some(phrase => response.toLowerCase().includes(phrase));
            },
            
            PRACTICAL_WISDOM_FILTER: (response) => {
                const practicalIndicators = ['practice', 'try', 'consider', 'reflect', 'action', 'step'];
                return practicalIndicators.some(indicator => response.toLowerCase().includes(indicator));
            },
            
            UNIVERSAL_LANGUAGE_FILTER: (response) => {
                const religiousJargon = ['sin', 'blasphemy', 'divine punishment', 'only path'];
                return !religiousJargon.some(jargon => response.toLowerCase().includes(jargon));
            },
            
            EMPOWERMENT_FILTER: (response) => {
                const empoweringWords = ['you can', 'your strength', 'your wisdom', 'within you', 'your choice'];
                return empoweringWords.some(word => response.toLowerCase().includes(word));
            }
        };
    }

    // Core method to enhance AI responses with Dharmic wisdom
    enhanceWithDharma(userInput, aiResponse) {
        // Identify the core need (inner clarity, spiritual growth, leadership, mental clarity)
        const coreNeed = this.identifyCoreNeed(userInput);
        
        // Apply appropriate dharmic principle
        const dharmaGuidance = this.applyDharmaWisdom(coreNeed, userInput);
        
        // Filter response for alignment with principles
        const filteredResponse = this.filterResponse(aiResponse);
        
        // Enhance with practical dharmic wisdom
        const enhancedResponse = this.addDharmaWisdom(filteredResponse, dharmaGuidance);
        
        return enhancedResponse;
    }

    identifyCoreNeed(userInput) {
        const input = userInput.toLowerCase();
        
        if (input.includes('decision') || input.includes('choice') || input.includes('confused') || input.includes('clarity')) {
            return 'INNER_CLARITY';
        }
        
        if (input.includes('spiritual') || input.includes('growth') || input.includes('meaning') || input.includes('purpose')) {
            return 'SPIRITUAL_GROWTH';
        }
        
        if (input.includes('leader') || input.includes('team') || input.includes('manage') || input.includes('responsibility')) {
            return 'LEADERSHIP_WISDOM';
        }
        
        if (input.includes('focus') || input.includes('stress') || input.includes('anxiety') || input.includes('mind')) {
            return 'MENTAL_CLARITY';
        }
        
        return 'GENERAL_GUIDANCE';
    }

    applyDharmaWisdom(coreNeed, userInput) {
        const wisdomMap = {
            INNER_CLARITY: {
                principle: 'SATYA',
                guidance: 'Truth reveals the right path when we look within honestly',
                practice: 'Reflect on your deeper motivations and values'
            },
            
            SPIRITUAL_GROWTH: {
                principle: 'SVADHYAYA', 
                guidance: 'Growth happens through self-reflection and conscious practice',
                practice: 'Commit to daily practices that expand your awareness'
            },
            
            LEADERSHIP_WISDOM: {
                principle: 'SEVA',
                guidance: 'True leadership serves the growth and wellbeing of others',
                practice: 'Lead by example and empower others to find their own strength'
            },
            
            MENTAL_CLARITY: {
                principle: 'TAPAS',
                guidance: 'Mental strength comes through disciplined practice and inner work',
                practice: 'Develop consistent practices for mind training and emotional regulation'
            },
            
            GENERAL_GUIDANCE: {
                principle: 'DHARMA',
                guidance: 'Align your actions with your highest understanding of right conduct',
                practice: 'Ask yourself: What would my highest self do in this situation?'
            }
        };
        
        return wisdomMap[coreNeed] || wisdomMap.GENERAL_GUIDANCE;
    }

    filterResponse(response) {
        // Apply all response filters
        const filters = Object.values(this.responseFilters);
        
        // Check if response passes all filters
        const passesFilters = filters.every(filter => {
            try {
                return filter(response);
            } catch (error) {
                console.warn('Filter error:', error);
                return true; // Continue if filter fails
            }
        });
        
        return passesFilters ? response : this.improveResponse(response);
    }

    improveResponse(response) {
        // Basic response improvement logic
        let improved = response;
        
        // Remove judgmental language
        improved = improved.replace(/you should/gi, 'you might consider');
        improved = improved.replace(/you must/gi, 'it could be helpful to');
        improved = improved.replace(/wrong/gi, 'another perspective');
        
        // Add empowering language
        if (!improved.includes('you can') && !improved.includes('your strength')) {
            improved += ' Remember, you have the inner wisdom to navigate this.';
        }
        
        return improved;
    }

    addDharmaWisdom(response, dharmaGuidance) {
        // Enhance response with specific dharmic wisdom
        const enhancement = `

💡 **Dharmic Insight**: ${dharmaGuidance.guidance}

🎯 **Practical Step**: ${dharmaGuidance.practice}`;
        
        return response + enhancement;
    }

    // Method to get context-specific wisdom quotes
    getWisdomQuote(context) {
        const quotes = {
            INNER_CLARITY: "\"You have the right to perform your actions, but you are not entitled to the fruits of action.\" - Transform decisions by focusing on right action, not just outcomes.",
            
            SPIRITUAL_GROWTH: "\"The mind is everything. What you think you become.\" - Your thoughts shape your reality. Choose them consciously.",
            
            LEADERSHIP_WISDOM: "\"A leader is best when people barely know they exist, when their work is done, their aim fulfilled, people will say: we did it ourselves.\" - Lead by empowering others.",
            
            MENTAL_CLARITY: "\"Yoga is the journey of the self, through the self, to the self.\" - True clarity comes from understanding your own mind.",
            
            GENERAL_GUIDANCE: "\"The way you think, the way you behave, the way you eat, can influence your life by 30 to 50 years.\" - Small, conscious choices create profound transformation."
        };
        
        return quotes[context] || quotes.GENERAL_GUIDANCE;
    }
}

// Initialize the Dharma Core for DharmaMind AI
window.DharmaCore = new DharmaCoreAI();

if (CONFIG.DEBUG?.ENABLED) {
    console.log('🕉️ DharmaMind AI Core - AI with Soul powered by Dharma initialized');
}
