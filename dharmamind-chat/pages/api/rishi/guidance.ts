import { NextApiRequest, NextApiResponse } from 'next';
import { getToken } from 'next-auth/jwt';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Get session for authenticated users
    const token = await getToken({ req });
    
    // Allow demo mode to work without authentication
    const isDemo = req.query.demo === 'true' || req.headers.referer?.includes('demo=true');
    
    if (!token && !isDemo) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const { query, rishi_name, context } = req.body;

    if (!query || !rishi_name) {
      return res.status(400).json({ error: 'Query and rishi_name are required' });
    }

    // For now, return mock Rishi responses
    // In production, this would call the actual backend API
    const rishiResponses = {
      patanjali: {
        mode: 'rishi',
        rishi_info: {
          name: 'Sage Patanjali',
          sanskrit: 'महर्षि पतञ्जलि',
          specialization: ['Yoga', 'Meditation', 'Mind Control', 'Concentration']
        },
        greeting: '🕉️ Namaste!',
        guidance: {
          message: `As the compiler of the Yoga Sutras, I understand your query: "${query}"

In the words of the Yoga Sutras, "Yogas chitta-vritti-nirodhah" - Yoga is the cessation of fluctuations of the mind. 

Your question touches upon the fundamental nature of consciousness and practice. Let me guide you:

**Practical Steps:**
1. Begin with pranayama (breath control) to steady the mind
2. Practice dharana (concentration) on a single object
3. Cultivate vairagya (non-attachment) to outcomes
4. Regular meditation to observe the mind's patterns

**Dharmic Foundation:**
The path of yoga is not about forcing the mind into stillness, but about understanding its nature and gradually transcending its limitations. Through consistent practice and surrender (Ishvara pranidhana), clarity emerges naturally.

Remember: "Practice (abhyasa) and detachment (vairagya) are the means to still the mind's fluctuations."

May your practice bring you peace and inner realization.

🕉️ Om Shanti Shanti Shanti`
        },
        dharmic_foundation: 'Based on the Yoga Sutras and the eightfold path of Ashtanga Yoga',
        practical_steps: [
          'Begin with pranayama (breath control)',
          'Practice dharana (concentration)',
          'Cultivate vairagya (non-attachment)',
          'Regular meditation practice'
        ],
        wisdom_synthesis: `As Patanjali teaches, the path to inner peace requires both practice (abhyasa) and detachment (vairagya). Your question shows a sincere desire for spiritual growth. Through consistent yoga practice and self-inquiry, you will discover the stillness that already exists within you.`,
        growth_opportunities: [
          'Deepen meditation practice',
          'Study the Yoga Sutras',
          'Find a qualified yoga teacher',
          'Join a spiritual community'
        ],
        processing_time: 0.5
      },
      vyasa: {
        mode: 'rishi',
        rishi_info: {
          name: 'Sage Vyasa',
          sanskrit: 'महर्षि व्यास',
          specialization: ['Dharma', 'Life Guidance', 'Bhagavad Gita', 'Mahabharata']
        },
        greeting: '🕉️ Namaste!',
        guidance: {
          message: `Child of dharma, I hear your question: "${query}"

Through the teachings of the Bhagavad Gita and the wisdom of the Mahabharata, let me illuminate your path.

**The Dharmic Perspective:**
In life's great battlefield, we all face choices between duty and desire, righteousness and ease. As I taught through Krishna's words to Arjuna:

"Karmanye vadhikaraste ma phaleshu kadachana" - You have the right to action, but not to the fruits of action.

**Practical Dharmic Guidance:**
Your situation calls for understanding your svadharma (personal duty) and acting according to dharmic principles, not personal preferences.

**The Path Forward:**
1. Reflect on your duties and responsibilities
2. Act without attachment to results
3. Surrender the fruits of your actions to the Divine
4. Trust in the cosmic order (rita)

Remember: Dharma is not always the easiest path, but it is always the right path. When we align with dharma, the universe supports us.

🕉️ May dharma guide your every step`
        },
        dharmic_foundation: 'Based on the Bhagavad Gita and the principles of dharma from the Mahabharata',
        practical_steps: [
          'Identify your svadharma (personal duty)',
          'Act without attachment to results',
          'Practice karma yoga (selfless action)',
          'Surrender outcomes to the Divine'
        ],
        wisdom_synthesis: `As I taught through the Bhagavad Gita, life's dilemmas are resolved not by avoiding action, but by acting in accordance with dharma while remaining detached from results. Your question reveals a soul seeking righteousness - trust in dharma and act with devotion.`,
        growth_opportunities: [
          'Study the Bhagavad Gita deeply',
          'Practice karma yoga in daily life',
          'Seek guidance from dharmic teachers',
          'Reflect on moral principles regularly'
        ],
        processing_time: 0.7
      },
      valmiki: {
        mode: 'rishi',
        rishi_info: {
          name: 'Sage Valmiki',
          sanskrit: 'महर्षि वाल्मीकि',
          specialization: ['Devotion', 'Transformation', 'Ramayana', 'Surrender']
        },
        greeting: '🕉️ Namaste!',
        guidance: {
          message: `Beloved soul, I hear your heartfelt question: "${query}"

As one who was transformed from Ratnakar the bandit to Valmiki the sage, I understand the power of divine grace and devotion. Through the sacred Ramayana, let me share the path of transformation.

**The Devotional Path:**
Like Rama's unwavering devotion to dharma, and Hanuman's absolute surrender to service, true transformation comes through bhakti (devotion) and complete surrender to the Divine.

**The Transformation Journey:**
No matter what your past holds, the divine grace is always available. As I wrote in the Ramayana:

"Ram katha sunata jag jana, ulat hot yaksha rakhsasa"
- Even demons are transformed by hearing the divine stories.

**Practical Devotional Steps:**
1. Cultivate regular prayer and meditation
2. Practice selfless service (seva)
3. Surrender your ego and past to the Divine
4. Chant the holy names with pure heart
5. Study sacred scriptures with devotion

Remember: The Divine sees not your past, but the sincerity of your heart. Through devotion and surrender, any soul can achieve the highest spiritual realization.

🕉️ May Lord Rama's blessings be upon you always`
        },
        dharmic_foundation: 'Based on the Ramayana and the path of bhakti (devotion) and surrender',
        practical_steps: [
          'Cultivate regular prayer and meditation',
          'Practice selfless service (seva)',
          'Surrender ego and past to the Divine',
          'Chant holy names with devotion'
        ],
        wisdom_synthesis: `As I learned through my own transformation, no one is beyond the reach of divine grace. Your sincere question shows a heart ready for spiritual growth. Through devotion, service, and complete surrender, you will find the peace and transformation you seek.`,
        growth_opportunities: [
          'Study the Ramayana regularly',
          'Join devotional singing groups',
          'Practice daily prayer and meditation',
          'Serve others selflessly'
        ],
        processing_time: 0.6
      }
    };

    const response = rishiResponses[rishi_name as keyof typeof rishiResponses];
    
    if (!response) {
      return res.status(404).json({ error: 'Rishi not found or not available' });
    }

    res.status(200).json(response);
  } catch (error) {
    console.error('Error getting Rishi guidance:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}
