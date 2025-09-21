#!/usr/bin/env python3
"""
Test the Advanced Emotional Engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_emotional_engine():
    print("🕉️ Testing Advanced Emotional Engine...\n")
    
    try:
        from app.engines.emotional import (
            AdvancedEmotionalEngine,
            EmotionalState,
            EmotionalIntensity,
            create_emotional_engine
        )
        print("✅ Successfully imported emotional engine classes")
        
        # Create engine instance
        engine = create_emotional_engine()
        print("✅ Created emotional engine instance")
        
        # Test emotional analysis
        test_messages = [
            "I'm feeling really sad and lost today",
            "I'm so angry at everything right now!",
            "I feel anxious about the future",
            "I'm experiencing deep peace and contentment",
            "I feel grateful for all my blessings"
        ]
        
        print("\n🧠 Testing Emotional Analysis:")
        import asyncio
        
        async def test_analysis():
            for msg in test_messages:
                try:
                    result = await engine.analyze_emotional_state(msg)
                    print(f"\nMessage: '{msg}'")
                    print(f"Detected: {result.primary_emotion.value} (Intensity: {result.intensity_level.value})")
                    print(f"Spiritual: {result.spiritual_opportunity}")
                    if result.dharmic_guidance:
                        print(f"Guidance: {result.dharmic_guidance[:100]}...")
                except Exception as e:
                    print(f"❌ Error analyzing '{msg}': {e}")
        
        asyncio.run(test_analysis())
        
        # Test emotional response generation
        print("\n💙 Testing Emotional Response Generation:")
        try:
            from app.engines.emotional.advanced_emotional_engine import EmotionalProfile
            
            profile = EmotionalProfile(
                primary_emotion=EmotionalState.SADNESS,
                secondary_emotions=[EmotionalState.ANXIETY],
                intensity=0.8,
                spiritual_opportunity="Transform suffering into wisdom"
            )
            
            async def test_response():
                response = await engine.generate_emotionally_intelligent_response(
                    emotional_profile=profile,
                    user_message="I lost my job and feel hopeless",
                    context={"situation": "job_loss", "support_needed": True}
                )
                return response
            
            response = asyncio.run(test_response())
            
            response = asyncio.run(test_response())
            
            print(f"Generated healing response: {response.response_text[:200]}...")
            if response.sanskrit_wisdom:
                print(f"Sanskrit wisdom: {response.sanskrit_wisdom}")
            if response.practice_suggestion:
                print(f"Practice: {response.practice_suggestion}")
            
        except Exception as e:
            print(f"❌ Error generating healing response: {e}")
            import traceback
            traceback.print_exc()
            
        print("\n✅ Emotional engine testing completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    test_emotional_engine()
