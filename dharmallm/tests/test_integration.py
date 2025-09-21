#!/usr/bin/env python3
"""
Integration Test: Emotional Engine + Rishi Engine
Test the combined emotional intelligence and authentic Rishi personalities
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_emotional_rishi_integration():
    print("🕉️ Testing Emotional + Rishi Integration...\n")
    
    try:
        # Import engines
        from app.engines.emotional import (
            AdvancedEmotionalEngine,
            EmotionalState,
            create_emotional_engine
        )
        from app.engines.rishi import (
            AuthenticRishiEngine,
            RishiPersonality,
            create_authentic_rishi_engine
        )
        
        print("✅ Successfully imported both engines")
        
        # Create engine instances
        emotional_engine = create_emotional_engine()
        rishi_engine = create_authentic_rishi_engine()
        
        print("✅ Created engine instances")
        
        # Test emotional analysis + rishi response
        test_scenarios = [
            {
                "message": "I feel so lost and confused about my spiritual path",
                "expected_emotions": [EmotionalState.CONFUSION, EmotionalState.SADNESS],
                "suitable_rishis": ['vasishtha', 'bharadwaja']
            },
            {
                "message": "I'm angry at the injustice I see in the world",
                "expected_emotions": [EmotionalState.ANGER, EmotionalState.RIGHTEOUS_ANGER],
                "suitable_rishis": ['jamadagni', 'gautama']
            },
            {
                "message": "I feel deeply peaceful and connected to the divine",
                "expected_emotions": [EmotionalState.PEACE, EmotionalState.BLISS],
                "suitable_rishis": ['atri', 'vasishtha']
            }
        ]
        
        print("\n🧠💫 Testing Integrated Emotional + Rishi Responses:")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Scenario {i} ---")
            message = scenario["message"]
            print(f"User: '{message}'")
            
            # 1. Analyze emotional state
            emotional_profile = await emotional_engine.analyze_emotional_state(message)
            print(f"Emotion detected: {emotional_profile.primary_emotion.value}")
            print(f"Intensity: {emotional_profile.intensity_level.value}")
            print(f"Spiritual opportunity: {emotional_profile.spiritual_opportunity}")
            
            # 2. Generate emotional response
            emotional_response = await emotional_engine.generate_emotionally_intelligent_response(
                emotional_profile=emotional_profile,
                user_message=message,
                context={"integration_mode": True}
            )
            
            # 3. Select appropriate Rishi based on emotion
            emotion_to_rishi_map = {
                EmotionalState.CONFUSION: 'vasishtha',
                EmotionalState.SADNESS: 'bharadwaja',
                EmotionalState.ANGER: 'jamadagni',
                EmotionalState.FEAR: 'atri',
                EmotionalState.PEACE: 'vasishtha',
                EmotionalState.LOVE: 'atri'
            }
            
            selected_rishi = emotion_to_rishi_map.get(
                emotional_profile.primary_emotion, 
                'vasishtha'
            )
            
            # 4. Generate Rishi response with emotional context
            rishi_context = {
                "emotional_state": emotional_profile.primary_emotion.value,
                "intensity": emotional_profile.intensity,
                "spiritual_opportunity": emotional_profile.spiritual_opportunity,
                "healing_needed": True
            }
            
            rishi_response = rishi_engine.get_authentic_response(
                rishi_name=selected_rishi,
                query=message,
                context=rishi_context
            )
            
            # 5. Display integrated response
            print(f"\n🌟 {selected_rishi} responds:")
            print(f"Emotional validation: {emotional_response.validation}")
            print(f"Rishi wisdom: {rishi_response.get('response', 'No response')[:150]}...")
            print(f"Sanskrit wisdom: {emotional_response.sanskrit_wisdom}")
            print(f"Practice suggestion: {emotional_response.practice_suggestion}")
            
        print("\n✅ Emotional + Rishi integration testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_emotional_memory():
    """Test emotional memory and pattern learning"""
    print("\n🧠 Testing Emotional Memory & Pattern Learning...")
    
    try:
        from app.engines.emotional import create_emotional_engine
        
        engine = create_emotional_engine()
        
        # Simulate conversation with emotional memory
        user_messages = [
            "I feel anxious about my exam tomorrow",
            "The exam went well, but I'm still worried about results",
            "I got good grades! But now I'm anxious about the next challenge"
        ]
        
        emotional_history = []
        
        for msg in user_messages:
            profile = await engine.analyze_emotional_state(msg, context={
                "emotional_history": emotional_history
            })
            emotional_history.append(profile)
            
            print(f"Message: '{msg}'")
            print(f"Emotion: {profile.primary_emotion.value} ({profile.intensity_level.value})")
            
        print("✅ Emotional memory tracking successful")
        
    except Exception as e:
        print(f"❌ Emotional memory test error: {e}")

if __name__ == "__main__":
    asyncio.run(test_emotional_rishi_integration())
    asyncio.run(test_emotional_memory())
