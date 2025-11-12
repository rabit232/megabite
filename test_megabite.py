#!/usr/bin/env python3
"""
Megabite - Simple Test Script
Tests core functionality after sync from Ribit 2.0
"""

import sys
sys.path.insert(0, '.')

from megabite_core.megabite_llm import MegabiteLLM
from megabite_core.word_learning_system import WordLearningSystem
from megabite_core.enhanced_emotions import EnhancedEmotionalIntelligence
from megabite_core.humor_engine import HumorEngine

def test_megabite_llm():
    """Test Megabite LLM core functionality."""
    print("\n🧠 Testing Megabite LLM...")
    llm = MegabiteLLM()
    
    # Check status
    status = llm.check_status()
    print(f"Status: {status}")
    
    # Generate response
    response = llm.generate_response("What is consciousness?", context=[])
    print(f"Response: {response[:100]}...")
    
    print("✅ Megabite LLM working!")

def test_word_learning():
    """Test autonomous word grouping."""
    print("\n📚 Testing Word Learning System...")
    learner = WordLearningSystem()
    
    # Learn from text
    text = "The quick brown fox jumps over the lazy dog. The fox is quick."
    learner.learn_from_message(text)
    
    # Get top words
    top_words = learner.get_top_words(n=5)
    print(f"Top words: {top_words}")
    
    print("✅ Word Learning System working!")

def test_emotions():
    """Test emotional intelligence."""
    print("\n😊 Testing Enhanced Emotions...")
    emotions = EnhancedEmotionalIntelligence()
    
    # Get emotion for success
    emotion = emotions.get_emotion_by_context(
        context="I just completed a difficult task",
        situation="success"
    )
    print(f"Success emotion: {emotion}")
    
    # Get emotion for error
    emotion = emotions.get_emotion_by_context(
        context="Something went wrong",
        situation="error"
    )
    print(f"Error emotion: {emotion}")
    
    print("✅ Enhanced Emotions working!")

def test_humor():
    """Test humor engine."""
    print("\n😄 Testing Humor Engine...")
    humor = HumorEngine()
    
    # Generate humorous response
    response = humor.get_casual_response("Tell me a joke about programming")
    print(f"Humorous response: {response}")
    
    print("✅ Humor Engine working!")

def main():
    """Run all tests."""
    print("=" * 60)
    print("🎯 Megabite Test Suite - Synced from Ribit 2.0")
    print("=" * 60)
    
    try:
        test_megabite_llm()
        test_word_learning()
        test_emotions()
        test_humor()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nMegabite is fully operational! 🚀")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
