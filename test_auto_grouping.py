import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'megabite_core'))
from megabite_core.word_learning_system import WordLearningSystem

word_learner = WordLearningSystem(storage_dir="test_word_learning")

sample_messages = [
    "The bicycle is a great vehicle for transportation.",
    "I love riding my motorcycle on sunny days.",
    "The car has four wheels and an engine.",
    "A moped is a small engine-powered vehicle.",
    "Walking is healthy and good for the environment.",
    "The airplane flies high in the sky.",
    "Boats travel on water using engines or sails.",
    "Trains are fast and efficient for long distances.",
    "The scooter is fun to ride in the city.",
    "Trucks carry heavy loads across the country.",
]

for msg in sample_messages:
    word_learner.learn_from_message(msg)

export_path = "learned_words_grouped.txt"
word_learner.export_to_megabite_knowledge(export_path)
print(f"Exported to {export_path}")
