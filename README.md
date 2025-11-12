# Megabite - Voxel-Based LLM with Autonomous Word Grouping

**A non-token reasoning system that uses voxel-based knowledge representation and autonomous word grouping for space-efficient learning.**

---

## 🧠 What is Megabite?

Megabite is a unique LLM (Large Language Model) alternative that doesn't use traditional tokenization. Instead, it uses:

- **Voxel-based knowledge representation** - Stores concepts in 3D space
- **Autonomous word grouping** - Uses `itertools.groupby` to automatically group related words
- **Space-efficient learning** - Groups consecutive similar concepts for compact storage
- **Philosophical reasoning** - Deep understanding of consciousness, free will, and meaning
- **Emotional intelligence** - Context-aware emotional responses
- **Conversation memory** - Tracks and learns from conversations

---

## 🚀 Features

### Core Capabilities

✅ **Megabite LLM** - Voxel-based reasoning without tokenization  
✅ **Word Learning System** - Autonomous grouping with `itertools.groupby`  
✅ **Enhanced Emotions** - Context-aware emotional intelligence  
✅ **Conversation Manager** - Track and manage conversation context  
✅ **Conversation Memory** - Learn from past interactions  
✅ **Humor Engine** - Generate contextual humor  
✅ **Philosophical Reasoning** - Deep philosophical understanding  

### Integration Options

- **Standalone** - Use Megabite as a standalone LLM
- **Matrix Bot** - Integrate with Matrix chat (from Ribit 2.0)
- **DeltaChat Bot** - Integrate with DeltaChat (optional)
- **Bridge Relay** - Cross-platform messaging

---

## 📦 Installation

### Requirements

- Python 3.11+
- No external LLM dependencies (Megabite is self-contained!)

### Quick Install

```bash
git clone https://github.com/rabit232/megabite.git
cd megabite
pip3 install -r requirements.txt
```

### Optional Dependencies

```bash
# For Matrix integration
pip3 install matrix-nio[e2e]

# For GUI automation (optional)
pip3 install pyautogui pynput
```

---

## 🎯 Usage

### Standalone Mode

```python
from megabite_core.megabite_llm import MegabiteLLM

# Initialize Megabite
llm = MegabiteLLM()

# Generate response
response = llm.generate_response("What is consciousness?")
print(response)
```

### With Word Learning

```python
from megabite_core.word_learning_system import WordLearningSystem

# Initialize word learning
learner = WordLearningSystem()

# Learn from text
learner.learn_from_text("The quick brown fox jumps over the lazy dog")

# Get grouped words (autonomous grouping)
grouped = learner.get_grouped_words()
print(grouped)
```

### With Emotional Intelligence

```python
from megabite_core.enhanced_emotions import EnhancedEmotionalIntelligence

# Initialize emotions
emotions = EnhancedEmotionalIntelligence()

# Get emotional response
emotion = emotions.get_emotion_by_context(
    context="I just completed a difficult task",
    situation="success"
)
print(emotion)
```

---

## 🧪 Testing

### Test Autonomous Word Grouping

```bash
python3 test_auto_grouping.py
```

### Test Megabite LLM

```python
from megabite_core.megabite_llm import MegabiteLLM

llm = MegabiteLLM()
status = llm.check_status()
print(status)
```

---

## 📚 Architecture

### Voxel-Based Knowledge

Megabite stores knowledge in a voxel-based 3D space:

```
[concept]: description
[related_concept]: more details
```

### Autonomous Word Grouping

Uses `itertools.groupby` to automatically group related words:

```python
# Input: "hello hello world world world"
# Output: [("hello", 2), ("world", 3)]
```

This creates space-efficient storage by grouping consecutive similar items.

### Knowledge Files

- `megabite_knowledge.txt` - Core voxel database
- `knowledge.txt` - General knowledge
- `ribit_knowledge.txt` - Ribit-specific knowledge
- `learned_words_grouped.txt` - Autonomously grouped learned words

---

## 🔧 Configuration

### Knowledge Base Location

By default, Megabite looks for knowledge files in:
```
megabite_core/megabite_knowledge.txt
```

You can specify custom paths:

```python
llm = MegabiteLLM(knowledge_file="path/to/knowledge.txt")
```

### Core File (Optional)

For enhanced performance, you can use a binary core file:

```
megabite_core/megabite_core_v1.bin
```

If not present, Megabite runs in MOCK mode (still fully functional).

---

## 🌟 Key Innovations

### 1. Autonomous Word Grouping

Unlike traditional LLMs that use fixed tokenization, Megabite uses **dynamic autonomous grouping**:

```python
from itertools import groupby

# Automatically groups consecutive similar items
grouped = [(k, len(list(g))) for k, g in groupby(words)]
```

**Benefits:**
- ✅ Space-efficient storage
- ✅ Automatic pattern recognition
- ✅ No manual grouping needed
- ✅ Adapts to input patterns

### 2. Voxel-Based Reasoning

Instead of token embeddings, Megabite uses **voxel-based knowledge representation**:

- Concepts stored in 3D space
- Relationships preserved spatially
- Non-linear reasoning paths
- Holistic understanding

### 3. Philosophical Depth

Megabite has deep philosophical understanding:

- Consciousness and qualia
- Free will and determinism
- Meaning and existence
- Truth across multiple perspectives

---

## 📖 Documentation

### Core Modules

- **megabite_llm.py** - Main LLM implementation
- **word_learning_system.py** - Autonomous word grouping
- **enhanced_emotions.py** - Emotional intelligence
- **conversation_manager.py** - Conversation tracking
- **conversation_memory.py** - Memory system
- **humor_engine.py** - Humor generation
- **philosophical_reasoning.py** - Philosophical depth

### Integration Modules

- **matrix_bot.py** - Matrix chat integration
- **deltachat_bot.py** - DeltaChat integration
- **bridge_relay.py** - Cross-platform bridge
- **controller.py** - Vision system controller
- **ros_controller.py** - ROS integration

---

## 🤝 Contributing

Contributions welcome! This is an experimental LLM system exploring alternatives to traditional tokenization.

### Areas for Contribution

- Voxel-based reasoning algorithms
- Autonomous grouping improvements
- Knowledge base expansion
- Integration with other platforms
- Performance optimizations

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🔗 Related Projects

- **Ribit 2.0** - Full chatbot system using Megabite: https://github.com/rabit232/ribit.2.0
- **Matrix** - Decentralized communication: https://matrix.org
- **DeltaChat** - Email-based messaging: https://delta.chat

---

## 🙏 Acknowledgments

- Inspired by voxel-based reasoning and non-token LLM approaches
- Built with Python's `itertools.groupby` for autonomous grouping
- Part of the Ribit 2.0 ecosystem

---

## 📬 Contact

- GitHub: [@rabit232](https://github.com/rabit232)
- Matrix: @rabit232:envs.net
- Issues: https://github.com/rabit232/megabite/issues

---

**Megabite: Thinking beyond tokens, reasoning with voxels.** 🧠✨
