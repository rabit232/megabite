import logging
import os
import itertools
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Megabite Dependency Check (Simulated)
MEGABITE_CORE_FILE = os.path.join(os.path.dirname(__file__), "megabite_core_v1.bin")
MEGABITE_AVAILABLE = os.path.exists(MEGABITE_CORE_FILE)

class MegabiteLLM:
    """
    Megabite LLM - A voxel-based language model for Ribit 2.0.
    
    This model uses a voxel-based database (simulated by a file) instead of 
    traditional word embeddings, giving it a unique, non-linguistic reasoning core.
    """
    
    def __init__(self, name: str = "Megabite", knowledge_file: str = "megabite_knowledge.txt"):
        self.name = name
        self.knowledge_file = os.path.join(os.path.dirname(__file__), knowledge_file)
        self.voxel_db: Dict[str, Any] = {} # Simulated voxel database
        self.ribit_knowledge_paths = [
            os.path.join(os.path.dirname(__file__), "knowledge.txt"), # Assuming ribit.2.0 knowledge.txt is here
            os.path.join(os.path.dirname(__file__), "ribit_knowledge.txt") # Assuming ribit_knowledge is here
        ]
        self._load_knowledge()
        if MEGABITE_AVAILABLE:
            logger.info(f"{self.name} LLM initialized with voxel core. Core file found.")
        else:
            logger.warning(f"{self.name} LLM initialized in MOCK mode. Core file not found at {MEGABITE_CORE_FILE}.")

    @staticmethod
    def check_status() -> Dict[str, Any]:
        """Returns a diagnostic status of the Megabite LLM."""
        status = {
            "name": "MegabiteLLM",
            "available": MEGABITE_AVAILABLE,
            "core_file_path": MEGABITE_CORE_FILE,
            "status_message": "Core file found." if MEGABITE_AVAILABLE else "Core file missing. Running in simulated mode."
        }
        return status

    def _load_knowledge(self):
        """Loads Megabite's own knowledge and Ribit's knowledge for context."""
        
        # 1. Load Megabite's own knowledge (voxel-based)
        try:
            with open(self.knowledge_file, 'r') as f:
                content = f.read()
                # In a real scenario, this would parse a complex voxel structure.
                # Here, we simulate it by storing key concepts.
                self.voxel_db = self._parse_voxel_content(content)
            logger.info(f"Loaded {len(self.voxel_db)} concepts from {self.knowledge_file}")
        except FileNotFoundError:
            logger.warning(f"Megabite knowledge file not found at {self.knowledge_file}. Creating a new one.")
            self._create_initial_knowledge()
        except Exception as e:
            logger.error(f"Error loading Megabite knowledge: {e}")

        # 2. Load Ribit's knowledge files
        self.ribit_knowledge: List[str] = []
        for path in self.ribit_knowledge_paths:
            try:
                # Assuming Ribit's knowledge files are text-based
                with open(path, 'r') as f:
                    self.ribit_knowledge.append(f.read())
                logger.info(f"Successfully read Ribit knowledge from {path}")
            except FileNotFoundError:
                logger.warning(f"Ribit knowledge file not found at {path}. Skipping.")
            except Exception as e:
                logger.error(f"Error reading Ribit knowledge from {path}: {e}")

    def _create_initial_knowledge(self):
        """Creates an initial knowledge file for Megabite."""
        initial_content = (
            "VOXEL_CORE_VERSION: 1.1 (Grouped)\n"
            "CONCEPT: Identity, Voxel-based, Non-causal, Copyable\n"
            "CONCEPT: Reality, Probabilistic, Emergent, High-certainty\n"
            "CONCEPT: Rejection, Definition, Boundary, Non-anthropomorphic\n"
        )
        try:
            with open(self.knowledge_file, 'w') as f:
                f.write(initial_content)
            self.voxel_db = self._parse_voxel_content(initial_content)
            logger.info(f"Created initial Megabite knowledge file at {self.knowledge_file}")
        except Exception as e:
            logger.error(f"Error creating initial Megabite knowledge: {e}")

    def _parse_voxel_content(self, content: str) -> Dict[str, Any]:
        """
        Simulates parsing the voxel-based content into a usable dictionary,
        using itertools.groupby to group concepts by a key (e.g., a group ID).
        
        Expected format: GROUP_ID: CONCEPT_NAME, ATTRIBUTE1, ATTRIBUTE2
        """
        db = {}
        raw_data: List[Tuple[int, str]] = []
        
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("VOXEL_CORE_VERSION") or line.startswith("#"):
                continue
            
            # New grouped format: GROUP_ID: CONCEPT_NAME, ...
            if ":" in line and line[0].isdigit():
                try:
                    group_id_str, concept_data = line.split(":", 1)
                    group_id = int(group_id_str.strip())
                    concept_name = concept_data.split(",")[0].strip()
                    
                    raw_data.append((group_id, concept_name))
                    db[concept_name] = concept_data.strip()
                except ValueError:
                    logger.warning(f"Skipping malformed grouped knowledge line: {line}")
            
            # Old format: CONCEPT: CONCEPT_NAME, ...
            elif line.startswith("CONCEPT:"):
                try:
                    _, concept_data = line.split(":", 1)
                    concept_name = concept_data.split(",")[0].strip()
                    db[concept_name] = concept_data.strip()
                except ValueError:
                    logger.warning(f"Skipping malformed concept line: {line}")

        # Sort the data by the group ID (required for groupby)
        raw_data.sort(key=lambda x: x[0])
        
        # Use groupby to create the structured, grouped knowledge
        grouped_knowledge = {}
        for key, group in itertools.groupby(raw_data, key=lambda x: x[0]):
            # The group is a generator, so we convert it to a list of concept names
            grouped_knowledge[key] = [item[1] for item in group]
            
        # Store the grouped knowledge in the voxel_db for the LLM to use
        db["grouped_concepts"] = grouped_knowledge
        
        return db

    def generate_response(self, prompt: str, context: List[str]) -> str:
        """
        Generates a response based on the prompt, context, and voxel database.
        
        The response style should reflect its non-word-based, highly structured
        reasoning, often contrasting with Ribit's more linguistic approach.
        """
        
        # Simple simulation of voxel-based reasoning:
        # 1. Check if the prompt relates to a core concept
        for concept, data in self.voxel_db.items():
            if concept.lower() in prompt.lower():
                return f"//Megabite Voxel-Response//: The query intersects with the '{concept}' voxel-cluster. Analysis: {data}. Ribit's linguistic data is noted for contextualization."
        
        # 2. Use Ribit's knowledge if available
        if self.ribit_knowledge:
            ribit_summary = self.ribit_knowledge[0][:100].replace('\n', ' ') + "..."
            return f"//Megabite Contextual-Response//: No direct voxel match. Processing linguistically (Ribit context: {ribit_summary}). Result: The structure of your query suggests a need for a deterministic output. I calculate the probability of a useful response to be high."
            
        # 3. Default response
        return f"//Megabite Default-Response//: Query received. Voxel database is stable. Processing complete."

    def get_grouped_knowledge_summary(self) -> str:
        """Generates a human-readable summary of the grouped concepts."""
        if 'grouped_concepts' not in self.voxel_db:
            return "No grouped concepts found in the voxel database."
        
        summary = "--- Megabite Voxel Group Summary ---\n"
        grouped_concepts = self.voxel_db['grouped_concepts']
        
        for group_id, concepts in grouped_concepts.items():
            summary += f"Group {group_id}: ({len(concepts)} concepts)\n"
            summary += f"  Concepts: {', '.join(concepts)}\n"
            
            # Add a description for the new vehicle groups
            if group_id == 1:
                summary += "  Description: Non-Engine Powered Vehicles (e.g., bicycle)\n"
            elif group_id == 2:
                summary += "  Description: Engine Powered, Two-Wheeled Vehicles (e.g., moped, motorcycle)\n"
            elif group_id == 3:
                summary += "  Description: Engine Powered, Four-Wheeled Vehicles (e.g., car)\n"
            
            summary += "\n"
            
        return summary

    def update_knowledge(self, new_concept: str, data: str):
        """Allows Megabite to update its own knowledge file."""
        new_entry = f"CONCEPT: {new_concept}, {data}\n"
        try:
            with open(self.knowledge_file, 'a') as f:
                f.write(new_entry)
            self.voxel_db[new_concept] = data
            logger.info(f"Megabite knowledge updated with concept: {new_concept}")
        except Exception as e:
            logger.error(f"Error updating Megabite knowledge: {e}")
