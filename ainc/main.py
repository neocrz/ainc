# main.py

import numpy as np
import re
import os
import json
import pickle
import threading
import collections
from dotenv import load_dotenv

from agents import ObserverAgent, ChatterAgent, MoodAgent, RecencyAgent, OnboardingAgent, OrchestratorAgent
from utils import print_header

# --- File Constants ---
CHAR_CARD_FILE = "char.json"
CHAT_HISTORY_FILE = "chat.json"
HDC_BRAIN_FILE = "hdc_brain.pkl"

# --- Data Persistence Functions ---
def load_session():
    try:
        with open(CHAT_HISTORY_FILE, 'r') as f:
            data = json.load(f)
            if "user_name" in data and "chat_log" in data:
                return data["user_name"], data["chat_log"]
            return None, []
    except (FileNotFoundError, json.JSONDecodeError):
        return None, []

def save_session(user_name, chat_log):
    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump({"user_name": user_name, "chat_log": chat_log}, f, indent=2)

def save_hdc_brain(hdc_system, filepath=HDC_BRAIN_FILE):
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(hdc_system, f)
    except Exception as e:
        print(f"!! Error saving HDC Brain: {e}")

def load_hdc_brain(filepath=HDC_BRAIN_FILE):
    try:
        with open(filepath, 'rb') as f:
            print("...HDC Brain state loaded from file.")
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError):
        print("!! Could not load HDC Brain. A new brain will be created.")
        return None

def save_state_in_background(user_name, chat_log, hdc_brain):
    print("... Saving state in background...")
    chat_log_copy = list(chat_log)
    save_session(user_name, chat_log_copy)
    save_hdc_brain(hdc_brain)
    print("... Background save complete.")

# --- The Core HDC Brain Class Definition ---
class HDCSystem:
    def __init__(self, dimension=10000, mood_history_length=5):
        self.dimension = dimension
        self.atomic_vectors = {}
        self.event_log = []
        self.world_state = {}
        self.character_states = {}
        self.mood_history_length = mood_history_length
        self.mood_vocabulary = {}
        self.permutation = np.random.permutation(dimension)
        self.inverse_permutation = np.argsort(self.permutation)
        self.base_time_hv = self._create_base_vector()
        self.current_time_hv = self.base_time_hv.copy()
        self._initialize_moods()

    def _initialize_moods(self):
        print("Initializing mood vocabulary...")
        for mood in ['neutral', 'happy', 'annoyed', 'curious', 'sad', 'surprised']:
            self.mood_vocabulary[mood] = self._get_vector(mood)

    def _create_base_vector(self): return np.random.choice([-1, 1], self.dimension)

    def _get_vector(self, concept_name):
        normalized_name = concept_name.lower().strip()
        if not normalized_name: return None
        if normalized_name not in self.atomic_vectors:
            print(f"    - New concept detected. Creating atomic vector for: '{normalized_name}'")
            self.atomic_vectors[normalized_name] = self._create_base_vector()
        return self.atomic_vectors[normalized_name]

    def _get_mood_vector(self, mood_name):
        normalized_mood = mood_name.lower().strip()
        if normalized_mood not in self.mood_vocabulary:
            print(f"    - New mood learned: '{normalized_mood}'. Creating vector.")
            self.mood_vocabulary[normalized_mood] = self._get_vector(normalized_mood)
        return self.mood_vocabulary[normalized_mood]

    def _permute(self, v): return v[self.permutation]
    def _bind(self, v1, v2): return v1 * v2
    def _bundle(self, vectors): return np.sum(vectors, axis=0)
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm
    def _cosine_similarity(self, v1, v2):
        v1_norm = self._normalize(v1); v2_norm = self._normalize(v2)
        return np.dot(v1_norm, v2_norm)

    def evaluate_dsl(self, dsl_string):
        dsl_string = dsl_string.strip()
        while '(' in dsl_string:
            start = dsl_string.rfind('('); end = dsl_string.find(')', start)
            sub_expression = dsl_string[start+1:end]
            placeholder = f"__placeholder_{len(self.atomic_vectors)}__"
            self.atomic_vectors[placeholder] = self.evaluate_dsl(sub_expression)
            dsl_string = dsl_string[:start] + placeholder + dsl_string[end+1:]
        if '+' in dsl_string:
            parts = dsl_string.split('+')
            return self._bundle([v for v in [self.evaluate_dsl(p) for p in parts] if v is not None])
        if '*' in dsl_string:
            parts = dsl_string.split('*')
            result_vec = self.evaluate_dsl(parts[0])
            for i in range(1, len(parts)):
                next_vec = self.evaluate_dsl(parts[i])
                if result_vec is not None and next_vec is not None:
                    result_vec = self._bind(result_vec, next_vec)
                else:
                    return None
            return result_vec
        return self._get_vector(dsl_string)

    def process_dsl_event(self, dsl_event_info):
        print(f"\n-> Processing Event: {dsl_event_info['description']}")
        event_content_hv = self.evaluate_dsl(dsl_event_info['dsl'])
        if event_content_hv is None:
            print("   Skipping event log, DSL evaluated to None.")
            return

        self.current_time_hv = self._permute(self.current_time_hv)
        memory_chunk = {'content': event_content_hv, 'time': self.current_time_hv, 'dsl_info': dsl_event_info}
        self.event_log.append(memory_chunk)
        print(f"   Event logged at Time Tick {len(self.event_log)}.")

        meta = dsl_event_info.get('meta', {})
        if 'mood_impact_info' in meta:
            info = meta['mood_impact_info']
            target = info['target'].lower()
            mood_name = info['mood'].lower()
            intensity = info.get('intensity', 1.0)
            
            if target not in self.character_states:
                self.character_states[target] = {
                    'recent_mood_impacts': collections.deque(maxlen=self.mood_history_length)
                }

            impact_mood_hv = self._get_mood_vector(mood_name)
            if impact_mood_hv is not None:
                self.character_states[target]['recent_mood_impacts'].append(impact_mood_hv * intensity)
                print(f"   MOOD IMPACT: '{target.capitalize()}' felt a moment of '{mood_name}'.")

    def get_general_context(self, message, top_k=3):
        query_is_dsl = any(role in message for role in ['_role*'])
        if query_is_dsl:
            context_query_hv = self.evaluate_dsl(message)
        else:
            stop_words = {'i', 'me', 'my', 'a', 'an', 'the', 'to', 'is', 'in', 'it', 'and', 'of', 'for', 'on', 'with'}
            words = re.findall(r'\b\w+\b', message.lower())
            key_concepts = [word for word in words if word not in stop_words]
            if not key_concepts: return []
            context_query_hv = self._bundle([self._get_vector(c) for c in key_concepts if self._get_vector(c) is not None])
        if context_query_hv is None: return []
        results = [{'sim': self._cosine_similarity(context_query_hv, c['content']), 'mem': c['dsl_info']} for c in self.event_log]
        results.sort(key=lambda x: x['sim'], reverse=True)
        return results[:top_k]


### =======================================================================
### **MAIN SCRIPT LOGIC**
### =======================================================================

# SETUP
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("FATAL ERROR: GEMINI_API_KEY not found in .env file."); exit()

# INITIALIZATION
user_name = None; character_data = {}; chat_log = []; hdc_brain = None
try:
    with open(CHAR_CARD_FILE, 'r') as f: character_data = json.load(f)
    char_name = character_data["name"]; char_persona = character_data["persona"]
except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
    print(f"FATAL ERROR: Could not load or parse '{CHAR_CARD_FILE}'. Details: {e}"); exit()

# AGENT INSTANTIATION
observer_agent = ObserverAgent(api_key=GEMINI_API_KEY)
chatter_agent = ChatterAgent(api_key=GEMINI_API_KEY, character_name=char_name)
chatter_agent.system_instruction = char_persona
orchestrator_agent = OrchestratorAgent(api_key=GEMINI_API_KEY)

# LOGIC BRANCH: LOAD GAME OR START NEW
if os.path.exists(CHAT_HISTORY_FILE) and os.path.exists(HDC_BRAIN_FILE):
    print_header(f"Found existing session for {char_name}! Loading game...")
    user_name, chat_log = load_session()
    hdc_brain = load_hdc_brain()
    if not hdc_brain or not user_name:
        print("Critical error loading session. Please delete save files and restart."); exit()
    print(f"Welcome back, {user_name}!")
else:
    print_header(f"Starting New Game with {char_name}")
    while not user_name:
        user_name = input("Please enter your name to begin: ").strip()
    hdc_brain = HDCSystem(dimension=10_000)
    onboarding_agent = OnboardingAgent(observer_agent=observer_agent, hdc_system=hdc_brain)
    onboarding_agent.seed_character_memory(character_data, user_name)
    first_message = character_data["first_mes"].replace("{{user}}", user_name).replace("{{char}}", char_name)
    print(f"\n{char_name}: \"{first_message}\"")
    chat_log.append({'who': char_name, 'message': first_message})
    print_header("Processing first interaction into timeline")
    parsed_first_message = observer_agent.parse_message(f'{char_name}: {first_message}')
    if parsed_first_message and parsed_first_message.get('dsl'):
        event_info = {'description': first_message, 'dsl': parsed_first_message['dsl']}
        if parsed_first_message.get('mood_info'):
            event_info['meta'] = {'mood_impact_info': parsed_first_message['mood_info']}
        hdc_brain.process_dsl_event(event_info)
    save_thread = threading.Thread(target=save_state_in_background, args=(user_name, chat_log, hdc_brain))
    save_thread.start()

# FINALIZE AGENT SETUP
mood_agent = MoodAgent(hdc_system=hdc_brain)
recency_agent = RecencyAgent(hdc_system=hdc_brain)

# --- INTERACTIVE LOOP ---
print_header(f"Starting Interactive Chat with {char_name} (type 'quit' to exit)")
active_save_thread = None
while True:
    try:
        if active_save_thread and active_save_thread.is_alive():
            print("... (Waiting for save to complete) ...")
            active_save_thread.join()

        user_input = input(f"{user_name}: ")
        if user_input.lower() in ['quit', 'exit']:
            print(f"{char_name}: \"Alright, see you around.\" (Saving session...)")
            break

        # 1. OBSERVE AND LOG USER'S ACTION
        parsed_user_event = observer_agent.parse_message(f"{user_name}: {user_input}")
        if parsed_user_event:
            chat_log.append({'who': user_name, 'message': user_input})
            event_info = {'description': user_input, 'dsl': parsed_user_event['dsl']}
            if parsed_user_event.get('mood_info'):
                # Make sure speaker name is consistent
                parsed_user_event['mood_info']['target'] = user_name
                event_info['meta'] = {'mood_impact_info': parsed_user_event['mood_info']}
            hdc_brain.process_dsl_event(event_info)

        # 2. CREATE THE CONTEXT FOR THE ORCHESTRATOR
        recent_history_text = ""
        recent_messages = chat_log[-10:]
        for entry in recent_messages:
            recent_history_text += f"{entry['who']}: {entry['message']}\n"
        
        # 3. DELEGATE: GET THE QUERY PLAN FROM THE ORCHESTRATOR AGENT
        query_plan = orchestrator_agent.create_query_plan(recent_history_text)

        # 4. EXECUTE THE PLAN
        briefing_context = {
            "moods": [],
            "relevant_memories": [],
            "foundational_memory": None
        }

        if query_plan and "queries" in query_plan:
            # Always ensure the main character and user are included in mood checks
            all_chars_to_check = set(query_plan.get("characters_in_scene", []))
            all_chars_to_check.add(char_name)
            all_chars_to_check.add(user_name)

            for char in all_chars_to_check:
                 briefing_context["moods"].append({
                    "character": char,
                    "dominant": mood_agent.get_dominant_mood(char),
                    "current": mood_agent.get_current_mood(char)
                })

            for query in query_plan["queries"]:
                q_type = query.get("query_type")
                if q_type in ["relevant_memories", "foundational_memory"]:
                    topic = query.get("topic")
                    if topic:
                        # Convert natural language topic to a simple DSL query
                        # This creates a broad query vector for the topic
                        topic_dsl = f"entities_present_role*{topic.lower().replace(' ', '_')} + object_role*{topic.lower().replace(' ', '_')}"
                        if q_type == "relevant_memories":
                            briefing_context["relevant_memories"].extend(
                                hdc_brain.get_general_context(topic_dsl, top_k=2)
                            )
                        # Only get the first foundational memory to avoid clutter
                        elif q_type == "foundational_memory" and not briefing_context["foundational_memory"]:
                             briefing_context["foundational_memory"] = recency_agent.get_event(topic_dsl, order='oldest')
        
        recent_history_text = ""
        recent_messages = chat_log[-10:]
        for entry in recent_messages:
            recent_history_text += f"{entry['who']}: {entry['message']}\n"
        
        briefing = "[A. SCENE'S EMOTIONAL STATE]\n"
        if briefing_context["moods"]:
            for mood_info in briefing_context["moods"]:
                if mood_info['character'].lower() == char_name.lower():
                    briefing += f"  - Your ({char_name}) dominant mood is: {mood_info['dominant']}. Your immediate reaction was: {mood_info['current']}.\n"
                else:
                    briefing += f"  - {mood_info['character']}'s dominant mood seems to be: {mood_info['dominant']}.\n"
        else:
            briefing += "  - Emotional states are unclear.\n"

        briefing += "\n[B. TOPIC-RELEVANT MEMORIES]\n"
        if briefing_context["relevant_memories"]:
            seen_descs = set()
            for item in briefing_context["relevant_memories"]:
                if item['mem']['description'] not in seen_descs:
                    briefing += f"  - '{item['mem']['description']}'\n"
                    seen_descs.add(item['mem']['description'])
        else:
            briefing += "  - No specific memories found on these topics.\n"
        
        briefing += "\n[C. FOUNDATIONAL MEMORY]\n"
        fm = briefing_context.get("foundational_memory")
        briefing += f"  - The first time a key topic came up, you remember: '{fm['description']}'\n" if fm else "  - This seems to be a new topic.\n"
        
        briefing += "\n[D. RECENT CONVERSATION]\n"
        briefing += recent_history_text if recent_history_text else "  - This is the start of the conversation.\n"

        # 6. GET RESPONSE FROM CHATTER AGENT
        response_text = chatter_agent.generate_response(user_input, briefing)
        print(f"\n{chatter_agent.character_name}: {response_text}")

        # 7. LOG AI'S ACTION & SAVE
        parsed_ai_event = observer_agent.parse_message(f"{chatter_agent.character_name}: {response_text}")
        if parsed_ai_event:
            chat_log.append({'who': chatter_agent.character_name, 'message': response_text})
            event_info = {'description': response_text, 'dsl': parsed_ai_event['dsl']}
            if parsed_ai_event.get('mood_info'):
                # Ensure speaker name is correct
                parsed_ai_event['mood_info']['target'] = char_name
                event_info['meta'] = {'mood_impact_info': parsed_ai_event['mood_info']}
            hdc_brain.process_dsl_event(event_info)

        if active_save_thread and active_save_thread.is_alive():
            active_save_thread.join()
        active_save_thread = threading.Thread(target=save_state_in_background, args=(user_name, chat_log, hdc_brain))
        active_save_thread.start()

    except KeyboardInterrupt:
        print(f"\n{char_name}: \"Whoa, gotta go!\" (Force saving session...)")
        break

# Final save on exit
if active_save_thread and active_save_thread.is_alive():
    print("... (Waiting for final save to complete) ...")
    active_save_thread.join()
print("Session ended.")