import numpy as np
import re
import os
import json
import pickle
import threading
import collections  # <-- THE CRITICAL FIX IS HERE
from dotenv import load_dotenv

from agents import ObserverAgent, ChatterAgent, MoodAgent, RecencyAgent, OnboardingAgent
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
            # This part is correct, it already handles None
            return self._bundle([v for v in [self.evaluate_dsl(p) for p in parts] if v is not None])
        if '*' in dsl_string:
            parts = dsl_string.split('*')
            result_vec = self.evaluate_dsl(parts[0])
            for i in range(1, len(parts)):
                next_vec = self.evaluate_dsl(parts[i])
                
                # --- THIS IS THE CORRECTED LOGIC ---
                # We only need to check if the vectors are valid objects, not if they are in a list.
                if result_vec is not None and next_vec is not None:
                    result_vec = self._bind(result_vec, next_vec)
                else:
                    # If any part of a bind operation is invalid, the entire expression is invalid.
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
    hdc_brain = HDCSystem(dimension=2048)
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

        parsed_user_event = observer_agent.parse_message(f"{user_name}: {user_input}")
        if parsed_user_event:
            chat_log.append({'who': user_name, 'message': user_input})
            event_info = {'description': user_input, 'dsl': parsed_user_event['dsl']}
            if parsed_user_event.get('mood_info'):
                event_info['meta'] = {'mood_impact_info': parsed_user_event['mood_info']}
            hdc_brain.process_dsl_event(event_info)

        print_header("Orchestrator is Gathering Context")
        char_current_mood = mood_agent.get_current_mood(char_name)
        char_dominant_mood = mood_agent.get_dominant_mood(char_name)
        user_dominant_mood = mood_agent.get_dominant_mood(user_name)
        print(f"-> Querying {char_name}'s mood: Momentary({char_current_mood}), Dominant({char_dominant_mood})")
        print(f"-> Querying {user_name}'s mood: Dominant({user_dominant_mood})")

        main_topic_dsl = None
        if parsed_user_event and parsed_user_event['dsl']:
            object_match = re.search(r'object_role\*\(?([\w\s]+)\)?', parsed_user_event['dsl'])
            if object_match:
                main_topic_dsl = f"object_role*{object_match.group(1).split()[0]}"
            else:
                attr_match = re.search(r'attribute_role\*\(?([\w\s]+)\)?', parsed_user_event['dsl'])
                if attr_match:
                    main_topic_dsl = f"attribute_role*{attr_match.group(1).split()[0]}"

        if main_topic_dsl:
            print(f"-> Main topic identified: {main_topic_dsl}")
            topic_relevant_memories = hdc_brain.get_general_context(main_topic_dsl, top_k=2)
            foundational_memory = recency_agent.get_event(main_topic_dsl, order='oldest')
        else:
            topic_relevant_memories = []
            foundational_memory = None

        recent_history_text = ""
        recent_messages = chat_log[-4:-1]
        for entry in recent_messages:
            recent_history_text += f"  - {entry['who']}: {entry['message']}\n"
        
        briefing = "[A. SCENE'S EMOTIONAL STATE]\n"
        briefing += f"  - Your ({char_name}'s) dominant mood (your underlying feeling) is: {char_dominant_mood}\n"
        briefing += f"  - Your ({char_name}'s) immediate reaction to the last event was: {char_current_mood}\n"
        briefing += f"  - The user's ({user_name}'s) dominant mood seems to be: {user_dominant_mood}\n\n"
        briefing += "[B. TOPIC-RELEVANT MEMORIES]\n"
        briefing += "".join([f"  - '{item['mem']['description']}'\n" for item in topic_relevant_memories]) if topic_relevant_memories else "  - No specific memories found on this topic.\n"
        briefing += "\n[C. FOUNDATIONAL MEMORY]\n"
        briefing += f"  - The first time this topic came up, you remember: '{foundational_memory['description']}'\n" if foundational_memory else "  - This seems to be a new topic.\n"
        briefing += "\n[D. RECENT CONVERSATION]\n"
        briefing += recent_history_text if recent_history_text else "  - This is the start of the conversation.\n"

        response_text = chatter_agent.generate_response(user_input, briefing)
        print(f"\n{chatter_agent.character_name}: {response_text}")

        parsed_ai_event = observer_agent.parse_message(f"{chatter_agent.character_name}: {response_text}")
        if parsed_ai_event:
            chat_log.append({'who': chatter_agent.character_name, 'message': response_text})
            event_info = {'description': response_text, 'dsl': parsed_ai_event['dsl']}
            if parsed_ai_event.get('mood_info'):
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