import numpy as np
import re
import os
import json
import pickle
import threading # --- NEW --- For asynchronous saving
from dotenv import load_dotenv

from agents import ObserverAgent, ChatterAgent, MoodAgent, RecencyAgent, OnboardingAgent
from utils import print_header

# --- File Constants ---
CHAR_CARD_FILE = "char.json"
CHAT_HISTORY_FILE = "chat.json"
HDC_BRAIN_FILE = "hdc_brain.pkl"

# --- Data Persistence for Session ---
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

# --- HDC Brain Persistence ---
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
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        print(f"!! Could not load HDC Brain: {e}. A new brain will be created.")
        return None

# --- NEW: Asynchronous Saving Worker ---
def save_state_in_background(user_name, chat_log, hdc_brain):
    """
    A worker function to save session and brain state in a separate thread.
    This prevents the UI from lagging during save operations.
    """
    print("... Saving state in background...")
    # Make copies to prevent race conditions, especially with the chat_log list
    chat_log_copy = list(chat_log)
    
    save_session(user_name, chat_log_copy)
    save_hdc_brain(hdc_brain) # pickle handles the object state safely
    print("... Background save complete.")

# --- The Core HDC Brain (Class definition is unchanged) ---
class HDCSystem:
    # ... (this class is unchanged)
    def __init__(self, dimension=10000):
        self.dimension = dimension
        self.atomic_vectors = {}
        self.event_log = []
        self.world_state = {}
        self.character_states = {}
        self.mood_vocabulary = {}
        self.permutation = np.random.permutation(dimension)
        self.inverse_permutation = np.argsort(self.permutation)
        self.base_time_hv = self._create_base_vector()
        self.current_time_hv = self.base_time_hv.copy()
        self._initialize_moods()
    def _initialize_moods(self):
        print("Initializing mood vocabulary...")
        for mood in ['neutral', 'happy', 'annoyed', 'curious']: self.mood_vocabulary[mood] = self._get_vector(mood)
    def _create_base_vector(self): return np.random.choice([-1, 1], self.dimension)
    def _get_vector(self, concept_name):
        normalized_name = concept_name.lower().strip()
        if not normalized_name: return None
        if normalized_name not in self.atomic_vectors:
            print(f"    - New concept detected. Creating atomic vector for: '{normalized_name}'")
            self.atomic_vectors[normalized_name] = self._create_base_vector()
        return self.atomic_vectors[normalized_name]
    def _permute(self, v): return v[self.permutation]
    def _bind(self, v1, v2): return v1 * v2
    def _bundle(self, vectors): return np.sum(vectors, axis=0)
    def _cosine_similarity(self, v1, v2):
        norm_v1 = np.linalg.norm(v1); norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)
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
            valid_vectors = [v for v in [self.evaluate_dsl(p) for p in parts] if v is not None]
            return self._bundle(valid_vectors)
        if '*' in dsl_string:
            parts = dsl_string.split('*')
            result_vec = self.evaluate_dsl(parts[0])
            for i in range(1, len(parts)):
                next_vec = self.evaluate_dsl(parts[i])
                if result_vec is not None and next_vec is not None: result_vec = self._bind(result_vec, next_vec)
            return result_vec
        return self._get_vector(dsl_string)
    def process_dsl_event(self, dsl_event_info):
        print(f"\n-> Processing Event: {dsl_event_info['description']}")
        event_content_hv = self.evaluate_dsl(dsl_event_info['dsl'])
        self.current_time_hv = self._permute(self.current_time_hv)
        memory_chunk = {'content': event_content_hv, 'time': self.current_time_hv, 'dsl_info': dsl_event_info}
        self.event_log.append(memory_chunk)
        print(f"   Event logged at Time Tick {len(self.event_log)}.")
        meta = dsl_event_info.get('meta', {})
        if 'mood_impact_info' in meta:
            info = meta['mood_impact_info']
            target = info['target'].lower(); mood_name = info['mood'].lower()
            intensity = info.get('intensity', 1.0)
            if target not in self.character_states: self.character_states[target] = {'mood_hv': self.mood_vocabulary['neutral'].copy()}
            current_mood_hv = self.character_states[target]['mood_hv']
            impact_mood_hv = self.mood_vocabulary[mood_name]
            self.character_states[target]['mood_hv'] = current_mood_hv + (impact_mood_hv * intensity)
            print(f"   MOOD CHANGE: '{target.capitalize()}' is now more '{mood_name}'.")
    def get_general_context(self, message, top_k=3):
        stop_words = {'i', 'me', 'my', 'should', 'go', 'a', 'about', 'an', 'the', 'to'}
        words = re.findall(r'\b\w+\b', message.lower())
        key_concepts = [word for word in words if word not in stop_words]
        if not key_concepts: return []
        context_query_hv = self._bundle([self._get_vector(c) for c in key_concepts if self._get_vector(c) is not None])
        results = [{'sim': self._cosine_similarity(context_query_hv, c['content']), 'mem': c['dsl_info']} for c in self.event_log]
        results.sort(key=lambda x: x['sim'], reverse=True)
        return results[:top_k]

### =======================================================================
### **MAIN SCRIPT LOGIC**
### =======================================================================

# --- The "New Game" and "Load Game" blocks are unchanged ---
# ... (all the code from `load_dotenv()` to the start of the `while` loop is the same)
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
    print(f"FATAL ERROR: Could not load or parse '{CHAR_CARD_FILE}'. A valid character card is required. Details: {e}"); exit()
# AGENT INSTANTIATION
observer_agent = ObserverAgent(api_key=GEMINI_API_KEY)
chatter_agent = ChatterAgent(api_key=GEMINI_API_KEY, character_name=char_name); chatter_agent.system_instruction = char_persona
# LOGIC BRANCH
if os.path.exists(CHAT_HISTORY_FILE) and os.path.exists(HDC_BRAIN_FILE):
    print_header(f"Found existing session for {char_name}! Loading game...")
    user_name, chat_log = load_session()
    hdc_brain = load_hdc_brain()
    if not hdc_brain or not user_name: print("Critical error loading session files. Please delete chat.json and hdc_brain.pkl and restart."); exit()
    print(f"Welcome back, {user_name}!")
else:
    print_header(f"Starting New Game with {char_name}")
    while not user_name:
        user_name = input("Please enter your name to begin: ").strip()
        if not user_name: print("A name is required to start the adventure.")
    hdc_brain = HDCSystem(dimension=2048)
    onboarding_agent = OnboardingAgent(observer_agent=observer_agent, hdc_system=hdc_brain)
    onboarding_agent.seed_character_memory(character_data, user_name)
    first_message = character_data["first_mes"].replace("{{user}}", user_name).replace("{{char}}", char_name)
    print(f"\n{char_name}: \"{first_message}\"")
    initial_chat_events = [{'who': char_name, 'message': first_message}]
    chat_log.extend(initial_chat_events)
    print_header("Processing first interaction into timeline")
    first_message_dsl = observer_agent.parse_message(f'{char_name}: {first_message}')
    if first_message_dsl: hdc_brain.process_dsl_event({'description': first_message, 'dsl': first_message_dsl})
    # MODIFIED: Use the new threaded save function for the first save
    save_thread = threading.Thread(target=save_state_in_background, args=(user_name, chat_log, hdc_brain))
    save_thread.start()
# FINALIZE AGENT SETUP
mood_agent = MoodAgent(hdc_system=hdc_brain)
recency_agent = RecencyAgent(hdc_system=hdc_brain)

# --- INTERACTIVE LOOP ---
print_header(f"Starting Interactive Chat with {char_name} (type 'quit' to exit)")
active_save_thread = None # Keep track of the save thread
while True:
    try:
        # Wait for any previous save to finish before proceeding
        if active_save_thread and active_save_thread.is_alive():
            print("... (Waiting for save to complete) ...")
            active_save_thread.join()

        user_input = input(f"{user_name}: ")
        if user_input.lower() in ['quit', 'exit']:
            print(f"{char_name}: \"Alright, see you around.\" (Saving session...)")
            break

        # A. LOG USER'S ACTION
        user_event_dsl = observer_agent.parse_message(f"{user_name}: {user_input}")
        if user_event_dsl:
            user_event = {'who': user_name, 'message': user_input}
            hdc_brain.process_dsl_event({'description': user_input, 'dsl': user_event_dsl})
            chat_log.append(user_event)

        # B. ORCHESTRATOR GATHERS CONTEXT (NOW WITH CHAT HISTORY)
        general_context = hdc_brain.get_general_context(user_input, top_k=2)
        char_mood = mood_agent.get_mood(char_name)
        first_sword_event = recency_agent.get_event("object_role*sword", order='oldest')

        # --- NEW: Build the recent conversation history snippet ---
        recent_history_text = ""
        # Get the last 3 items from the log. Slicing is safe for lists < 3 items.
        recent_messages = chat_log[-3:]
        for entry in recent_messages:
            recent_history_text += f"  - {entry['who']}: {entry['message']}\n"
        
        # --- MODIFIED: Construct the enhanced briefing ---
        briefing = f"[A. CURRENT CHARACTER STATES]\n  - {char_name}'s current mood is: {char_mood}\n\n"
        briefing += "[B. GENERAL RELEVANT MEMORIES]\n"
        briefing += "".join([f"  - '{item['mem']['description']}'\n" for item in general_context]) if general_context else "  - No general memories found.\n"
        briefing += "\n[C. SPECIFIC HISTORICAL CONTEXT]\n"
        briefing += f"  - The foundational memory about this topic is: '{first_sword_event['description']}'\n" if first_sword_event else "  - No specific historical event found.\n"
        briefing += "\n[D. RECENT CONVERSATION]\n"
        briefing += recent_history_text if recent_history_text else "  - This is the start of the conversation.\n"

        # C. CHATTER AGENT RESPONDS
        response_text = chatter_agent.generate_response(user_input, briefing)
        print(f"\n{chatter_agent.character_name}: {response_text}")

        # D. LOG THE AI'S OWN ACTION
        ai_event_dsl = observer_agent.parse_message(f"{chatter_agent.character_name}: {response_text}")
        if ai_event_dsl:
            ai_event = {'who': chatter_agent.character_name, 'message': response_text}
            hdc_brain.process_dsl_event({'description': response_text, 'dsl': ai_event_dsl})
            chat_log.append(ai_event)

        # E. --- MODIFIED: SAVE ASYNCHRONOUSLY ---
        if active_save_thread and active_save_thread.is_alive():
            active_save_thread.join() # Ensure the previous save is done before starting a new one
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