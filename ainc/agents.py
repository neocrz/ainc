import json
import re
from google import genai
from google.genai import types
from prompts import OBSERVER_AGENT_PROMPT, CHATTER_AGENT_PROMPT, ORCHESTRATOR_AGENT_PROMPT
from utils import print_header

class ObserverAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-lite"
        self.system_instruction = OBSERVER_AGENT_PROMPT

    def parse_message(self, message):
        print(f"\n>> ObserverAgent received: '{message}'")
        print(">> Calling Gemini API to parse...")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[message],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    max_output_tokens=250, temperature=0.1
                )
            )
            hdc_match = re.search(r"<HDC>(.*?)</HDC>", response.text, re.DOTALL)
            mood_match = re.search(r'<MOOD\s+speaker="([^"]+)"\s+mood="([^"]+)"\s+intensity="([^"]+)">', response.text)
            if not hdc_match:
                print(f">> Gemini parsing FAILED. No <HDC> tag found in response: {response.text}")
                return None
            dsl_string = hdc_match.group(1).strip()
            print(f">> Gemini parsed successfully: {dsl_string}")
            result = {'dsl': dsl_string, 'mood_info': None}
            if mood_match:
                speaker = mood_match.group(1).strip().lower()
                mood = mood_match.group(2).strip().lower()
                try: intensity = float(mood_match.group(3).strip())
                except ValueError: intensity = 1.0
                result['mood_info'] = {'target': speaker, 'mood': mood, 'intensity': intensity}
                print(f">> Gemini detected mood: {speaker.capitalize()} is feeling {mood} (Intensity: {intensity})")
            return result
        except Exception as e:
            print(f"!! An error occurred while calling Gemini API: {e}")
            return None

class ChatterAgent:
    def __init__(self, api_key, character_name="Kai"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-lite"
        self.system_instruction = CHATTER_AGENT_PROMPT.replace("Kai", character_name)
        self.character_name = character_name
    def generate_response(self, user_message, context_briefing):
        print_header(f"Calling Chatter Agent ({self.character_name}) for Final Response")
        full_prompt = (f"The user says to you: \"{user_message}\"\n\n" f"[CONTEXT BRIEFING]\n{context_briefing}")
        print("--- Sending to Gemini: ---"); print(full_prompt); print("--------------------------")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[full_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    max_output_tokens=500, temperature=0.8
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"!! An error occurred while calling Gemini API: {e}")
            return f"({self.character_name} seems lost in thought and doesn't respond.)"

# --- HDC Querying Agents ---

class MoodAgent:
    def __init__(self, hdc_system):
        self.hdc = hdc_system

    def _find_best_match(self, query_hv):
        if query_hv is None: return 'Neutral'
        best_match = 'Neutral'; max_similarity = -2.0
        for mood, mood_hv in self.hdc.mood_vocabulary.items():
            sim = self.hdc._cosine_similarity(query_hv, mood_hv)
            if sim > max_similarity:
                max_similarity = sim; best_match = mood
        return best_match.capitalize()

    def get_current_mood(self, character_name):
        char_name = character_name.lower()
        if char_name not in self.hdc.character_states or not self.hdc.character_states[char_name]['recent_mood_impacts']:
            return 'Neutral'
        last_impact_hv = self.hdc.character_states[char_name]['recent_mood_impacts'][-1]
        return self._find_best_match(last_impact_hv)

    def get_dominant_mood(self, character_name):
        char_name = character_name.lower()
        if char_name not in self.hdc.character_states or not self.hdc.character_states[char_name]['recent_mood_impacts']:
            return 'Neutral'
        mood_history = self.hdc.character_states[char_name]['recent_mood_impacts']
        if not mood_history: return 'Neutral'
        composite_mood_hv = self.hdc._bundle(list(mood_history))
        return self._find_best_match(composite_mood_hv)

class RecencyAgent:
    def __init__(self, hdc_system):
        self.hdc = hdc_system
    def get_event(self, context_query_dsl, order='recent', threshold=0.1):
        query_hv = self.hdc.evaluate_dsl(context_query_dsl)
        if query_hv is None: return None
        candidates = [c for c in self.hdc.event_log if self.hdc._cosine_similarity(query_hv, c['content']) > threshold]
        if not candidates: return None
        sort_reverse = True if order == 'recent' else False
        candidates.sort(key=lambda c: self.hdc._cosine_similarity(c['time'], self.hdc.current_time_hv), reverse=sort_reverse)
        return candidates[0]['dsl_info']

class OnboardingAgent:
    def __init__(self, observer_agent, hdc_system):
        self.observer = observer_agent; self.hdc = hdc_system
    def seed_character_memory(self, character_data, user_name):
        print_header("Onboarding Agent: Seeding Character Persona into HDC Memory")
        char_name = character_data.get("name", "The character")
        description = character_data.get("description", "")
        description = description.replace("{{char}}", char_name).replace("{{user}}", user_name)
        facts = re.split(r'(?<=[.!?])\s+', description)
        if not facts or (len(facts) == 1 and not facts[0].strip()):
            print("-> No detailed description found to seed memory."); return
        print(f"-> Found {len(facts)} facts to learn about {char_name}.")
        for fact in facts:
            fact = fact.strip()
            if not fact: continue
            system_statement = f"System learns a fact: {fact}"
            parsed_fact = self.observer.parse_message(system_statement)
            if parsed_fact and parsed_fact.get('dsl'):
                event_info = {'description': fact, 'dsl': parsed_fact['dsl'], 'meta': {'source': 'character_card'}}
                self.hdc.process_dsl_event(event_info)
        print(f"-> {char_name}'s core memories have been seeded.")

class OrchestratorAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-lite"
        self.system_instruction = ORCHESTRATOR_AGENT_PROMPT

    def create_query_plan(self, conversation_history):
        print_header("Orchestrator Agent is creating a query plan...")
        print(f">> Orchestrator analyzing:\n{conversation_history}")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[conversation_history],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    max_output_tokens=500,
                    temperature=0.2
                )
            )
            plan = json.loads(response.text)
            print(">> Orchestrator plan received:")
            print(json.dumps(plan, indent=2))
            return plan
        except (Exception, json.JSONDecodeError) as e:
            print(f"!! Orchestrator Agent failed to create a valid plan: {e}")
            return None