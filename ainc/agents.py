# agents.py

import re
from google import genai
from google.genai import types
from prompts import OBSERVER_AGENT_PROMPT, CHATTER_AGENT_PROMPT
from utils import print_header

# --- AI-Powered Agents ---

class ObserverAgent:
    # ... (ObserverAgent class is unchanged) ...
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-1.5-flash-latest"
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
            match = re.search(r"<HDC>(.*?)</HDC>", response.text, re.DOTALL)
            if match:
                dsl_string = match.group(1).strip()
                print(f">> Gemini parsed successfully: {dsl_string}")
                return dsl_string
            else:
                print(f">> Gemini parsing FAILED. No <HDC> tag found in response: {response.text}")
                return None
        except Exception as e:
            print(f"!! An error occurred while calling Gemini API: {e}")
            return None

class ChatterAgent:
    # ... (ChatterAgent class is unchanged) ...
    def __init__(self, api_key, character_name="Kai"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-1.5-flash-latest"
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

# --- NEW: HDC Querying Agents (Formalized) ---

class MoodAgent:
    """A specialized agent to query the emotional state of characters."""
    def __init__(self, hdc_system):
        self.hdc = hdc_system

    def get_mood(self, character_name):
        char_name = character_name.lower()
        if char_name not in self.hdc.character_states:
            return 'Neutral'
        current_mood_hv = self.hdc.character_states[char_name]['mood_hv']
        best_match = 'Unknown'; max_similarity = -1.0
        for mood, mood_hv in self.hdc.mood_vocabulary.items():
            sim = self.hdc._cosine_similarity(current_mood_hv, mood_hv)
            if sim > max_similarity:
                max_similarity = sim
                best_match = mood
        return best_match.capitalize()

class RecencyAgent:
    """A specialized agent to find events based on their place in the timeline."""
    def __init__(self, hdc_system):
        self.hdc = hdc_system

    def get_event(self, context_query_dsl, order='recent', threshold=0.1):
        query_hv = self.hdc.evaluate_dsl(context_query_dsl)
        candidates = [c for c in self.hdc.event_log if self.hdc._cosine_similarity(query_hv, c['content']) > threshold]
        if not candidates:
            return None
        
        sort_reverse = True if order == 'recent' else False
        candidates.sort(key=lambda c: self.hdc._cosine_similarity(c['time'], self.hdc.current_time_hv), reverse=sort_reverse)
        return candidates[0]['dsl_info']
    
# agents.py (add this new class)

class OnboardingAgent:
    """
    A specialized agent responsible for parsing a character card and seeding
    the HDC brain with the character's core persona and background.
    """
    def __init__(self, observer_agent, hdc_system):
        self.observer = observer_agent
        self.hdc = hdc_system

    def seed_character_memory(self, character_data, user_name):
        """
        Parses the description and persona from the character card and
        populates the HDC system with these foundational memories.
        """
        print_header("Onboarding Agent: Seeding Character Persona into HDC Memory")
        
        char_name = character_data.get("name", "The character")
        description = character_data.get("description", "")
        
        # Replace placeholders
        description = description.replace("{{char}}", char_name).replace("{{user}}", user_name)

        # Break the description into individual sentences (facts)
        # This regex handles sentences ending in periods, exclamation marks, etc.
        facts = re.split(r'(?<=[.!?])\s+', description)
        
        if not facts or (len(facts) == 1 and not facts[0].strip()):
            print("-> No detailed description found to seed memory.")
            return

        print(f"-> Found {len(facts)} facts to learn about {char_name}.")
        
        for fact in facts:
            fact = fact.strip()
            if not fact:
                continue
            
            # Formulate the statement as if the system is learning it.
            # This provides clear context for the ObserverAgent.
            system_statement = f"System learns a fact: {fact}"
            
            # Use the existing ObserverAgent to parse this fact into DSL
            dsl_string = self.observer.parse_message(system_statement)
            
            if dsl_string:
                # Process this fact as a foundational event in the HDC brain
                event_info = {
                    'description': fact,
                    'dsl': dsl_string,
                    'meta': {'source': 'character_card'} 
                }
                self.hdc.process_dsl_event(event_info)
        
        print(f"-> {char_name}'s core memories have been seeded.")