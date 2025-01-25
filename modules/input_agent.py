# modules/input_agent.py
import google.generativeai as genai
import json
from typing import List, Dict

class InputAgent:
    def __init__(self):
        self.api_key = 'AIzaSyCIg8ie1z9mhMBVYY7tq1PY8iBl9vgkYfc'
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def process_query(self, user_input: str) -> List[Dict]:
        prompt = f"""Based on this user request: "{user_input}", identify which task is needed.
Available tasks:
1. Soil classification analysis
2. Crop yield prediction

Format response as a single JSON array with exactly one task like this:
[{{"task": "Soil classification analysis", "image_required": true, "additional_info_required": false}}]
or
[{{"task": "Crop yield prediction", "image_required": false, "additional_info_required": true}}]"""
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text.strip())
        except Exception as e:
            print(f"Error: {e}")
            return [{"task": "Unknown", "image_required": False, "additional_info_required": False}]