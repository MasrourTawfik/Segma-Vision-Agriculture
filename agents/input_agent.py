# modules/input_agent.py
import google.generativeai as genai
import json
import os
from typing import List, Dict

class InputAgent:
    def __init__(self):
        self.api_key = 'AIzaSyCIg8ie1z9mhMBVYY7tq1PY8iBl9vgkYfc'
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def process_query(self, user_input: str) -> List[Dict]:
        prompt = """
        You are an intelligent assistant that determines which tasks to perform based on user input.
        The tasks you can perform are:
        1. Plant disease detection/classification (requires image)
        2. Weed detection (requires image)
        3. Soil classification analysis (requires image or soil data)
        4. Agriculture crop yield prediction (requires environmental data)
        5. Plant identification (requires image)

        Instructions:
        - Analyze the query
        - Identify required task(s)
        - Specify image requirements
        - For crop yield prediction, prompt for city, days, and crop name
        
        Respond in JSON format:
        [
            {"task": "Task Name", "image_required": true/false, "additional_info_required": true/false}
        ]

        User input: {user_input}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"Error processing query: {e}")
            return []
            
    def validate_task(self, task: Dict) -> bool:
        required_keys = ['task', 'image_required', 'additional_info_required']
        return all(key in task for key in required_keys)