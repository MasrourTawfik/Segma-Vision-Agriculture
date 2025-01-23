# app.py
import streamlit as st
from agents.input_agent import InputAgent
from agents.yield_prediction import CropYieldPredictor
from agents.ui_components import UIManager

class AgriculturalApp:
    def __init__(self):
        st.set_page_config(page_title="Agricultural Analysis System", layout="wide")
        self.input_agent = InputAgent()
        self.crop_predictor = CropYieldPredictor()
        self.ui_manager = UIManager()
        
    def run(self):
        self.ui_manager.render_header()
        self.ui_manager.render_sidebar()
        
        user_query = st.text_input("What would you like to analyze?")
        
        if user_query:
            tasks = self.input_agent.process_query(user_query)
            self.ui_manager.render_tasks(tasks, self.crop_predictor)

if __name__ == "__main__":
    app = AgriculturalApp()
    app.run()