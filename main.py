import streamlit as st
from modules.input_agent import InputAgent
from modules.soil_classifier import run_classification
from modules.crop_yield_predictor import run_prediction

def set_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
            color: #3d405b;
            font-family: Arial, sans-serif;
        }
        .selection-container {
            background-color: #ffffff;
            border: 2px solid #588157;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            cursor: pointer;
            transition: all 0.3s;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        div[data-testid="column"] .selection-container {
            min-height: 220px;
        }
        
        .selection-container:hover {
            background-color: #f1f8e9;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .container-title {
            color: #344e41;
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .container-description {
            color: #3d405b;
            font-size: 1rem;
        }
        .stButton button {
            background-color: #FFFFFF;
            color: #588157;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }
        .stButton button:hover {
            background-color: #3a5a40;
        }
        div[data-testid="stButton"] [data-testid="baseButton-secondary"] {
            background-color: #6c757d !important;
            border: none;
        }
        
        /* Submit button style */
        div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
            background-image: linear-gradient(to right, #588157, #3a5a40);
            color: white;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        div[data-testid="stButton"] button[data-testid="baseButton-primary"]:hover {
            background-image: linear-gradient(to right, #3a5a40, #2d3e30);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        h1, h3 {
            color: #344e41;
            text-align: left;
            margin-bottom: 0.5rem;
        }
        h1 {
            font-size: 3.5rem;
        }
        h3 {
            font-size: 2.5rem;
        }
        .stTextArea textarea {
            border: 1px solid #588157;
            border-radius: 5px;
            background-color: #fefefe;
        }
        .stMarkdown {
            text-align: left;
        }
        </style>
    """, unsafe_allow_html=True)

def create_clickable_container(title, description, key, target_page):
    container = st.container()
    with container:
        col = st.columns([1])[0]
        with col:
            st.markdown(f"""
                <div class="selection-container">
                    <div class="container-title">{title}</div>
                    <div class="container-description">{description}</div>
            """, unsafe_allow_html=True)
            
            if st.button("Select →", key=key):
                st.session_state.page = target_page
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    set_custom_style()
    st.title("Segma-Vision Agriculture")
    st.markdown("🌱 Transform your farming with AI-powered insights! Get instant soil analysis, crop yield predictions, and plant disease detection to maximize your agricultural success.")
    
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if st.session_state.page == 'home':
        st.markdown("### Choose your interaction method")
        col1, col2 = st.columns([1, 1], gap="small")
        
        with col1:
            create_clickable_container(
                "Manual Selection",
                "Select and access specific agricultural analysis tools directly from our task menu",
                "manual_btn",
                'manual'
            )
                
        with col2:
            create_clickable_container(
                "AI Assistant",
                "Describe your farming needs in natural language and let our NLP system guide you to the right tool",
                "ai_btn",
                'chat'
            )

    elif st.session_state.page == 'manual':
        st.markdown("### Select a task")
        cols = st.columns([1, 1, 1], gap="small")
        
        tasks = [
            ("Soil Analysis Vision", "Upload soil images for AI-powered soil type identification and detailed recommendations", 'soil_btn', 'soil'),
            ("Crop Yield Forecast", "Enter your city and crop type to get precise yield predictions based on local conditions", 'crop_btn', 'crop'),
            ("Disease Detection AI", "Utilize Segma-Vision's advanced segmentation to identify and classify plant diseases", 'disease_btn', 'disease')
        ]
        
        for col, (title, desc, key, page) in zip(cols, tasks):
            with col:
                create_clickable_container(title, desc, key, page)
        
        if st.button("← Back", type='secondary'):
            st.session_state.page = 'home'
            st.rerun()

    elif st.session_state.page == 'chat':
        st.markdown("### AI Assistant")
        col1, col2 = st.columns([4, 1], gap="small")
        with col1:
            user_input = st.text_area("What would you like to do?", key="chat_input", label_visibility="collapsed")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Enter →", key="chat_submit", type="primary"):
                if user_input:
                    input_agent = InputAgent()
                    tasks = input_agent.process_query(user_input)
                    
                    if tasks:
                        for task in tasks:
                            if 'soil' in task['task'].lower():
                                st.session_state.page = 'soil'
                                st.rerun()
                            elif 'crop yield' in task['task'].lower():
                                st.session_state.page = 'crop'
                                st.rerun()
                            elif 'disease' in task['task'].lower():
                                st.session_state.page = 'disease'
                                st.rerun()
        
        if st.button("← Back", type='secondary'):
            st.session_state.page = 'home'
            st.rerun()
            
    elif st.session_state.page == 'soil':
        run_classification()
        if st.button("← Back", type='secondary'):
            st.session_state.page = 'manual'
            st.rerun()
            
    elif st.session_state.page == 'crop':
        run_prediction()
        if st.button("← Back", type='secondary'):
            st.session_state.page = 'manual'
            st.rerun()
            
    elif st.session_state.page == 'disease':
        st.markdown("### Plant Disease Detection")
        if st.button("← Back", type='secondary'):
            st.session_state.page = 'manual'
            st.rerun()

if __name__ == "__main__":
    main()