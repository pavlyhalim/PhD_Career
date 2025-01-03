import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from bs4 import BeautifulSoup
import requests
import json
from cluster_model import PersonaModel
import random
import time

st.set_page_config(
    page_title="PhD Career Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_apis():
    """
    Load and cache the language model and search utility.
    """
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    search = GoogleSerperAPIWrapper(
        serpapi_api_key=st.secrets["SERPAPI_KEY"]
    )
    return llm, search

llm, search = initialize_apis()

def create_agent_with_tools(llm, search):
    job_search_tool = Tool(
        name="Job Search",
        description="Search for specific PhD-level positions and requirements",
        func=lambda q: search.run(f"PhD jobs {q} requirements salary")
    )
    
    market_tool = Tool(
        name="Market Research",
        description="Research industry trends for PhD careers",
        func=lambda q: search.run(f"industry trends PhD careers {q}")
    )
    
    return initialize_agent(
        [job_search_tool, market_tool],
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True  # <<---
    )


# ----------------------------------------
# JSON normalization helper
# ----------------------------------------
def normalize_response(response_text):
    """
    Parse the agent's response into a dictionary. 
    If the response isn't valid JSON, attempt to extract the JSON portion.
    Ensures a consistent structure by verifying expected keys.
    """
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError:
        # Attempt to extract JSON content
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            response_json = json.loads(response_text[start:end])
        except Exception as e:
            st.error(f"Error parsing response: {e}")
            return {}
    
    expected_keys = [
        'Top 5 specific roles matching persona and profile',
        'Required technical skills for these roles',
        'Targeted certifications based on persona type',
        'Professional organizations aligned with persona',
        'Conferences matching research background',
        'Companies hiring PhDs with this persona profile',
        'Location-specific salary ranges',
        'Career progression pathway for this persona',
        'Remote opportunities matching persona strengths',
        'International markets valuing this persona type'
    ]
    
    for key in expected_keys:
        response_json.setdefault(key, [])
        
    return response_json

# ----------------------------------------
# Core logic to generate career recommendations
# ----------------------------------------
def get_career_recommendations(agent, prediction_data, profile_data):
    """
    Builds a prompt from the user's profile and predicted persona,
    and then returns structured career recommendations.
    """
    persona_desc = prediction_data['Persona_Description'].iloc[0]
    confidence = prediction_data['Confidence_Score'].iloc[0]
    
    prompt = f"""
    Analyze career paths for a PhD with ML-predicted profile:
    Predicted Persona: {persona_desc}
    Model Confidence: {confidence:.2%}
    
    Profile Details:
    - Division: {profile_data['academic_division']}
    - Research: {profile_data['research_areas']}
    - Technical Skills: {profile_data['technical_skills']}
    - Soft Skills: {profile_data['soft_skills']}
    - Experience: {profile_data['years_experience']} years
    - Goals: {profile_data['career_goals']}
    - Locations: {', '.join(profile_data['preferred_locations'])}
    
    Based on the ML model's persona prediction and profile, determine:
    1. Top 5 specific roles matching persona and profile
    2. Required technical skills for these roles
    3. Targeted certifications based on persona type
    4. Professional organizations aligned with persona
    5. Conferences matching research background
    6. Companies hiring PhDs with this persona profile
    7. Location-specific salary ranges
    8. Career progression pathway for this persona
    9. Remote opportunities matching persona strengths
    10. International markets valuing this persona type
    
    Return as JSON with these categories as keys.
    """
    
    response = agent.run(prompt)
    recommendations = normalize_response(response)
    return recommendations

def display_persona_analysis(prediction, recommendations):
    """
    Display the user's predicted persona and top career insights.
    """
    st.header("Career Analysis Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your ML-Predicted Persona")
        st.info(f"**{prediction['Persona_Description'].iloc[0]}**")
        st.metric("Model Confidence", f"{prediction['Confidence_Score'].iloc[0]:.2%}")

    with col2:
        st.subheader("Top Career Paths")
        for role in recommendations.get('Top 5 specific roles matching persona and profile', []):
            st.write(f"- {role}")

        st.write("\n**Salary Ranges:**")
        salary_ranges = recommendations.get('Location-specific salary ranges', [])
        if isinstance(salary_ranges, list):
            for range_info in salary_ranges:
                if isinstance(range_info, dict):
                    location = range_info.get('location', 'Unknown location')
                    salary = range_info.get('salary_range', 'Not specified')
                    st.write(f"- **{location}**: {salary}")
                else:
                    st.write(f"- {range_info}")
        else:
            st.write(f"- {salary_ranges}")

    st.balloons()

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Career Progression")
        career_path = recommendations.get('Career progression pathway for this persona', [])
        if isinstance(career_path, list):
            for step in career_path:
                st.write(f"- {step}")
        else:
            st.write(career_path)

    with col4:
        st.subheader("Target Companies") 
        companies = recommendations.get('Companies hiring PhDs with this persona profile', [])
        if isinstance(companies, list):
            for company in companies:
                st.write(f"- {company}")
        else:
            st.write(companies)

def display_skills_development(recommendations):
    """
    Display the required technical skills, recommended certifications,
    relevant organizations, and conferences for the user's persona.
    """
    st.header("Skills & Development")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Required Technical Skills")
        skills = recommendations.get('Required technical skills for these roles', [])
        if isinstance(skills, list):
            for skill in skills:
                st.write(f"- {skill}")
        else:
            st.write(skills)
               
        st.subheader("Recommended Certifications")
        certs = recommendations.get('Targeted certifications based on persona type', [])
        if isinstance(certs, list):
            for cert in certs:
                st.write(f"- {cert}")
        else:
            st.write(certs)
    
    with col2:
        st.subheader("Professional Organizations")
        orgs = recommendations.get('Professional organizations aligned with persona', [])
        if isinstance(orgs, list):
            for org in orgs:
                st.write(f"- {org}")
        else:
            st.write(orgs)
           
        st.subheader("Relevant Conferences")
        conferences = recommendations.get('Conferences matching research background', [])
        if isinstance(conferences, list):
            for conf in conferences:
                st.write(f"- {conf}")
        else:
            st.write(conferences)

def display_global_opportunities(recommendations):
    """
    Display remote and international opportunities based on the user's persona.
    """
    st.header("Global Opportunities")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Remote Opportunities")
        remote_opps = recommendations.get('Remote opportunities matching persona strengths', [])
        if isinstance(remote_opps, list):
            for opp in remote_opps:
                st.write(f"- {opp}")
        else:
            st.write(remote_opps)
    
    with col2:
        st.subheader("International Markets")
        intl_markets = recommendations.get('International markets valuing this persona type', [])
        if isinstance(intl_markets, list):
            for market in intl_markets:
                st.write(f"- {market}")
        else:
            st.write(intl_markets)

# ----------------------------------------
# Form to collect user profile
# ----------------------------------------
def collect_user_profile():
    """
    Render a form in Streamlit to collect the user's academic and personal info.
    Returns a dictionary of the collected inputs.
    """
    with st.form("user_profile"):
        st.write("### Please fill in your details below to get tailored career advice:")
        col1, col2 = st.columns(2)
        
        with col1:
            academic_division = st.selectbox(
                "Academic Division",
                ["Humanities", "STEM", "Social Sciences", "Professional fields"],
                help="Select the broad academic division of your PhD."
            )
            specialization = st.text_input(
                "Specific Field/Specialization",
                placeholder="e.g., Computational Linguistics, Nanotechnology, etc."
            )
            research_areas = st.text_area(
                "Research Areas",
                placeholder="List relevant research areas (e.g., Machine Learning, Public Policy)..."
            )
            years_experience = st.number_input(
                "Years of Research Experience",
                min_value=0,
                max_value=50,
                step=1,
                help="Your approximate total years of research experience."
            )

        with col2:
            technical_skills = st.text_area(
                "Technical Skills",
                placeholder="List your most important technical skills..."
            )
            soft_skills = st.text_area(
                "Soft Skills",
                placeholder="List your most important soft skills (e.g., communication, leadership, teamwork)..."
            )
            preferred_locations = st.multiselect(
                "Preferred Job Locations",
                ["Remote", "USA", "Europe", "Asia", "Australia", "Other"],
                help="Select all regions you are interested in working in."
            )
            career_goals = st.text_area(
                "Career Goals",
                placeholder="Share your long-term or short-term career aspirations..."
            )
            
        submitted = st.form_submit_button("Analyze Career Options")
        
        return {
            "academic_division": academic_division,
            "specialization": specialization,
            "research_areas": research_areas,
            "years_experience": years_experience,
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "preferred_locations": preferred_locations,
            "career_goals": career_goals,
            "submitted": submitted
        }

# ----------------------------------------
# Main function controlling the app flow
# ----------------------------------------
def main():
    # App Title
    st.title("PhD Career Advisor")
    st.markdown(
        """
        Welcome to the **PhD Career Advisor**! This tool uses a robust ML model 
        to predict your 'persona' and provide curated career recommendations 
        based on your research background, skills, and interests.

        Explore new advanced features, including a **Chat With Your Career Counselor** 
        """
    )
    
    tabs = st.tabs([
        "Profile Input", 
        "Career Analysis", 
        "Global Opportunities", 
        "Skills Development",
        "Chat With Counselor"
    ])
    
    # Tab 1: Collect user input
    with tabs[0]:
        profile_data = collect_user_profile()
        
        # Only proceed if the user submitted the form
        if profile_data["submitted"]:
            with st.spinner("Loading ML model..."):
                model = PersonaModel.load_model()
                
                # Prepare input data for model prediction
                input_data = pd.DataFrame({
                    'Academic_Division': [profile_data["academic_division"]],
                    'Years since graduation': [profile_data["years_experience"]],
                    'Level of responsibility': [3],
                    'Analytical thinking': [3],
                    'Leadership': [3],
                    'Innovation': [3],
                    'sector': ['Unknown'],
                    'Job location': ['Unknown'],
                    'Change in career': ['No'],
                    'Degree of independence': [3],
                    'Persistence': [3],
                    'Resilience': [3],
                    'Rigor': [3],
                    'Risk-taking': [3],
                    'Self-control': [3],
                    'Social orientation': [3],
                    'Stress tolerance': [3],
                    'Integrity': [3],
                    'Initiative': [3],
                    'Independence': [3]
                })
                
                # Run prediction
                prediction = model.predict(input_data)
                st.success("Your persona model prediction is complete!")
            
            with st.spinner("Generating career recommendations..."):
                agent = create_agent_with_tools(llm, search)
                recommendations = get_career_recommendations(agent, prediction, profile_data)
            
            # Store data in session state
            st.session_state.update({
                'prediction': prediction,
                'recommendations': recommendations,
                'profile_data': profile_data,
                'agent': agent
            })
            st.success("Analysis complete! Please move to the next tabs for more insights.")
    
    # Tabs 2-4: Display results if they're available
    if (
        'prediction' in st.session_state 
        and 'recommendations' in st.session_state 
        and 'profile_data' in st.session_state
    ):
        with tabs[1]:
            display_persona_analysis(
                st.session_state['prediction'],
                st.session_state['recommendations']
            )
        
        with tabs[2]:
            display_global_opportunities(st.session_state['recommendations'])
            
        with tabs[3]:
            display_skills_development(st.session_state['recommendations'])
        
        # Tab 4: Chat With Counselor
        with tabs[4]:
            st.write("### Real-Time Career Counselor")
            st.write("Ask follow-up questions about your persona or career paths below:")
            
            # We’ll store chat messages in a session state list
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            
            # Display the conversation so far
            for msg in st.session_state['chat_history']:
                if msg['role'] == 'user':
                    st.write(f"**You:** {msg['content']}")
                else:
                    st.write(f"**Counselor:** {msg['content']}")
            
            # If your version of Streamlit doesn't have st.chat_input, you can replace it with st.text_input.
            user_input = st.chat_input("Type your question...") if hasattr(st, 'chat_input') else st.text_input("Type your question...")
            
            if user_input:
                st.session_state['chat_history'].append({'role': 'user', 'content': user_input})
                
                # We pass the conversation to the LLM agent
                with st.spinner("Counselor is thinking..."):
                    answer = st.session_state['agent'].run(
                        f"{user_input} (User's Persona: {st.session_state['prediction']['Persona_Description'].iloc[0]})"
                    )
                st.session_state['chat_history'].append({'role': 'assistant', 'content': answer})
                
                st.experimental_rerun()
    
    else:
        st.warning("Please complete the 'Profile Input' section first.")
    
    st.markdown("---")
    st.write("**NYU GSAS!**")


if __name__ == "__main__":
    main()
