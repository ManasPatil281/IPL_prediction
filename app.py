import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
from PIL import Image

# Set page configuration first
st.set_page_config(page_title='IPL Win Predictor', layout='centered')

# Function to load the model using st.cache_resource
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('pipe_new.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model file not found. Please upload the correct model file.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Function to validate inputs
def validate_inputs(target, score, overs, wickets):
    if target <= score:
        st.error("Target score should be greater than the current score.")
        return False
    if overs == 0 and score > 0:
        st.error("Current score can't be greater than zero if no overs have been bowled.")
        return False
    if wickets > 10:
        st.error("Wickets cannot exceed 10.")
        return False
    return True

# Load model
pipe = load_model()

# Add background image and other styling
st.markdown("""
    <style>
    body {
        background-color: #e0f7fa;
        font-family: 'Arial';
    }
    .stButton > button {
        background-color: #00796b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.image(Image.open('ipl-share-img.png'), use_column_width=True)

st.title('🏏 IPL Win Probability Predictor')

# Sidebar for user input
st.sidebar.header('Input Parameters')
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 
          'Sharjah', 'Mohali', 'Bengaluru']

batting_team = st.sidebar.selectbox('🏏 Select Batting Team', sorted(teams))
bowling_team = st.sidebar.selectbox('🎯 Select Bowling Team', sorted(teams))
selected_city = st.sidebar.selectbox('🌆 Select Host City', sorted(cities))

target = st.sidebar.number_input('🎯 Target Score', min_value=0)
score = st.sidebar.number_input('🏆 Current Score', min_value=0)
overs = st.sidebar.slider('⏱ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
wickets = st.sidebar.slider('🏏 Wickets Lost', min_value=0, max_value=10, step=1)

if st.sidebar.button('🔮 Predict Probability'):
    if validate_inputs(target, score, overs, wickets):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Prediction
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display Win Probabilities
        st.markdown(f"### **{batting_team} Win Probability: {round(win * 100, 2)}%**")
        st.markdown(f"### **{bowling_team} Win Probability: {round(loss * 100, 2)}%**")

        # Plotting the pie chart
        fig = px.pie(values=[win, loss], names=['Win', 'Loss'], 
                     color_discrete_sequence=px.colors.sequential.Blues,
                     title='Win/Loss Probability', hole=0.4)

        fig.update_layout(annotations=[dict(text=f'{batting_team}', x=0.5, y=0.5, 
                                            font_size=20, showarrow=False)])
        fig.update_traces(textinfo='percent+label')

        st.plotly_chart(fig)

# Customize the footer
st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    footer:after {
        content: '© 2024 IPL Win Predictor';
        visibility: visible;
        display: block;
        position: relative;
        padding: 10px;
        top: 2px;
        background-color: #00796b;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
