
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from agents.agent import strenght_weakness_agent, potential_vs_rating

# Load the trained model
model = joblib.load('best_model.pkl')

# Set up page configuration
st.set_page_config(page_title="Soccer Player Potential Predictor", page_icon="⚽", layout="wide")

# Custom CSS for theming
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
        font-family: Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #004d66;
        color: white;
    }
    .stButton>button {
        background-color: #004d66;
        color: white;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00334d;
        color: white;
    }
    .stSlider>div>div {
        color: #004d66;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #004d66;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation with radio buttons for better visual appeal
st.sidebar.title("Navigation")
st.sidebar.markdown("### Go to:")
page = st.sidebar.radio(
    "",
    ["Introduction", "Data Exploration", "Predictions", "Feature Importance"]
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Introduction"

# Set the current page based on user selection
st.session_state.page = page

# Introduction Page
if st.session_state.page == "Introduction":
    st.title("Welcome to the Soccer Player Potential Predictor App")
    st.write("""
    #### This app is designed to assist football scouts in evaluating and predicting the future potential of soccer players.
    By leveraging machine learning models and advanced data analytics, scouts can make informed decisions about player development and recruitment.

    **Key Features:**
    - **Data Exploration:** Upload player data to explore various attributes and statistics.
    - **Potential Prediction:** Predict the future potential of players based on their current attributes.
    - **Strengths and Weaknesses Analysis:** Get detailed insights into the strengths and weaknesses of players.
    - **Feature Importance:** Understand the importance of different attributes in predicting player potential.

    **How to Use:**
    - Navigate through different sections using the sidebar.
    - Explore player data in the Data Exploration section to visualize and analyze attributes.
    - Input player attributes manually in the Predictions section to get potential predictions.
    - View the importance of different features in the Feature Importance section.

    **Why Use This App?**
    This tool provides football scouts with:
    - Objective insights into player potential.
    - Data-driven analysis for better decision-making.
    - Enhanced ability to identify promising talent.


    **Get Started:**
    Use the sidebar to navigate to the different sections of the app and start exploring player data and predictions.
    """)

# Data Exploration Page
elif st.session_state.page == "Data Exploration":
    st.title("Data Exploration")
    # File uploader for data exploration
    uploaded_file = 'unique_player_data.csv'
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Sidebar filters
        st.sidebar.header("Filters")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        feature = st.sidebar.selectbox("Feature to analyze", numeric_columns)
        min_rating = st.sidebar.slider("Minimum Potential", int(df['potential'].min()), int(df['potential'].max()), int(df['potential'].min()))
        max_rating = st.sidebar.slider("Maximum Potential", int(df['potential'].min()), int(df['potential'].max()), int(df['potential'].max()))

        # Apply filters to the dataframe
        filtered_df = df[(df['potential'] >= min_rating) & (df['potential'] <= max_rating)]

        # Display filtered data
        st.write(f"Data filtered by Potential between {min_rating} and {max_rating}")
        st.write(filtered_df.head())

        # Interactive plots
        st.write(f"Analysis of {feature}")

        # Histogram
        st.write("Histogram")
        fig = px.histogram(filtered_df, x=feature)
        st.plotly_chart(fig)

        # Box Plot
        st.write("Box Plot")
        fig = px.box(filtered_df, y=feature)
        st.plotly_chart(fig)

        # Scatter Plot
        st.write("Scatter Plot")
        scatter_x = st.selectbox("X-axis", numeric_columns, index=0)
        scatter_y = st.selectbox("Y-axis", numeric_columns, index=1)
        fig = px.scatter(filtered_df, x=scatter_x, y=scatter_y)
        st.plotly_chart(fig)

        # Add a dropdown with Player names
        player_names = df['player_name'].unique()
        player_name = st.selectbox("Select Player", player_names, index=0)
        player_data = df[df['player_name'] == player_name]
        st.write("Player Data:")
        st.write(player_data)
        st.write(strenght_weakness_agent(player_data))

# Predictions Page
elif st.session_state.page == "Predictions":
    st.title("Predictions")
    st.header("Input Player Attributes Manually")
    st.write("The generalization score of the model is RSME = 3.42")

    # Input attributes for prediction. Attributes are the top 10 features
    importances = model.named_steps['model'].feature_names_in_
    attributes = {}
    for f in importances:
        attributes[f] = st.slider(f, value=50, min_value=0, max_value=100, step=1)
    

    # Predict button for manual input
    if st.button("Predict Potential"):
        input_data = pd.DataFrame([attributes])
        input_data.columns = [col.split('__')[-1] for col in input_data.columns.to_list()]
        prediction = model.predict(input_data)[0]
        rounded_prediction = round(prediction)
        st.subheader(f"Predicted Potential: {rounded_prediction}")
        st.write(potential_vs_rating({'overall_rating': input_data['overall_rating'][0], 'potential': rounded_prediction}))


# Feature Importance Page
elif st.session_state.page == "Feature Importance":
    st.title("Feature Importance")
    # Assuming you have a feature importance array or similar from your model
    importances = model.named_steps['model'].feature_importances_
    feature_names = model.named_steps['model'].feature_names_in_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df['feature'] = feature_importance_df['feature'].str.split('__').str[-1]
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=True) 
     
    # Plot feature importances
    st.write("Feature Importance Plot")
    fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
