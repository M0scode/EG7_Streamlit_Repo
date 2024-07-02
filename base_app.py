"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Importing necessary libraries
import streamlit as st
import joblib
import os
import pandas as pd

# Load your vectorizer from the pkl file
vectorizer_path = r"C:\Users\thabi\Downloads\streamlit\EG7_Streamlit_Repo\models\tfidfvect.pkl"
with open(vectorizer_path, "rb") as vec_file:
    news_vectorizer = joblib.load(vec_file)

# Load your raw data
data_path = r"C:\Users\thabi\Downloads\streamlit\EG7_Streamlit_Repo\train.csv"
raw_data = pd.read_csv(data_path)

def main():
    """News Classifier App with Streamlit"""

    # Main title and subheader with logo
    st.markdown("""
        <style>
            .container {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .main-title {
                font-size: 3em; 
                color: #4CAF50; 
                font-weight: bold; 
                text-align: center;
                margin-bottom: 0;
            }
            .subheader {
                font-size: 1.5em; 
                color: #009688; 
                text-align: center;
                margin-top: 0;
            }
        </style>
        <div class="container">
            <div>
                <h1 class="main-title">News Classifier</h1>
                <h2 class="subheader">Your news articles classified!</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    options = ["Meet the Team", "All About the App", "EDA", "Model Selection", "Prediction", "Information", "Conclusion"]
    selection = st.sidebar.selectbox("Choose Option", options)
   

    # "Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown("Some information here")

    # "Prediction" page
    elif selection == "Prediction":
        st.info("Prediction with ML Models")
        news_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            vect_text = news_vectorizer.transform([news_text]).toarray()
            model_path = r"C:\Users\thabi\Downloads\streamlit\EG7_Streamlit_Repo\models\Logistic_regression.pkl"
            with open(model_path, "rb") as model_file:
                predictor = joblib.load(model_file)
            prediction = predictor.predict(vect_text)
            st.success(f"Text Categorized as: {prediction[0]}")

    # "Meet the Team" page
    elif selection == "Meet the Team":
        st.markdown("""
            <style>
                .team-title {
                    font-size: 2.5em;
                    font-weight: bold;
                    color: #2196F3;
                    text-align: center;
                    margin-top: 20px;
                    margin-bottom: 20px;
                }
                .member-container {
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .member-info {
                    margin-left: 20px;
                    font-size: 1.2em;
                }
                .member-name {
                    font-size: 1.5em;
                    font-weight: bold;
                }
                .member-role {
                    font-style: italic;
                }
            </style>
            <div class="team-title">Meet the Team</div>
        """, unsafe_allow_html=True)
        team_members = [
            {"name": "Veronicah Sihlangu", "role": "Team Leader"},
            {"name": "Rofhiwa Ramphele", "role": "Project Manager"},
            {"name": "Sandiso Magwaza", "role": "Github Manager"},
            {"name": "Nomfundo Sithole", "role": "Lead Data Scientist"},
            {"name": "Thabisile Xaba", "role": "Machine Learning Specialist"},
            {"name": "Keneilwe Madihlaba", "role": "Frontend Developer"},
        ]
        for member in team_members:
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown(f'<h3 style="color:#2196F3;">{member["name"]}</h3>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:1.1em;font-style:italic;">{member["role"]}</p>', unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)

    # "All About the App" page
    elif selection == "All About the App":
        st.markdown("""
             <style>
                .header-title { 
                    color: #4CAF50; 
                    font-size: 28px; 
                    font-weight: bold;
                }
                .sub-header { 
                    color: #009688; 
                    font-size: 24px; 
                    font-weight: bold;
                }
                .highlight { 
                    background-color: #f0f0f0; 
                    border-left: 6px solid #2196F3; 
                    padding: 15px; 
                    margin: 15px 0; 
                    font-size: 18px;
                }
                ul {
                    list-style-type: none;
                    padding: 0;
                    font-size: 18px;
                }
                li {
                    padding: 8px 0;
                }
                .emoji {
                    font-size: 1.5em;
                }
            </style>
            <div class="header-title">Welcome to the News Classifier App! 🌟</div>
            <div class="highlight">
                This application is designed to classify news articles into different categories. The categories include:
                <ul>
                    <li>📈 <b>Business</b></li>
                    <li>💻 <b>Technology</b></li>
                    <li>⚽ <b>Sports</b></li>
                    <li>🎬 <b>Entertainment</b></li>
                    <li>🎓 <b>Education</b></li>
                </ul>
            </div>
            <div class="sub-header">How to Use the App</div>
            <p>Simply type in the text of a news article in the provided text box, and the app will analyze the text and predict which category it belongs to.</p>
            
            <div class="sub-header">Features</div>
            <ul>
                <li>🌟 <b>User-Friendly Interface</b>: Easy to navigate and use.</li>
                <li>🔍 <b>Multiple Categories</b>: Supports classification into business, technology, sports, entertainment, and education.</li>
                <li>⚡ <b>Real-Time Prediction</b>: Instant results upon text input.</li>
            </ul>
            
            <div class="sub-header">Behind the Scenes</div>
            <p>The app utilizes advanced machine learning models to analyze the text and make predictions. It leverages natural language processing techniques to understand the context and content of the text, ensuring accurate classification.</p>
            
            <div class="sub-header">Try It Out</div>
            <p>Head over to the Prediction page, enter your text, and see the magic happen!</p>
            
            <div class="sub-header">We hope you enjoy using the News Classifier App!</div>
        """, unsafe_allow_html=True)


    # Placeholder for other sections
    elif selection == "EDA":
        st.info("Exploratory Data Analysis")

    elif selection == "Model Selection":
        st.info("Model Selection")

    elif selection == "Conclusion":
        st.info("Conclusion")

# Required to let Streamlit instantiate the app
if __name__ == '__main__':
    main()
