import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd


with open("C:\\Users\\User\\OneDrive - Ashesi University\\Intro to AIRandomForestRegressor.pkl",'rb') as file:
    saved_model = pickle.load(file)

st.set_page_config(page_title = 'My Webpage',page_icon='😄',layout='wide')

def process_input(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

#Header Section
with st.container():
    st.title('Predict the Overall Rating of Your Favorite Players 🤩')
    st.write('On this website you can enter some attributes of your favorite player and using AI we will predict the overall rating of the player') 
    
with st.container():
    st.write('---')
    st.header("Prediction")
    st.write('##')
    st.write('Enter the following attributes')

with st.form("player_form"):
    potential = st.number_input("potential",min_value=0,max_value=100)
    value_eur = st.number_input("value _eur",min_value=0)
    wage_eur = st.number_input("wage _eur",min_value=0)
    passing = st.number_input("passing",min_value=0,max_value=100)
    movement_reactions = st.number_input("movement_reactions",min_value=0,max_value=100)
    mentality_composure = st.number_input("mentality_composure",min_value=0,max_value=100)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    player_attributes = {'potential':potential,"value_eur":value_eur,"wage_eur":wage_eur,'passing':passing,'movement_reactions':movement_reactions,'mentality_composure':mentality_composure}
    
    df = pd.DataFrame(player_attributes, index=[0])
    processed_df = process_input(df)
    
    prediction = saved_model.predict(processed_df)
    
    st.write("Predicted rating is:",prediction)
    