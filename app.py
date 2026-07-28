import streamlit as st
import joblib
from transformer import TitanicTransformer
import pandas as pd

model = joblib.load("spaceship_titanic_pipeline.joblib")

PassengerId = st.text_input("Enter the id: ")
HomePlanet = st.text_input("HomePlanet: ")
CryoSleep = st.selectbox([True , False])
Cabin = st.text_input("Enter the cabin: ")
Destination = st.text_input("destination: ")
Age = st.number_input("age")
VIP = st.selectbox([True , False])
RoomService = st.number_input("paid: ")
FoodCourt = st.number_input("paid: ")
ShoppingMall = st.number_input("paid: ")
Spa = st.number_input("paid: ")
VRDeck = st.number_input("paid: ")
Name = st.text_input("name: ")