import streamlit as st
import joblib
from transformer import TitanicTransformer
import pandas as pd

model = joblib.load("spaceship_titanic_pipeline.joblib")

PassengerId = st.text_input("Enter the id: ")
HomePlanet = st.text_input("HomePlanet: ")
CryoSleep = st.selectbox("sleep" , [True , False])
Cabin = st.text_input("Enter the cabin: ")
Destination = st.text_input("destination: ")
Age = st.number_input("age")
VIP = st.selectbox("vip" , [True , False])
RoomService = st.number_input("paid: ")
FoodCourt = st.number_input("paid1: ")
ShoppingMall = st.number_input("paid2: ")
Spa = st.number_input("paid3: ")
VRDeck = st.number_input("paid4: ")
Name = st.text_input("name: ")

predict_clicked = st.button("Predict transported", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame([{
        "PassengerId": PassengerId, 
        "HomePlanet": HomePlanet,
        "CryoSleep": CryoSleep ,
        "Cabin": Cabin,
        "Destination": Destination,
        "Age": Age,
        "VIP": VIP,
        "RoomService": RoomService,
        "FoodCourt": FoodCourt,
        "ShoppingMall" : ShoppingMall,
        "Spa" : Spa,
        "VRDeck" : VRDeck,
        "Name" : Name
    }])

    prediction = model.predict(input_df)[0]

    if prediction == True:
        st.success("Transported!")
    else:
        st.error("not")   