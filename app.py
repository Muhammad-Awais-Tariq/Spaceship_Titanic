import streamlit as st
import joblib
import pandas as pd
from transformer import TitanicTransformer

st.set_page_config(
    page_title="Spaceship Titanic Predictor",
    page_icon="🚀",
    layout="centered",
)

model = joblib.load("spaceship_titanic_pipeline.joblib")

TRANSPORTED_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2x2anp5cGxvanc2aGRoemRlZ2N4M2IwaHdiYXVuNjk2NTFiYjEweSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/gEwrCXUgVijde/giphy.gif"
NOT_TRANSPORTED_GIF = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2x2anp5cGxvanc2aGRoemRlZ2N4M2IwaHdiYXVuNjk2NTFiYjEweSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/0SsFX6FdTmHf1ooJLE/giphy.gif"

st.title("Spaceship Titanic — Transport Predictor")
st.caption("Fill in a passenger's details below to predict whether they were transported to another dimension.")

st.divider()

with st.expander("How to format your entries (read me first!)", expanded=False):
    st.markdown("**Sample record, field by field:**")
    st.code("0032_02  Europa  True  D/0/S  55 Cancri e  23.0  False  0.0  0.0  0.0  0.0  0.0", language=None)

    st.markdown("Use this table as your reference for each field:")
    guide = pd.DataFrame(
        {
            "Field": [
                "PassengerId", "HomePlanet", "CryoSleep", "Cabin",
                "Destination", "Age", "VIP", "RoomService",
                "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name",
            ],
            "Expected format": [
                "gggg_pp  (4-digit group + 2-digit passenger number, e.g. 0032_02)",
                "One of: Europa, Earth, Mars",
                "True or False — was the passenger in suspended animation?",
                "deck/num/side, e.g. D/0/S  — deck is a letter, side is P (Port) or S (Starboard)",
                "One of: TRAPPIST-1e, 55 Cancri e, PSO J318.5-22",
                "Whole number, e.g. 23",
                "True or False",
                "Amount billed, e.g. 0 or 245.5 (0 if unused)",
                "Amount billed, e.g. 0 or 112 (0 if unused)",
                "Amount billed, e.g. 0 or 30 (0 if unused)",
                "Amount billed, e.g. 0 or 55 (0 if unused)",
                "Amount billed, e.g. 0 or 88 (0 if unused)",
                "Full name, e.g. John Doe",
            ],
        }
    )
    st.dataframe(guide, use_container_width=True, hide_index=True)
    st.info("Tip: leave a billing amount at 0 if the passenger never used that service.")

st.divider()

with st.form("passenger_form"):
    st.subheader("Passenger identity")
    col1, col2 = st.columns(2)
    with col1:
        PassengerId = st.text_input(
            "Passenger ID",
            placeholder="0032_02",
            help="Format: gggg_pp — 4-digit group number, underscore, 2-digit passenger number.",
        )
    with col2:
        Name = st.text_input("Full name", placeholder="John Doe")

    col3, col4 = st.columns(2)
    with col3:
        HomePlanet = st.selectbox("Home planet", ["Europa", "Earth", "Mars"])
    with col4:
        Destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])

    st.subheader("Travel details")
    col5, col6, col7 = st.columns(3)
    with col5:
        Cabin = st.text_input(
            "Cabin",
            placeholder="D/0/S",
            help="Format: deck/num/side — e.g. D/0/S. Side must be P (Port) or S (Starboard).",
        )
    with col6:
        Age = st.number_input("Age", min_value=0, max_value=120, value=23, step=1)
    with col7:
        CryoSleep = st.selectbox("Cryo sleep", [True, False])

    VIP = st.selectbox("VIP passenger", [True, False])

    st.subheader("Amenity billing")
    st.caption("Enter 0 for any service the passenger did not use.")
    col8, col9, col10, col11, col12 = st.columns(5)
    with col8:
        RoomService = st.number_input("Room service", min_value=0.0, value=0.0, step=1.0)
    with col9:
        FoodCourt = st.number_input("Food court", min_value=0.0, value=0.0, step=1.0)
    with col10:
        ShoppingMall = st.number_input("Shopping mall", min_value=0.0, value=0.0, step=1.0)
    with col11:
        Spa = st.number_input("Spa", min_value=0.0, value=0.0, step=1.0)
    with col12:
        VRDeck = st.number_input("VR deck", min_value=0.0, value=0.0, step=1.0)

    st.write("")
    predict_clicked = st.form_submit_button("Predict transport outcome", use_container_width=True)

if predict_clicked:
    missing = [
        label
        for label, value in [("Passenger ID", PassengerId), ("Cabin", Cabin), ("Name", Name)]
        if not value.strip()
    ]
    if missing:
        st.warning(f"Please fill in: {', '.join(missing)}")
    else:
        input_df = pd.DataFrame(
            [
                {
                    "PassengerId": PassengerId,
                    "HomePlanet": HomePlanet,
                    "CryoSleep": CryoSleep,
                    "Cabin": Cabin,
                    "Destination": Destination,
                    "Age": Age,
                    "VIP": VIP,
                    "RoomService": RoomService,
                    "FoodCourt": FoodCourt,
                    "ShoppingMall": ShoppingMall,
                    "Spa": Spa,
                    "VRDeck": VRDeck,
                    "Name": Name,
                }
            ]
        )

        with st.spinner("Consulting the ship's manifest..."):
            prediction = model.predict(input_df)[0]

        st.divider()

        if prediction == True:
            st.success("Transported to another dimension!")
            st.image(TRANSPORTED_GIF, use_container_width=True)
        else:
            st.error("Not transported.")
            st.image(NOT_TRANSPORTED_GIF, use_container_width=True)