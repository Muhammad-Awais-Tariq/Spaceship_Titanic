import streamlit as st
import joblib
from transformer import TitanicTransformer
import pandas as pd

model = joblib.load("spaceship_titanic_pipeline.joblib")

