import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler , FunctionTransformer , OrdinalEncoder , OneHotEncoder 
from sklearn.compose import ColumnTransformer
from xgboost                 import XGBClassifier

spaceship_df = pd.read_csv("F://Spaceship_Titanic//Data//train (3).csv")
X = spaceship_df.drop(columns=["Transported"])
y = spaceship_df["Transported"]

Vip_mode = X["VIP"].mode()[0]
home_planet_mode = X["HomePlanet"].mode()[0]
desination_planent_mode = X["Destination"].mode()[0]
age_median = X["Age"].median()


numeric_coloumns = ["Age" , "VIP_Numeric" , "CryoSleep_Numeric" , "Total_spend" , "Group_size" , "Is_alone" , "Starboard_side" , "RoomService" , "FoodCourt" , "ShoppingMall" , "Spa" , "VRDeck" ]
categorical_coloumns_one_hot = ["HomePlanet" , "Destination" , "Deck"]
categorical_coloumns_ordinal = [ "Age_bracket" , "Spending_bracket"]


numeric_pipeline = Pipeline(
    [
        ("Scaler" , StandardScaler()),
    ]
)


categorical_ohe_pipeline = Pipeline(
    [
        ("Encoder" , OneHotEncoder(handle_unknown="ignore"))
    ]
)


categorical_ord_pipeline = Pipeline(
    [
        ("Encoder" , OrdinalEncoder(
            categories=[
                ['child','teen','adult','senior'],
                ["No_spend" , "Low_spender", "High_spender"]
            ]
        ))
    ]
)


preprocessor = ColumnTransformer(
    [
        ("num" , numeric_pipeline , numeric_coloumns),
        ("categorical_ohe" , categorical_ohe_pipeline , categorical_coloumns_one_hot),
        ("categorical_ord" , categorical_ord_pipeline , categorical_coloumns_ordinal)

    ]
)


boosted_forest_pipeline = Pipeline(
    [
        ("Feature_engineering" , FunctionTransformer(transform)),
        ("Preprocessing" , preprocessor),
        ("Model" , XGBClassifier(subsample = 0.6, n_estimators = 100, min_child_weight = 5, max_depth = 5, learning_rate = 0.05, gamma = 1, colsample_bytree = 0.8
))
    ]
)