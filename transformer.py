import pandas as pd
import numpy as np

class TitanicTransformer:
    def __init__(self, Vip_mode, home_planet_mode, desination_planent_mode, age_median , deck_mode , spend_median):
        self.Vip_mode = Vip_mode
        self.home_planet_mode = home_planet_mode
        self.desination_planent_mode = desination_planent_mode
        self.age_median = age_median
        self.deck_mode = deck_mode
        self.spend_median = spend_median

    def __call__(self, df):
        X = df.copy()
        X["VIP"] = X["VIP"].fillna(self.Vip_mode)
        X["HomePlanet"] = X["HomePlanet"].fillna(self.home_planet_mode)
        X["Destination"] = X["Destination"].fillna(self.desination_planent_mode)
        X["Age"] = X["Age"].fillna(self.age_median)
        X["VIP_Numeric"] = X.VIP.astype(int)
        X["Age_bracket"] = pd.cut(X['Age'], bins=[-np.inf, 12, 18, 60, np.inf], labels=['child','teen','adult','senior'])
        X["Total_spend"] = X[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
        X["CryoSleep"] = X["CryoSleep"].where(X["CryoSleep"].notna(), X["Total_spend"] == 0)
        X["CryoSleep_Numeric"] = X.CryoSleep.astype(int)  
        X["Spending_bracket"] = "No_spend"
        mask = X["Total_spend"] > 0
        X.loc[mask, "Spending_bracket"] = np.where(
            X.loc[mask, "Total_spend"] <= self.spend_median,
            "Low_spender", "High_spender"
        )
        X["Group"] = X["PassengerId"].str.split("_").str[0]
        group_counts = X["Group"].value_counts()
        X["Group_size"] = X["Group"].map(group_counts)
        X["Is_alone"] = (X["Group_size"] == 1).map({True: 1, False: 0})
        X['Deck'] = X['Cabin'].str.split('/').str[0]
        X["Deck"] = X["Deck"].fillna(self.deck_mode)
        X["Starboard_side"] = np.where(X["Cabin"].str.split("/").str[2] == "S", 1, 0)
        spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        X[spend_cols] = X[spend_cols].fillna(0)
        X = X.drop(columns=["PassengerId", "CryoSleep", "Cabin", "VIP", "Name"])
        return X