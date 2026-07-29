# Spaceship Titanic Survival Prediction

An end-to-end machine learning project predicting passenger transportation on the Spaceship Titanic, built with a fully custom feature engineering pipeline, compared across four different models, and deployed as an interactive Streamlit web app.

## Live Demo

Try it here: [https://spaceshiptitanic7.streamlit.app/](https://spaceshiptitanic7.streamlit.app/)

Enter a passenger's details (home planet, cabin, spending, age, group size, etc.) and get a live prediction of whether they were transported to another dimension.

---

## Project Structure

```
Spaceship_Titanic/
│
├── Data/                              # Raw datasets (gitignored — not pushed to GitHub)
│   ├── train (3).csv                  # Original Kaggle training data
│   └── test (3).csv                   # Original Kaggle test data
│
├── Exploration/
│   ├── Reports/
│   │   └── raw_report.html            # Exploratory data profiling report (sweetviz)
│   └── Exploration.ipynb              # Notebook: EDA, feature engineering iteration,
│                                       # model comparison, hyperparameter tuning
│
├── Predictions/
│   ├── prediction_with_75%_data.csv   # Kaggle submission — model trained on the 70/30 split
│   └── prediction_with_100%_data.csv  # Kaggle submission — final model trained on full data
│
├── final_pipeline.py                  # Clean, final training script — builds the
│                                       # feature-engineered pipeline, trains the
│                                       # final model, and saves it with joblib
├── transformer.py                     # Custom TitanicTransformer class used inside
│                                       # the sklearn pipeline (feature engineering
│                                       # logic, picklable for deployment)
├── app.py                             # Streamlit app — loads the saved model and
│                                       # serves the interactive prediction UI
├── spaceship_titanic_pipeline.joblib  # Serialized final trained pipeline (model +
│                                       # preprocessing + feature engineering, all in one)
│
├── pyproject.toml                     # Project dependencies (for uv)
├── uv.lock                            # Locked dependency versions
├── .python-version                    # Python version pin
├── .gitignore                         # Excludes Data/ folder from version control
└── README.md                          # This file
```

> **Note on data:** The `Data/` folder (raw CSVs, predictions) is excluded via `.gitignore` and is **not** pushed to GitHub. To run this project locally, download `train.csv` and `test.csv` from the [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic/data) and place them in a local `Data/` folder.

---

## What Each File Does

### `Exploration/Exploration.ipynb`
The working notebook where all the experimentation happened:
- Exploratory data analysis on the raw Spaceship Titanic dataset (with a sweetviz profiling report)
- Iterative feature engineering (group size, spending brackets, deck extraction, cabin side)
- Building and debugging the sklearn `Pipeline` + `ColumnTransformer`
- Training and cross-validating 4 different models
- Hyperparameter tuning with `RandomizedSearchCV` for each model
- Comparing cross-validation scores against held-out test scores across all models

### `transformer.py`
Contains the `TitanicTransformer` class — a callable class (not a plain function) so it can be pickled and reloaded outside the training script. It encapsulates all feature engineering logic and holds the **fixed statistics computed only from the training set** (VIP mode, home planet mode, destination mode, age median, deck mode, spend median), so the exact same transformation is applied consistently to training data, test data, and any new single-passenger input at prediction time — with no data leakage.

### `final_pipeline.py`
The clean, production version of the pipeline:
1. Loads `train.csv`
2. Computes fixed imputation/binning statistics from the training data
3. Builds the full `Pipeline`: feature engineering → preprocessing (scaling, one-hot, ordinal encoding) → model
4. Fits the final chosen model (tuned XGBoost) on the full training set
5. Serializes the trained pipeline with `joblib` for deployment

### `app.py`
The Streamlit web app. Loads `spaceship_titanic_pipeline.joblib`, presents a form for entering passenger details, and returns a live prediction.

---

## Feature Engineering

Beyond the raw Spaceship Titanic columns, the following engineered features were built into the pipeline:

| Feature | Description |
|---|---|
| `VIP_Numeric` | `VIP` cast to numeric, missing values filled with the training mode |
| `Age_bracket` | `Age` binned into `child` / `teen` / `adult` / `senior` |
| `Total_spend` | Sum of `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |
| `CryoSleep_Numeric` | `CryoSleep` imputed using `Total_spend == 0` where missing, then cast to numeric |
| `Spending_bracket` | `Total_spend` split into `No_spend` / `Low_spender` / `High_spender` |
| `Group` / `Group_size` | Extracted from the group prefix of `PassengerId` |
| `Is_alone` | Binary flag for passengers traveling with no group aboard |
| `Deck` | Extracted from the first segment of `Cabin` |
| `Starboard_side` | Binary flag for whether the cabin's side segment is `S` |

**Preprocessing:**
- **Numeric columns** (`Age`, `VIP_Numeric`, `CryoSleep_Numeric`, `Total_spend`, `Group_size`, `Is_alone`, `Starboard_side`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) → `StandardScaler`
- **Nominal categorical** (`HomePlanet`, `Destination`, `Deck`) → `OneHotEncoder`
- **Ordinal categorical** (`Age_bracket`, `Spending_bracket`) → `OrdinalEncoder` with explicitly defined category order

All missing-value imputation (VIP, home planet, destination, age, deck) is computed **once from the training set** and reused at inference time — avoiding leakage and ensuring the pipeline works correctly on a single new passenger, not just batches.

---

## Models Compared

Four models were trained through the identical feature engineering + preprocessing pipeline on a 70/30 train-test split, each tuned with `RandomizedSearchCV`:

| Model | CV Accuracy | Held-out Test Accuracy |
|---|---|---|
| Logistic Regression | ~0.796 | ~0.782 |
| Decision Tree | ~0.786 | ~0.776 |
| Random Forest | ~0.805 | ~0.789 |
| **XGBoost (final)** | **~0.807** | **~0.795** |

**Winner: XGBoost** — best cross-validation and held-out test performance among the four models, and selected as the final deployed model.

**Final XGBoost hyperparameters** (found via `RandomizedSearchCV`):
```python
XGBClassifier(
    subsample=0.6,
    n_estimators=100,
    min_child_weight=5,
    max_depth=5,
    learning_rate=0.05,
    gamma=1,
    colsample_bytree=0.8
)
```

---

## Submission Iterations

The model went through two training stages, reflected in the `Predictions/` folder:

| Version | Training Data | CV Accuracy | Accuracy (no CV) |
|---|---|---|---|
| `prediction_with_75%_data.csv` | 70/30 train-test split | ~0.807 | ~0.795 (held-out test set) |
| `prediction_with_100%_data.csv` (final, deployed) | Full training set | ~0.804 | ~0.782 (Kaggle public leaderboard) |

For the final version, the model is trained on 100% of the available training data, so there is no local held-out set left to score against — its true generalization accuracy is the score reported by Kaggle after submission, which came out to about 0.78, essentially matching the training-time cross-validation accuracy.

**Kaggle competition:** [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

---

## How to Run Locally

### Prerequisites
- Python (see `.python-version`)
- `uv` package manager (or `pip`)

### Setup
```bash
git clone <repository-url>
cd Spaceship_Titanic

# Download train.csv and test.csv from Kaggle's Spaceship Titanic competition
# and place them inside a local Data/ folder (not included in this repo)

uv sync
```

### Train the model
```bash
uv run python final_pipeline.py
```
This regenerates `spaceship_titanic_pipeline.joblib`.

### Run the Streamlit app
```bash
uv run streamlit run app.py
```
Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Technologies Used

- [scikit-learn](https://scikit-learn.org/) — pipelines, preprocessing, models, cross-validation, hyperparameter search
- [XGBoost](https://xgboost.readthedocs.io/) — gradient-boosted tree model
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data manipulation
- [Sweetviz](https://github.com/fbdesignpro/sweetviz) — exploratory data profiling
- [Streamlit](https://streamlit.io/) — interactive web app deployment
- [joblib](https://joblib.readthedocs.io/) — model serialization

---

## Key Learnings from This Project

- Building a leak-free sklearn `Pipeline`/`ColumnTransformer` from raw data to model, including custom feature engineering via a picklable transformer class
- Correctly distinguishing between cross-validation scores, held-out test scores, and true leaderboard performance
- Comparing linear, tree-based, and boosted models under an identical preprocessing pipeline
- Hyperparameter tuning with `RandomizedSearchCV` across four different model families
- Serializing a full pipeline (feature engineering + preprocessing + model) for deployment, including handling the "closures aren't picklable" limitation with a custom class
- Deploying a trained pipeline behind a live Streamlit interface

---

## Author

Muhammad Awais Tariq

## References

- [Kaggle Spaceship Titanic Competition](https://www.kaggle.com/competitions/spaceship-titanic)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

If you found this project useful, consider giving it a star.