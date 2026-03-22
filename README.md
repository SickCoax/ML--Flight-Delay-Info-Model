# Flight Delay Info Model

### 🚧 Problem

Flight delays are unpredictable and can disrupt travel plans, causing inconvenience for passengers and operational challenges for airlines.

### 💡 Solution

This project builds a machine learning model that predicts flight delays in advance and estimates delay duration in minutes — even identifying early arrivals — helping travelers and systems make smarter decisions.

## Dataset

- Dataset used : Flight Delay Dataset

The dataset contains features such as:

- Airline
- Origin Airport
- Destination Airport
- Scheduled Departure Time
- Scheduled Arrival Time
- Distance
- Day of Week
- Month 
- Flight Number / ID

Target variables:

- **Delay** → Delayed or Not Delayed [For Classification] 
- **Arrival Delay (minutes)** → Positive (delay), Zero (on-time), Negative (early arrival)  [For Regression]

This dataset does not include a direct delay classification label. Hence, a new feature called "Delay" was engineered using existing attributes (such as "Arrival Delay"), where flights are categorized as delayed or not delayed.

### 🚨 Note : This project uses Flight Delay Dataset.

**Download it from** : [Flight Delay Dataset](https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv)

After downloading, place the files inside "dataset" folder.

## Task

- Handling missing values and data cleaning
- Feature engineering (creating new features/columns)
- Encoding categorical variables using Ordinal Encoding
- Handling imbalanced dataset
- Model training using Gradient Boosting (Regression & Classification)
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluation using metrics: MSE, MAE, Precision, Recall, F1-score

### Additional Task

- Built a modular pipeline with separate scripts for preprocessing, regression, and classification
- Developed a command-line interface (CLI) to interactively choose between delay prediction and delay classification
- Integrated end-to-end workflow: data loading → preprocessing → training → evaluation → prediction


## Library Used

- Pandas
- SciPy
- scikit-learn