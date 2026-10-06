# IPL Match Winner Prediction

A machine learning web application that predicts the winning chances of two IPL teams based on historical match data and match-related information.

The project uses a **Random Forest Classifier** for prediction and **Flask** to provide a simple web interface where users can enter match details and view the predicted winner along with the winning probability of both teams.

## Features

- Select any two IPL teams
- Enter match details such as:
  - Toss winner
  - Toss decision
  - Venue
  - City
  - Season
  - Match type
- Predict the winner between the two selected teams
- Display winning probability for both teams
- Simple and responsive web interface
- Handles historical IPL team name changes
- Model trained using IPL match data

## Technologies Used

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- GridSearchCV
- One-Hot Encoding

### Web Development
- Flask
- HTML
- CSS
- JavaScript

### Other Tools
- Jupyter Notebook
- Joblib
- Git & GitHub

## Machine Learning Approach

The IPL dataset is first cleaned and preprocessed before training the model.

The following features are used for prediction:

- Team 1
- Team 2
- Toss Winner
- Toss Decision
- Venue
- City
- Season
- Match Type

The original match winner is converted into a binary target:

- `1` → Team 1 won
- `0` → Team 2 won

Categorical features are transformed using **One-Hot Encoding**.

A **Random Forest Classifier** is then trained on the processed data. `GridSearchCV` is used to test different model parameters and select the best configuration.

## Model Performance

The final Random Forest model achieved approximately:

**Test Accuracy: 53.91%**

The purpose of this project is to demonstrate the complete machine learning workflow from data preprocessing and model training to integrating the trained model with a web application.

IPL matches depend on many real-world factors that are not available in the dataset, so the predicted probabilities should be treated as model estimates rather than guaranteed outcomes.

## Project Structure

```text
IPL-Match-Winner-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── ipl.csv
│   ├── sample_data.csv
│   └── README.md
│
├── model/
│   ├── ipl_winner_model.joblib
│   ├── metadata.json
│   └── README.md
│
├── notebooks/
│   ├── IPL_match_winner_original.ipynb
│   └── IPL_match_winner_final.ipynb
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd IPL-Match-Winner-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

### 5. Run the Flask application

```bash
python app.py
```

### 6. Open the application

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Retraining the Model

The model can be trained again using:

```bash
python train_model.py
```

The trained model is saved inside the `model` directory and is loaded by the Flask application for making predictions.

## How It Works

1. The user selects two IPL teams.
2. Match information is entered through the web interface.
3. Flask sends the input data to the trained machine learning model.
4. The model calculates the probability of Team 1 and Team 2 winning.
5. The team with the higher predicted probability is displayed as the predicted winner.

## Disclaimer

This project is created for educational purposes. The predictions are based on historical IPL data and machine learning estimates. They should not be considered guaranteed predictions of actual IPL match results.
