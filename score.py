from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the dataset (dummy data)
data = {
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "District": ["District A", "District B", "District A", "District C", "District B"],
    "Weather": ["Sunny", "Rainy", "Sunny", "Cloudy", "Rainy"],
    "Road_Type": ["Highway", "City Road", "Highway", "City Road", "Highway"],
    "Vehicle_Type": ["Car", "Bike", "Car", "Truck", "Car"],
    "Time": ["Morning", "Afternoon", "Evening", "Night", "Morning"],
    "Accident": [0, 1, 0, 1, 0]  # 0: No Accident, 1: Accident
}

df = pd.DataFrame(data)

# Encode categorical variables
df = pd.get_dummies(df, columns=["Gender", "District", "Weather", "Road_Type", "Vehicle_Type", "Time"])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=["Accident"])
y = df["Accident"]

# Get the feature names
feature_names = X.columns

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the entire dataset
clf.fit(X, y)

# Function to preprocess user input and make predictions
def predict_risk_score(user_input):
    # Preprocess user input and encode categorical variables
    user_df = pd.DataFrame(user_input, index=[0])
    # Ensure all possible categories are present
    for column in feature_names:
        if column not in user_df.columns:
            user_df[column] = 0
    user_df = pd.get_dummies(user_df)  # Re-encode to ensure consistent order
    
    # Remove any duplicate columns
    user_df = user_df.loc[:, ~user_df.columns.duplicated()]
    
    # Reorder columns to match training data
    user_df = user_df.reindex(columns=feature_names, fill_value=0)
    
    # Make prediction
    prediction_prob = clf.predict_proba(user_df)[0][1]  # Probability of positive class (accident)
    
    print("Predicted probability:", prediction_prob)  # Add this line for debugging
    
    # Assign risk score based on predicted probability
    if prediction_prob is None:
        risk_score = None
    elif prediction_prob < 0.2:
        risk_score = 0
    elif prediction_prob < 0.4:
        risk_score = 1
    elif prediction_prob < 0.6:
        risk_score = 2
    elif prediction_prob < 0.85:  # Adjusted threshold for risk score 4
        risk_score = 3
    else:
        risk_score = 4
    
    print("Final risk score:", risk_score)  # Add this line for debugging
    
    return risk_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.to_dict()
    risk_score = predict_risk_score(user_input)
    return render_template('result.html', risk_score=risk_score)

@app.route('/sourceToDestination')
def source_to_destination():
    return render_template('sourceToDestination.html')

@app.route('/redirect_to_mtb')
def redirect_to_mtb():
    return redirect(url_for('mtb'))

@app.route('/mtb')
def mtb():
    return render_template('Mtb.html')

if __name__ == '__main__':
    app.run(debug=True)
