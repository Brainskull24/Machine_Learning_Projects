from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import os

# Load the dataset
df = pd.read_csv('data.csv')

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare data
X = df_imputed.drop(['MEDV','ZN','CHAS'], axis=1)
y = df_imputed['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Define a function to predict house price
def predict_price(attributes, model):
    return model.predict([attributes])[0]

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    rm = float(data['RM'])
    indus = float(data['INDUS'])
    # Add more attributes as needed

    # Prepare the attributes for prediction
    attributes = [rm, indus]
    # Add more attributes as needed

    # Predict house price using Linear Regression model
    price_lr = predict_price(attributes, model_lr)

    # Predict house price using Random Forest model
    price_rf = predict_price(attributes, model_rf)

    return jsonify({'price_lr': price_lr, 'price_rf': price_rf})

# Define route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
