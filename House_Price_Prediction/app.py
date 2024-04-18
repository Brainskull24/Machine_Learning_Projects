from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from forms import HousePriceForm

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = HousePriceForm(request.form)
    predicted_price = None

    if request.method == 'POST' and form.validate():
        try:
            # Extract form data
            data = [form.crime_rate.data, form.num_rooms.data, form.indus.data,
                    form.house_age.data, form.distance.data, form.tax_rate.data]

            # Preprocess data and make prediction
            predicted_price = predict_price(data)
        except Exception as e:
            print("An error occurred:", str(e))
            return "An error occurred while processing your request."

    return render_template('index.html', form=form, predicted_price=predicted_price)

# Function to preprocess data and make prediction
def predict_price(data):
    features = ['crime_rate', 'num_rooms', 'indus', 'house_age', 'distance', 'tax_rate']
    sample_data = pd.DataFrame([data], columns=features)
    scaler = StandardScaler()
    sample_data_scaled = scaler.fit_transform(sample_data)
    predicted_price = model.predict(sample_data_scaled)
    return predicted_price[0]

if __name__ == '__main__':
    app.run(debug=True)
