from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    price = None
    if request.method == 'POST':
        try:

            name = int(request.form['name'])
            year = int(request.form['year'])
            km_driven = int(request.form['km_driven'])
            fuel = int(request.form['fuel'])
            seller_type = int(request.form['seller_type'])
            transmission = int(request.form['transmission'])
            owner = int(request.form['owner'])
            mileage = float(request.form['mileage'])
            engine = float(request.form['engine'])
            max_power = float(request.form['max_power'])
            seats = int(request.form['seats'])


            features = np.array([[name, year, km_driven, fuel, seller_type, transmission,
                                  owner, mileage, engine, max_power, seats]])

            prediction = model.predict(features)[0]
            price = round(prediction, 2)
        except Exception as e:
            price = f"Error: {e}"

    return render_template('index.html', price=price)

if __name__ == "__main__":
    app.run(debug=True)
