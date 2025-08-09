import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cars_data = pd.read_csv('Cardetails.csv')
print(cars_data.head())

cars_data.drop(columns=['torque'], inplace=True)
print(cars_data.head())
print(cars_data.shape)
print(cars_data.isnull().sum())

cars_data.dropna(inplace=True)
print(cars_data.isnull().sum())
print(cars_data.shape)

print(cars_data.duplicated().sum())
cars_data.drop_duplicates(inplace=True)
print(cars_data.duplicated().sum())
print(cars_data.shape)

print(cars_data)
print(cars_data.info())

for col in cars_data.columns:
    print('Unique values of ' + col)
    print(cars_data[col].unique())
    print("=======================")

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

def clean_data(value):
    value = value.split(' ')[0]
    value = value.strip()
    if value == '':
        value = 0
    return float(value)

cars_data['name'] = cars_data['name'].apply(get_brand_name)
print(cars_data)

cars_data['mileage'] = cars_data['mileage'].apply(clean_data)
print(cars_data)

cars_data['max_power'] = cars_data['max_power'].apply(clean_data)
print(cars_data)

cars_data['engine'] = cars_data['engine'].apply(clean_data)
print(cars_data)

for col in cars_data.columns:
    print('Unique values of ' + col)
    print(cars_data[col].unique())
    print("=======================")

# Manual label encoding
cars_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra',
 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi',
 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], inplace=True)

cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)

cars_data.reset_index(inplace=True)
cars_data.drop(columns=['index'], inplace=True)

input_data = cars_data.drop(columns=['selling_price'])
output_data = cars_data['selling_price']
print(input_data)
print(output_data)

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as model.pkl")


predict = model.predict(x_test)
print(predict)

print(x_train.head(1))
input_data_model = pd.DataFrame([
    [5,2025,100,1,1,1,1,12.99,2494.0,200.6,5.0]
], columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])
print(input_data_model)

print(model.predict(input_data_model))
