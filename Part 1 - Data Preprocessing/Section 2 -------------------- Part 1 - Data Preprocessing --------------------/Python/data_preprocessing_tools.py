import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox

# Sample data
data = {
    'Age': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 1, 2, 3, 4, 5,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    'Mileage': [10000, 20000, 30000, 40000, 50000,
                6000, 12000, 22000, 32000, 42000,
                50000, 15000, 25000, 35000, 45000,
                55000, 65000, 75000, 85000, 95000,
                11000, 21000, 31000, 41000, 51000,
                1000, 9000, 19000, 29000, 39000,
                49000, 59000, 69000, 79000, 89000,
                7500, 17500, 27500, 37500, 47500,
                8500, 18500, 28500, 38500, 48500,
                9500, 29500, 39500, 49500, 59500],

    'Engine_Size': [1.0, 1.5, 2.0, 2.5, 3.0,
                    1.2, 1.8, 2.2, 2.7, 3.2,
                    1.1, 1.6, 2.1, 2.6, 3.1,
                    1.3, 1.4, 2.3, 2.8, 3.3,
                    1.5, 1.7, 2.4, 2.9, 3.4,
                    1.9, 2.0, 2.5, 3.0, 3.5,
                    1.4, 1.9, 2.6, 2.1, 2.8,
                    1.6, 2.2, 2.3, 3.0, 2.9,
                    1.8, 1.9, 2.7, 2.8, 3.1,
                    2.1, 2.3, 2.5, 3.2, 3.3],

    'Horsepower': [50, 75, 100, 125, 150,
                   55, 80, 105, 130, 155,
                   60, 85, 110, 135, 160,
                   65, 90, 115, 140, 165,
                   70, 95, 120, 145, 170,
                   75, 100, 125, 150, 175,
                   80, 110, 140, 170, 200,
                   85, 115, 145, 175, 205,
                   90, 120, 150, 180, 210,
                   95, 125, 155, 185, 215],

    'Number_of_Doors': [2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4,
                        2, 4, 4, 2, 4],

    'Price': [5000, 7000, 9000, 11000, 13000,
              6000, 8000, 10000, 12000, 14000,
              7000, 9000, 11000, 13000, 15000,
              8000, 10000, 12000, 14000, 16000,
              9000, 11000, 13000, 15000, 17000,
              10000, 12000, 14000, 16000, 18000,
              11000, 13000, 15000, 17000, 19000,
              12000, 14000, 16000, 18000, 20000,
              13000, 15000, 17000, 19000, 21000,
              14000, 16000, 18000, 20000, 22000]
}

# Load and prepare the dataset
def load_data():
    dataset = pd.DataFrame(data)
    X = dataset.iloc[:, :-1].values  # Features
    y = dataset.iloc[:, -1].values    # Target variable (Price)
    return X, y

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)  # Use RandomForestRegressor
    regressor.fit(X_train, y_train)
    return regressor

# Predict the price
def predict_price(regressor, input_features):
    input_features = np.array(input_features).reshape(1, -1)  # Reshape for a single prediction
    predicted_price = regressor.predict(input_features)
    return predicted_price[0]

# Function to handle the prediction button click
def on_predict():
    try:
        features = [float(entry.get()) for entry in entries]
        predicted_price = predict_price(regressor, features)
        result_var.set(f"Predicted Price: ${predicted_price:,.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Load data and train the model
X, y = load_data()
regressor = train_model(X, y)

# Create the GUI
root = tk.Tk()
root.title("Car Price Predictor")

# Input labels and entry fields
labels = ['Age (Years)', 'Mileage (Miles)', 'Engine Size (Liters)', 'Horsepower', 'Number of Doors']
entries = []

for label in labels:
    frame = tk.Frame(root)
    frame.pack(pady=5)

    lbl = tk.Label(frame, text=label)
    lbl.pack(side=tk.LEFT)

    entry = tk.Entry(frame)
    entry.pack(side=tk.RIGHT)
    entries.append(entry)

# Button to predict price
predict_button = tk.Button(root, text="Predict Price", command=on_predict)
predict_button.pack(pady=20)

# Variable to hold the result
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var)
result_label.pack(pady=20)

# Start the GUI main loop
root.mainloop()
