# Upland.me-Property-Price-Predictor
This project uses a neural network built with TensorFlow and Keras to predict future real estate prices based on property attributes like size, location, and market trends. Users can input details interactively, and the model provides a price prediction along with a 12-month forecast visualization.

![E6FF0EE4-FBAD-4D7F-8695-5280068841B0_1_105_c](https://github.com/user-attachments/assets/8ed31059-c387-4cf0-a6fe-e30634cf50fd)

Description:

This project utilizes a deep learning model built with TensorFlow and Keras to predict future real estate prices based on various property attributes. The dataset is generated randomly and includes features such as property price, size, distance to landmarks, sales volume, neighborhood rating, regional average price, and recent price growth.

Key Features:

	+  Data Generation: Randomly simulated dataset with real estate market attributes.
	+  Data Processing: Standardization using Scikit-Learn’s StandardScaler.
	+  Neural Network Model: Multi-layer perceptron with ReLU activation and dropout.
	+  Interactive User Input: Predicts future property prices based on user-provided data.
	+  Visualization: Plots future price trends based on predicted growth.

Programms needed:

	•	Python (3.8 oder neuer)
	•	Pip (Python-Paketmanager)
	•	TensorFlow (pip install tensorflow)
	•	NumPy (pip install numpy)
	•	Matplotlib (pip install matplotlib)
	•	Scikit-Learn (pip install scikit-learn)

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generating Random Data
np.random.seed(42)
num_samples = 5000  # More Data for Generalization

usd_price = np.random.uniform(3, 5000, num_samples)
size_up2 = np.random.uniform(20, 5000, num_samples)
landmark_distance = np.random.uniform(0, 10, num_samples)
sales_count = np.random.randint(1, 100, num_samples)
neighborhood_rating = np.random.uniform(1, 10, num_samples)
region_avg_price = np.random.uniform(20000, 200000, num_samples)
price_growth_30d = np.random.uniform(-5, 15, num_samples)
price_per_up2 = usd_price * 1000 / size_up2

# Calculate Future Price
future_price_upx = (
    (usd_price * 1000) + (5 * size_up2) + (-200 * landmark_distance) +
    (15 * sales_count) + (1600 * neighborhood_rating) + (0.7 * region_avg_price) +
    (600 * price_growth_30d) + (8 * price_per_up2) + np.random.normal(0, 500, num_samples)
)

# Prepare Data
X = np.column_stack([usd_price, size_up2, landmark_distance, sales_count, neighborhood_rating, region_avg_price, price_growth_30d, price_per_up2])
y = future_price_upx
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.Huber(), metrics=['mae'])

# Training
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# Price Prediction and Progrnose
def predict_price():
    try:
        input_type = input("Calculate Price in USD or UPX ? (usd/upx): ").strip().lower()
        if input_type == 'usd':
            usd_price = float(input("💵 Property value in USD: "))
            upx_price = usd_price * 1000  # conversion
        elif input_type == 'upx':
            upx_price = float(input("💰 Property value in UPX: "))
            usd_price = upx_price / 1000  # conversion
        else:
            print("⚠️ Invalid!")
            return

        size_up2 = float(input("🏠 Property size in UP²: "))
        landmark_distance = float(input("📍 Distance to landmark (km): "))
        sales_count = int(input("💰 neighborhood sales: "))
        neighborhood_rating = float(input("🏙️ Rating (1-10): "))
        region_avg_price = float(input("🌍 Average Price in region (UPX): "))
        price_growth_30d = float(input("📈 Price growth in the last 30 Days (%): "))
        price_per_up2 = upx_price / size_up2

        input_data = np.array([[usd_price, size_up2, landmark_distance, sales_count, neighborhood_rating, region_avg_price, price_growth_30d, price_per_up2]])
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)[0][0]

        print(f"📌 Predicted Price: {round(prediction, 2)} UPX")

        # Future Price
        plot_future_price(prediction, price_growth_30d)

    except ValueError:
        print("⚠️ Invalid! Only numbers.")

# function for simulating future price developmen
def plot_future_price(start_price, growth_rate):
    months = np.arange(1, 13)  # 12 months 
    future_prices = [start_price * ((1 + growth_rate / 100) ** month) for month in months]

    plt.figure(figsize=(8, 5))
    plt.plot(months, future_prices, marker='o', linestyle='-', color='b', label="Prognose")
    plt.axhline(start_price, color='r', linestyle='--', label="current price")
    plt.xlabel("future months")
    plt.ylabel("Price in UPX")
    plt.title("predicted price developmen")
    plt.legend()
    plt.grid(True)
    plt.show()


predict_price()
```

# Outputs

Output 1:

	•	63/63 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 72846.5234 - mae: 72847.0234 - val_loss: 13355.3564 - val_mae: 13355.8564
	•	Calculate Price in USD or UPX ? (usd/upx): usd
	•	💵 Property value in USD: 55
	•	🏠 Property size in UP²: 300
	•	📍 Distance to landmark (km): 3
	•	💰 neighborhood sales: 300
	•	🏙️ Rating (1-10): 6
	•	🌍 Average Price in region (UPX): 500000
	•	📈 Price growth in the last 30 Days (%): 4
	•	1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 139ms/step
	•	📌 Predicted Price: 1000605.125 UPX

![image](https://github.com/user-attachments/assets/86ed0b31-6f71-4717-871d-c39d468fa2a0)

Output 2:

	•	63/63 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 73789.9375 - mae: 73790.4375 - val_loss: 17415.4844 - val_mae: 17415.9844
	•	Calculate Price in USD or UPX ? (usd/upx): upx
	•	💰 Property value in UPX: 60
	•	🏠 Property size in UP²: 160
	•	📍 Distance to landmark (km): 5
	•	💰 neighborhood sales: 1500
	•	🏙️ Rating (1-10): 3
	•	🌍 Average Price in region (UPX): 40
	•	Price growth in the last 30 Days (%): 1
	•	1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 103ms/step
	•	 📌 Predicted Price: 4156429.0 UPX

![image](https://github.com/user-attachments/assets/85010c65-0b01-46fb-b716-09b304709460)

# Summary • Please note!

This project aimed to create a comprehensive tool for simulating future price development for Uplands´ virtual economy, based on randomly generated market trends. By integrating predictive algorithms and real-time data analysis, the tool provides forecasts for property or product values in a specific neighborhood or district. While the model offers valuable insights, it is important to note that the predictions are still highly inaccurate and cannot be reliably applied in real-world scenarios at this stage. The data should be treated with caution, as the model requires further refinement and testing before it can provide consistent and trustworthy results.
