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

# function for simulating future price development
def plot_future_price(start_price, growth_rate):
    months = np.arange(1, 13)  # 12 months 
    future_prices = [start_price * ((1 + growth_rate / 100) ** month) for month in months]

    plt.figure(figsize=(8, 5))
    plt.plot(months, future_prices, marker='o', linestyle='-', color='b', label="Prognose")
    plt.axhline(start_price, color='r', linestyle='--', label="current price")
    plt.xlabel("future months")
    plt.ylabel("Price in UPX")
    plt.title("predicted price development")
    plt.legend()
    plt.grid(True)
    plt.show()


predict_price()
