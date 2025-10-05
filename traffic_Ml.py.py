import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Simulated dataset
data = {
    'hour': [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]*7,  # morning & evening peaks
    'day_of_week': [i for i in range(7) for _ in range(10)], # 0=Mon,6=Sun
    'traffic_volume': [50, 200, 500, 300, 150, 400, 800, 1000, 600, 300]*7
}

df = pd.DataFrame(data)
print(df.head())
plt.scatter(df['hour'], df['traffic_volume'])
plt.xlabel("Hour of Day")
plt.ylabel("Traffic Volume")
plt.title("Traffic Volume vs Time of Day")
plt.show()
X = df[['hour', 'day_of_week']]   # Features
y = df['traffic_volume']          # Target (what we want to predict)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Traffic Volume Prediction")
plt.show()


