import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from model import train_model, evaluate_model
from plot import plot_regression


X = np.array([50, 60, 70, 80, 90, 100, 120, 150]).reshape(-1, 1)
y = np.array([300, 360, 400, 430, 500, 550, 610, 700])

model = LinearRegression()
model.fit(X, y)

predicted_price = model.predict([[110]])
print(f"پیش‌بینی قیمت برای 110 متر: {predicted_price[0]:.2f} میلیون تومان")

plt.scatter(X, y, color='blue', label='داده‌های واقعی')
plt.plot(X, model.predict(X), color='red', label='خط رگرسیون')
plt.xlabel('متراژ (متر مربع)')
plt.ylabel('قیمت (میلیون تومان)')
plt.title('رگرسیون خطی: متراژ vs قیمت')
plt.legend()
plt.grid(True)
plt.show()




def plot_regression(X, y, model):
    plt.scatter(X, y, color='blue', label='داده‌های واقعی')
    plt.plot(X, model.predict(X), color='red', label='خط رگرسیون')
    plt.xlabel('متراژ (متر مربع)')
    plt.ylabel('قیمت (میلیون تومان)')
    plt.title('رگرسیون خطی: متراژ vs قیمت')
    plt.legend()
    plt.grid(True)
    plt.show()




def main():
    df = pd.read_csv('data/housing.csv')
    X = df[['area']].values
    y = df['price'].values

    model = train_model(X, y)

    score = evaluate_model(model, X, y)
    print(f"📈 دقت مدل (R²): {score:.2f}")

    sample_area = 110
    prediction = model.predict([[sample_area]])[0]
    print(f"🏠 قیمت پیش‌بینی‌شده برای {sample_area} متر: {prediction:.2f} میلیون تومان")

    plot_regression(X, y, model)

if __name__ == "__main__":
    main()
