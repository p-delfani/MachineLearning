import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# داده‌ها
X = np.array([50, 60, 70, 80, 90, 100, 120, 150]).reshape(-1, 1)
y = np.array([300, 360, 400, 430, 500, 550, 610, 700])

# آموزش مدل
model = LinearRegression()
model.fit(X, y)

# پیش‌بینی
predicted_price = model.predict([[110]])
print(f"پیش‌بینی قیمت برای 110 متر: {predicted_price[0]:.2f} میلیون تومان")

# رسم نمودار
plt.scatter(X, y, color='blue', label='داده‌های واقعی')
plt.plot(X, model.predict(X), color='red', label='خط رگرسیون')
plt.xlabel('متراژ (متر مربع)')
plt.ylabel('قیمت (میلیون تومان)')
plt.title('رگرسیون خطی: متراژ vs قیمت')
plt.legend()
plt.grid(True)
plt.show()
