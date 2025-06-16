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



import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def load_data(path):
    return pd.read_csv(path)

def build_pipeline():
    categorical_features = ['EducationLevel', 'JobRole']
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns (e.g., YearsExperience)
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    return pipeline


from sklearn.metrics import mean_absolute_error, r2_score

def train(pipeline, X, y):
    pipeline.fit(X, y)
    return pipeline

def evaluate(pipeline, X, y):
    preds = pipeline.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return mae, r2


from preprocess import load_data, build_pipeline
from model import train, evaluate

def main():
    data = load_data("data/employees.csv")

    X = data.drop("Salary", axis=1)
    y = data["Salary"]

    pipeline = build_pipeline()
    model = train(pipeline, X, y)

    mae, r2 = evaluate(model, X, y)

    print(f"✅ مدل آموزش دید! MAE: {mae:.2f} | R²: {r2:.2f}")

    sample = {
        'YearsExperience': [5],
        'EducationLevel': ['Master'],
        'JobRole': ['Data Scientist']
    }

    import pandas as pd
    sample_df = pd.DataFrame(sample)
    predicted_salary = model.predict(sample_df)[0]
    print(f"💰 پیش‌بینی حقوق: {predicted_salary:.2f} دلار در سال")

if __name__ == "__main__":
    main()

