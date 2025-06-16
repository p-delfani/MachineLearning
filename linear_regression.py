import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def load_data(path):
    return pd.read_csv(path)

def build_pipeline():
    categorical_features = ['EducationLevel', 'JobRole']
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    return pipeline

def train(pipeline, X, y):
    pipeline.fit(X, y)
    return pipeline

def evaluate(pipeline, X, y):
    preds = pipeline.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return mae, r2

def main():
    # Load data
    data = load_data("employees.csv")

    # Split features and target
    X = data.drop("Salary", axis=1)
    y = data["Salary"]

    # Build and train model
    pipeline = build_pipeline()
    model = train(pipeline, X, y)

    # Evaluate model
    mae, r2 = evaluate(model, X, y)
    print("Model trained successfully.")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Predict salary for a sample input
    sample = pd.DataFrame([{
        'YearsExperience': 5,
        'EducationLevel': 'Master',
        'JobRole': 'Data Scientist'
    }])

    predicted_salary = model.predict(sample)[0]
    print(f"Predicted salary: ${predicted_salary:.2f} per year")

if __name__ == "__main__":
    main()
