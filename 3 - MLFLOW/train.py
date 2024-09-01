# Importations
import pandas as pd
import numpy as np
import mlflow
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer  # Ajoutez cette ligne
from xgboost import XGBRegressor

# Defining arguments 
experiment_name = "mlflow-getaround_project"
test_size = 0.2
random_state = 0

# Dataset import
dataset = pd.read_csv('get_around_pricing_project.csv')

# Pandas Preprocessings
dataset = dataset.drop(columns=['Unnamed: 0'])

# Sklearn Preprocessings
print("Separating labels from features...")
target = 'rental_price_per_day'
X = dataset.drop(columns=[target])
Y = dataset[target]
print("...Done.\n")

print("Y : ")
print(Y.head())
print("\nX :")
print(X.head())

# Verify the distribution of classes in out target Y
class_counts = Y.value_counts()
print("Distribution of classes :")
print(class_counts)

# Filter rare classes if necessary
min_class_count = 2
rare_classes = class_counts[class_counts < min_class_count].index

# Delete rare classes from X and Y
X_filtered = X[~Y.isin(rare_classes)]
Y_filtered = Y[~Y.isin(rare_classes)]

# Convert columns to float
X_filtered = X_filtered.astype({column: 'float64' for column in X_filtered.select_dtypes(include='int').columns})

# Dividing into train and test set
print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X_filtered, Y_filtered, test_size=test_size, stratify=Y_filtered, random_state=random_state)
print("...Done.\n")

# Identify numeric and categorical features 
numeric_features = ['mileage', 'engine_power']
categorical_features = [
    'model_key', 
    'fuel', 
    'paint_color', 
    'car_type', 
    'private_parking_available', 
    'has_gps', 
    'has_air_conditioning',
    'automatic_car', 
    'has_getaround_connect', 
    'has_speed_regulator', 
    'winter_tires'
]

# Pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute les valeurs manquantes avec la moyenne
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

print("Performing preprocessings on train set...")
X_train_transformed = preprocessor.fit_transform(X_train)
print("...Done.\n")

print(pd.DataFrame(X_train_transformed).head(5))

print("Performing preprocessings on test set...")
X_test_transformed = preprocessor.transform(X_test)
print("...Done.\n")

print(pd.DataFrame(X_test_transformed).head(5))

# Configur MLflow environment
mlflow.set_tracking_uri("https://predictions-fastapi-1e8cd1420e70.herokuapp.com/")
mlflow.set_experiment(experiment_name)


experiment = mlflow.get_experiment_by_name(experiment_name)


mlflow.sklearn.autolog()

# Models dictionnary 
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
    "XGBoost Regressor": XGBRegressor(learning_rate=0.05, max_depth=8, min_child_weight=4, n_estimators=150)
}

# Training and evaluation
for model_name, model in models.items():
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name):
        # Pipeline for every model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Training model
        pipeline.fit(X_train, Y_train)

        # Predictions
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        # Calculate metrics
        train_r2 = r2_score(Y_train, train_pred)
        test_r2 = r2_score(Y_test, test_pred)
        mse = mean_squared_error(Y_test, test_pred)
        rmse = mean_squared_error(Y_test, test_pred, squared=False)
        mae = mean_absolute_error(Y_test, test_pred)

        # Show metrics
        print(f"\nModel: {model_name}")
        print(f"Train R2: {train_r2}")
        print(f"Test R2: {test_r2}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        # Save metrics and preprocessor
        mlflow.log_metric("train_r2_score", train_r2)
        mlflow.log_metric("test_r2_score", test_r2)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("root_mean_squared_error", rmse)
        mlflow.log_metric("mean_absolute_error", mae)

        preprocessor_path = f"preprocessor_{model_name}.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)

        # Save preprocessor in MLflow
        mlflow.log_artifact(preprocessor_path)
