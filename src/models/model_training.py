from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def prepare_data_for_training(df, target_col='Modal Price (Rs./Quintal)', test_size=0.2, random_state=42):
    """
    Prepare data for training by splitting into features and target
    """
    # Drop date column if it exists
    drop_cols = [target_col]
    if 'date' in df.columns:
        drop_cols.append('date')
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """
    Train a Random Forest model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using MSE and R2 score
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {'RMSE': rmse, 'R2': r2}

def train_and_evaluate_models(data):
    """
    Train and evaluate models for all commodities
    """
    results = {}
    for commodity, df in data.items():
        print(f"\nTraining model for {commodity}...")
        print("DataFrame columns:", df.columns.tolist())
        
        X_train, X_test, y_train, y_test = prepare_data_for_training(df)
        model = train_model(X_train, y_train)
        evaluation = evaluate_model(model, X_test, y_test)
        results[commodity] = {'model': model, 'evaluation': evaluation}
    return results