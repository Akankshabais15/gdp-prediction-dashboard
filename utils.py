import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# -------------------------------
# Data Preprocessing
# -------------------------------
def preprocess_df(df):
    df = df.copy()
    df = df.dropna()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

# -------------------------------
# Train & Evaluate Models
# -------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Linear Regression"):
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_name == "XGBoost":
        if xgb is None:
            raise ValueError("XGBoost is not installed. Run: pip install xgboost")
        model = xgb.XGBRegressor(n_estimators=200, random_state=42, objective="reg:squarederror", verbosity=0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return model, preds, {"MAE": mae, "RMSE": rmse, "R2": r2}, scaler

# -------------------------------
# Plotting helper
# -------------------------------
def plot_actual_vs_pred(y_test, preds, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(y=preds, mode='lines+markers', name=model_name))
    fig.update_layout(title=f'Actual vs Predicted GDP - {model_name}', xaxis_title='Index', yaxis_title='GDP', template="plotly_white")
    return fig
