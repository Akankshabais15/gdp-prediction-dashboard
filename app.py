import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_df, train_and_evaluate, plot_actual_vs_pred

st.set_page_config(
    page_title="GDP Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load and preprocess data
# -------------------------------
df = pd.read_csv("IndData.csv")
df = preprocess_df(df)
target_col = "GDP (current US$)"
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Analysis", "Model Comparison"])

# -------------------------------
# Prediction Page
# -------------------------------
if page == "Prediction":
    st.header("📈 GDP Prediction")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = ["Linear Regression", "Random Forest", "XGBoost"]
    results = {}

    st.subheader("Model Predictions on Test Data")
    for model_name in models:
        try:
            model, preds, metrics, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, model_name)
            results[model_name] = {"model": model, "preds": preds, "scaler": scaler, "metrics": metrics}
            st.success(f"**{model_name}** → R² = {metrics['R2']:.3f}, RMSE = {metrics['RMSE']:.2f}, MAE = {metrics['MAE']:.2f}")
        except Exception as e:
            st.warning(f"{model_name} failed: {e}")

    st.subheader("Manual Input for Prediction")
    input_vals = {}
    cols = st.columns(3)

    for i, col in enumerate(X.columns):
        mn, mx = float(X[col].min()), float(X[col].max())
        val = float(X[col].median())

    # Special handling for Year
        if col.lower() == "year":
            input_vals[col] = cols[i % 3].number_input(
            col,
            value=1990,           
            min_value=int(mn),
            max_value=2050        
        )
        else:
            input_vals[col] = cols[i % 3].number_input(
            col,
            value=val,
            min_value=mn,
            max_value=mx
        )


    if st.button("Predict GDP in Current US$"):
        X_sample = pd.DataFrame([input_vals])[X.columns]
        st.success("Predictions:")
        for name, r in results.items():
            try:
                pred = r["model"].predict(r["scaler"].transform(X_sample))
                st.write(f"- **{name}**: ${float(pred[0]):,.2f}")
            except Exception as e:
                st.write(f"- **{name}**: ❌ Error in prediction")

# -------------------------------
# Analysis Page
# -------------------------------
elif page == "Analysis":
    st.header("📊 GDP Analysis (Indian Economy)")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    # GDP Trend
    st.subheader("GDP Trend Over Years")
    fig_gdp = px.line(df, x='Year', y='GDP (current US$)', markers=True, title='GDP Trend (Current US$)')
    st.plotly_chart(fig_gdp, use_container_width=True)

    # Annual GDP Growth
    st.subheader("Annual GDP Growth (%)")
    fig_growth = px.line(df, x='Year', y='GDP growth (annual %)', markers=True, title='Annual GDP Growth (%)')
    st.plotly_chart(fig_growth, use_container_width=True)

    # Population vs Life Expectancy
    st.subheader("Population vs Life Expectancy")
    fig_pop = go.Figure()
    fig_pop.add_trace(go.Scatter(x=df['Year'], y=df['Population, total'], mode='lines+markers', name='Population'))
    fig_pop.add_trace(go.Scatter(x=df['Year'], y=df['Life expectancy at birth, total (years)'], mode='lines+markers', name='Life Expectancy'))
    fig_pop.update_layout(title='Population and Life Expectancy Over Years', xaxis_title='Year', yaxis_title='Population / Life Expectancy')
    st.plotly_chart(fig_pop, use_container_width=True)

    # Sector Contributions
    st.subheader("Sector Contributions to GDP (%)")
    sectors = [
        'Agriculture, forestry, and fishing, value added (% of GDP)',
        'Industry (including construction), value added (% of GDP)',
        'Exports of goods and services (% of GDP)',
        'Imports of goods and services (% of GDP)',
        'Military expenditure (% of GDP)',
        'Merchandise trade (% of GDP)'
    ]
    fig_sector = px.bar(df, x='Year', y=sectors, title='Sector Contributions to GDP (%)')
    st.plotly_chart(fig_sector, use_container_width=True)

    # Exports vs Imports
    st.subheader("Exports vs Imports")
    fig_trade = go.Figure()
    fig_trade.add_trace(go.Scatter(x=df['Year'], y=df['Exports of goods and services (% of GDP)'], name='Exports', line=dict(color='green')))
    fig_trade.add_trace(go.Scatter(x=df['Year'], y=df['Imports of goods and services (% of GDP)'], name='Imports', line=dict(color='red')))
    fig_trade.update_layout(title='Exports and Imports (% of GDP)', xaxis_title='Year', yaxis_title='% of GDP')
    st.plotly_chart(fig_trade, use_container_width=True)

# -------------------------------
# Model Comparison Page
# -------------------------------
elif page == "Model Comparison":
    st.title("📊 Model Comparison: Actual vs Predicted GDP")

    X = df.drop(columns=["GDP (current US$)", "Year", "Unnamed: 0"], errors="ignore")
    y = df["GDP (current US$)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models_to_compare = ["Linear Regression", "Random Forest", "XGBoost"]
    all_metrics = {}

    for model_name in models_to_compare:
        try:
            model, preds, metrics, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, model_name)
            all_metrics[model_name] = metrics

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.values, mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=preds, mode="lines+markers", name=model_name))
            fig.update_layout(title=f"Actual vs Predicted GDP ({model_name})", xaxis_title="Index", yaxis_title="GDP", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            
        except Exception as e:
            st.error(f"{model_name} failed: {e}")

    # Summary Chart
    if all_metrics:
        st.subheader("📊 Overall Model Performance (R² Score)")
        summary_df = pd.DataFrame({
            "Model": list(all_metrics.keys()),
            "R2 Score": [m["R2"] for m in all_metrics.values()]
        })
        fig_summary = px.bar(summary_df, x="Model", y="R2 Score", color="Model", text="R2 Score", title="Model Comparison by R² Score", template="plotly_white")
        fig_summary.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_summary, use_container_width=True)
