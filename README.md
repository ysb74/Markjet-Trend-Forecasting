# Market Trend Forecasting

## Goal

This project aims to predict future market dynamics in the real estate sector, such as changes in demand, property value appreciation/devaluation, rent price fluctuations, and overall leasing activity. By leveraging various economic, demographic, and historical real estate data, the models developed here provide insights into potential market shifts.

## Features

The forecasting models utilize a diverse set of features, including:

  * **Economic Indicators:** Interest rates, Gross Domestic Product (GDP), unemployment rates, inflation.
  * **Demographic Shifts:** Population growth, migration patterns, age distribution changes.
  * **Historical Data:** Past property sales prices, rental rates, listing volumes, and transaction counts.
  * **Construction Data:** New construction permits issued, housing starts.
  * **Local Policies:** Conceptual representation of policy changes (e.g., zoning, tax incentives).

## Algorithms

A range of time series and regression algorithms are employed to capture different aspects of market trends:

  * **ARIMA (AutoRegressive Integrated Moving Average):** A classical statistical method for time series forecasting.
  * **Prophet:** A forecasting procedure developed by Facebook, optimized for business forecasts with strong seasonal effects and holidays.
  * **LSTM (Long Short-Term Memory) Neural Networks:** A type of recurrent neural network (RNN) well-suited for sequence prediction problems, capable of learning long-term dependencies.

## Tools/Libraries

  * **Data Handling:** `pandas`, `numpy`
  * **Data Ingestion:** `requests`, `BeautifulSoup` (for conceptual scraping, though APIs are preferred for real data)
  * **Statistical Models:** `statsmodels`
  * **Time Series Forecasting:** `prophet`
  * **Deep Learning:** `tensorflow`, `keras`
  * **Machine Learning Utilities:** `scikit-learn`
  * **Model Persistence:** `joblib`
  * **MLOps (Experiment Tracking & Model Management):** `mlflow`
  * **Visualization:** `matplotlib`, `seaborn`

## Project Structure