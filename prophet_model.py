import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ── 1. Load & inspect ────────────────────────────────────────────────────────
file_path = '/Users/olga/Desktop/BIA 678 PROJECT/DATA_2026-04-36.csv'
df = pd.read_csv(file_path)
print("Shape:", df.shape)
print(df.head())

# ── 2. Prepare data for Prophet ──────────────────────────────────────────────
# Prophet requires exactly two columns: 'ds' (datestamp) and 'y' (value)
prophet_df = df[['issue_date', 'payment_amount']].copy()
prophet_df.columns = ['ds', 'y']

# Parse dates — adjust the format string to match your CSV
# Common formats: '%d/%m/%Y'  '%Y-%m-%d'  '%m/%d/%Y'
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%m/%d/%Y')

# Drop rows with missing dates or amounts
prophet_df = prophet_df.dropna(subset=['ds', 'y'])

# Optional: if payment_amount is a string like "$1,200.00", clean it first:
prophet_df['y'] = prophet_df['y'].replace('[\$,]', '', regex=True).astype(float)

# Aggregate to daily revenue (Prophet expects one row per date)
prophet_df = prophet_df.groupby('ds', as_index=False)['y'].sum()

print("\nPrepared data sample:")
print(prophet_df.head())
print(f"Date range: {prophet_df['ds'].min()} → {prophet_df['ds'].max()}")

# ── 3. Build & fit the model ──────────────────────────────────────────────────
model = Prophet(
    yearly_seasonality=True,   # captures annual revenue cycles
    weekly_seasonality=True,   # captures day-of-week patterns
    daily_seasonality=False,   # only enable if you have sub-daily data
    interval_width=0.95        # 95% confidence interval on the forecast
)

model.fit(prophet_df)  # <-- pass the dataframe, not the column names

# ── 4. Forecast ───────────────────────────────────────────────────────────────
FORECAST_DAYS = 365
future = model.make_future_dataframe(periods=FORECAST_DAYS, freq='D')
forecast = model.predict(future)

# Key forecast columns:
# 'ds'    — the date
# 'yhat'  — predicted revenue
# 'yhat_lower' / 'yhat_upper' — confidence interval bounds
print("\nForecast tail:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig1 = model.plot(forecast)
fig1.suptitle('Revenue Forecast', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.tight_layout()
plt.show()

fig2 = model.plot_components(forecast)
fig2.suptitle('Forecast Components (Trend + Seasonality)', fontsize=14)
plt.tight_layout()
plt.show()