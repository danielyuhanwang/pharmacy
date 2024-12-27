import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import prophet
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
from prophet.plot import plot_cross_validation_metric
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
plt.style.use('fivethirtyeight')
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100

class TimeSeriesProphet():

    def __init__(self, data, drug_name):
        self.data = data[data["Product_Category"] == drug_name]
        self.data = self.data[['Date','Units_Sold']]
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        split_date = '2022-01-01'
        self.train_data = self.data.loc[self.data.index <= split_date].copy()
        self.test_data = self.data.loc[self.data.index > split_date].copy()

        self.train_data = self.train_data.rename(columns={'Units_Sold': 'TRAINING SET UNITS SOLD'})
        self.test_data = self.test_data.rename(columns={'Units_Sold': 'TEST SET UNITS SOLD'})
        self.proph_model = self.train_data.reset_index() \
        .rename(columns={'Date':'ds',
                     'TRAINING SET UNITS SOLD':'y'})
        self.model = Prophet()
        self.model.fit(self.proph_model)
        self.test_prophet = self.test_data.reset_index() \
        .rename(columns={'Date':'ds',
                         'TEST SET UNITS SOLD':'y'})
        self.test_results = self.model.predict(self.test_prophet)
    def mod(self):
        return self.model
    def error(self):
        mse = np.sqrt(mean_squared_error(y_true=self.test_prophet['y'],
                       y_pred=self.test_results['yhat']))
        abs_mse = mean_absolute_error(y_true=self.test_prophet['y'],
                       y_pred=self.test_results['yhat'])
        abs_mse_percentage = mean_absolute_percentage_error(y_true=self.test_prophet['y'],
                       y_pred=self.test_results['yhat'])
        return mse, abs_mse, abs_mse_percentage
    def forecast_diagram(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig = self.model.plot(self.test_results, ax=ax)
        ax.set_title('Prophet Forecast')
        plt.show()
    def predict(self, years):
        
        future = self.model.make_future_dataframe(periods=365 * years, freq='d', include_history=True)
        forecast = self.model.predict(future)
        forecast_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        forecast_results = forecast[forecast_columns]
        return forecast_results
    def predict_plot(self, years):
        future = self.model.make_future_dataframe(periods=365 * years, freq='d', include_history=True)
        forecast = self.model.predict(future)
        forecast_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        forecast_results = forecast[forecast_columns]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig = self.model.plot(forecast_results, ax=ax)
        ax.set_title('Prophet Forecast')
        plt.show()
    def forecast_diagram_with_comparison(self):
        test_yhat = self.test_results[['ds', 'yhat']]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig = self.model.plot(self.test_results, ax=ax)
        ax.scatter(self.test_prophet['ds'], self.test_prophet['y'], color='r', label='Actual Test Data')
        ax.legend()
        ax.set_title('Prophet Forecast with Actual Test Data')
        plt.show()
    def components(self):
        fig = self.model.plot_components(self.test_results)
        plt.show()
    

class ProphetCV(TimeSeriesProphet):
    def __init__(self, data, drug_name):
        super().__init__(data, drug_name)

        self.df_cv = cross_validation(self.model, initial = '730 days', period = '180 days', horizon = '365 days')
        
    def generalization(self):
        fig = plot_cross_validation_metric(self.df_cv, metric='rmse')

    def grid_search(self):
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [1.0, 5.0, 10.0],
            'holidays_prior_scale': [1.0, 10.0, 20.0]
        }
        
        best_params = None
        best_rmse = float('inf')
        
        for changepoint in param_grid['changepoint_prior_scale']:
            for seasonality in param_grid['seasonality_prior_scale']:
                for holiday in param_grid['holidays_prior_scale']:
                    model = Prophet(
                        changepoint_prior_scale=changepoint,
                        seasonality_prior_scale=seasonality,
                        holidays_prior_scale = holiday
                    )
                    model.fit(self.proph_model)
            
                    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
                    df_p = performance_metrics(df_cv)
            
                    if df_p['rmse'].mean() < best_rmse:
                        best_rmse = df_p['rmse'].mean()
                        best_params = {
                            'changepoint_prior_scale': changepoint,
                            'seasonality_prior_scale': seasonality,
                            'holidays_prior_scale': holiday
                        }
        self.params = best_params
        print("Best Parameters:", best_params)
        print("Best RMSE:", best_rmse)
        return self.params
    def predict_optimal(self, years):
        final_model = Prophet(
    changepoint_prior_scale = self.params['changepoint_prior_scale'],
    seasonality_prior_scale = self.params['seasonality_prior_scale'],
    holidays_prior_scale= self.params['holidays_prior_scale']
)
        final_model.fit(self.proph_model)
        future = final_model.make_future_dataframe(periods=365 * years, freq='d', include_history=True)
        forecast = final_model.predict(future)
        forecast_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        forecast_results = forecast[forecast_columns]
        return forecast_results
    def predict_optimal_graph(self, years):
        final_model = Prophet(
    changepoint_prior_scale = self.params['changepoint_prior_scale'],
    seasonality_prior_scale = self.params['seasonality_prior_scale'],
    holidays_prior_scale= self.params['holidays_prior_scale']
)
        final_model.fit(self.proph_model)
        future = final_model.make_future_dataframe(periods=365 * years, freq='d', include_history=True)
        forecast = final_model.predict(future)
        forecast_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        forecast_results = forecast[forecast_columns]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig = final_model.plot(forecast_results, ax=ax)
        ax.set_title('Prophet Forecast')
        plt.show()
        
        
        
                