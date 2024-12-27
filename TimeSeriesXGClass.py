import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df
class XG_TimeSeries():
    def __init__(self,data, product_name):
        data = data.set_index("Date")
        data.index = pd.to_datetime(data.index)
        data = data[data["Product_Category"] == product_name]
        data = data[['Units_Sold']]
        train= data.loc[data.index < '01-01-2022']
        test = data.loc[data.index >= '01-01-2022']
        self.df = create_features(data)
        self.train = create_features(train)
        self.test = create_features(test)
        self.FEATURES = ['dayofyear', 'dayofmonth', 'dayofweek','weekofyear', 'quarter', 'month', 'year']
        self.TARGET = 'Units_Sold'
        
        self.X_train = self.train[self.FEATURES]
        self.y_train = self.train[self.TARGET]
        
        self.X_test = self.test[self.FEATURES]
        self.y_test = self.test[self.TARGET]
        self.X_train['weekofyear'] = self.X_train['weekofyear'].astype('int32')
        self.X_test['weekofyear'] = self.X_test['weekofyear'].astype('int32')
        self.reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:squarederror',
                       max_depth=3,
                       learning_rate=0.01)
        self.reg.fit(self.X_train, self.y_train,
        eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
        verbose=100)
        self.test['prediction'] = self.reg.predict(self.X_test)
    def feature_importance_plot(self):
        fi = pd.DataFrame(data=self.reg.feature_importances_,
             index=self.reg.feature_names_in_,
             columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()

    def forecast(self):
        self.df2 = self.df.merge(self.test[['prediction']], how='left', left_index=True, right_index=True)
        ax = self.df2[['Units_Sold']].plot(figsize=(15, 5))
        self.df2['prediction'].plot(ax=ax, style='.')
        plt.legend(['Truth Data', 'Predictions'])
        ax.set_title('Raw Data and Prediction')
        plt.show()

    def rmse(self):
        score = np.sqrt(mean_squared_error(self.test['Units_Sold'], self.test['prediction']))
        print(f'RMSE Score on Test set: {score:0.2f}')


class XGBoostCV(XG_TimeSeries):
    def __init__(self, data, drug_name):
        super().__init__(data, drug_name)
        self.tss = TimeSeriesSplit(n_splits=5, test_size= 182, gap=24)
        self.df = self.df.sort_index()
        
    def generalization(self):
        preds = []
        scores = []
        for train_idx, val_idx in self.tss.split(self.df):
            self.train = self.df.iloc[train_idx]
            self.test = self.df.iloc[val_idx]
        
            self.train = create_features(self.train)
            self.test = create_features(self.test)
        
            self.X_train = self.train[self.FEATURES]
            self.y_train = self.train[self.TARGET]
        
            self.X_test = self.test[self.FEATURES]
            self.y_test = self.test[self.TARGET]
            
            self.X_train['weekofyear'] = self.X_train['weekofyear'].astype('int32')
            self.X_test['weekofyear'] = self.X_test['weekofyear'].astype('int32')
        
            self.reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                                   n_estimators=1000,
                                   early_stopping_rounds=50,
                                   objective='reg:linear',
                                   max_depth=3,
                                   learning_rate=0.01)
            self.reg.fit(self.X_train, self.y_train,
                    eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                    verbose = False)

        self.y_pred = self.reg.predict(self.X_test)
        preds.append(self.y_pred)
        self.score = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        scores.append(self.score)
        return scores

    def grid_search(self):
        param_grid = {
    'n_estimators': [500, 1000],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

        grid = ParameterGrid(param_grid)
        
        self.best_params = None
        best_score = float('inf')
        all_scores = []
        
        
        fold = 0
        tss = TimeSeriesSplit(n_splits=5)  
        for params in grid:
            fold_scores = []
            for train_idx, val_idx in tss.split(self.df):
                self.train = self.df.iloc[train_idx]
                self.test = self.df.iloc[val_idx]
        
                self.train = create_features(self.train)
                self.test = create_features(self.test)
        
                self.X_train = self.train[self.FEATURES]
                self.y_train = self.train[self.TARGET]
                self.X_test = self.test[self.FEATURES]
                self.y_test = self.test[self.TARGET]
                
                self.X_train['weekofyear'] = self.X_train['weekofyear'].astype('int32')
                self.X_test['weekofyear'] = self.X_test['weekofyear'].astype('int32')
        
            
                self.reg = xgb.XGBRegressor(
                    base_score=0.5,
                    booster='gbtree',
                    early_stopping_rounds=50,
                    objective='reg:squarederror', 
                    **params
                )
        
                self.reg.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                    verbose=False 
                )
        
                self.y_pred = self.reg.predict(self.X_test)
                self.score = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
                fold_scores.append(self.score)
        
            mean_score = np.mean(fold_scores)
            all_scores.append((params, mean_score))
        
            if mean_score < best_score:
                best_score = mean_score
                self.best_params = params
        
        print(f"Best Parameters: {self.best_params}")
        print(f"Best Error: {best_score}")
        self.final_model = xgb.XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    objective='reg:squarederror',  # Ensure correct objective
    early_stopping_rounds=50,     # Optional: Early stopping for further training
    **self.best_params                  # Pass the best parameters
)
        self.final_model.fit(self.X_train, self.y_train,
                    eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                    verbose=False )
        return self.best_params
    def graph_future(self):
        future = pd.date_range('2023-12-31','2024-12-31', freq='d')
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True
        self.df['isFuture'] = False
        df_and_future = pd.concat([self.df, future_df])
        df_and_future = create_features(df_and_future)
        future_w_features = df_and_future.query('isFuture').copy()
        future_w_features['weekofyear'] = future_w_features['weekofyear'].astype('int32')
        future_w_features['pred'] = self.final_model.predict(future_w_features[self.FEATURES])
        self.df['pred'] = self.df['Units_Sold']
        combined_with_pred = pd.concat([self.df, future_w_features], axis = 0)
        cutoff_date = pd.Timestamp('2023-12-31')

        past_data = combined_with_pred[combined_with_pred.index <= cutoff_date]
        future_data = combined_with_pred[combined_with_pred.index > cutoff_date]

        plt.figure(figsize=(10, 5))
        past_data['pred'].plot(color='blue', ms=1, lw=1, label='Past')
        future_data['pred'].plot(color='red', ms=1, lw=1, label='Future')
        
        plt.title('Past + Future Predictions')
        plt.legend()
        plt.show()

        