# pharmacy sales forecasting project

# I built time-series forecasting models in Python (using Prophet and XG Boost), fine-tuned them with grid search cross-validation, to generate insight from a pharmacy sales dataset. I conveyed my findings in a slide deck to present actionable business action to improve sales.

# Description of each file:
healthcare_product_sales.csv -> dataset of pharmacy sales
Pharmacy_TimeSeries_Prophet.ipynb -> jupyter notebook containing construction, fine-tuning, and usage of Prophet model
TimeSeriesClass.py -> python file containing the TimeSeriesProphet and ProphetCV classes, classes that are utilized in Pharmacy_TimeSeries_Prophet.ipynb to streamline model creation, fine-tuning, data analysis, and visualization
Pharmacy_TimeSeries_XGBoost.ipynb -> jupyter notebook containing construction, fine-tuning, and usage of XGBoost
TimeSeriesXGClass.py -> python file containing the XG_TimeSeries and XGBoostCV classes, classes that are utilized in Pharmacy_TimeSeries_XGBoost.ipynb to streamline model creation, fine-tuning, data analysis, and visualization
pharmacy_timeseries_deliverable.pptx -> slide deck containing business insights and actionable recommendations from the models.
# How to run
To run, the following python libraries must be installed: pandas, numpy, matplotlib, seaborn, xgboost, sklearn, prophet. 

# Summary of Findings
Overall, findings indicate relatively healthly growth in all drug categories for the pharmacy. While there are clear seasonal trends for pain relievers, cold remedies, vitamins, and skin care, the seasonal trends for first aid are less clear. However, slight negative trends for cold remedies and skin care signify the need to anticipate seasonal peaks in sales for these categories and change business strategy.

# Acknowledgements
This project was inspired by Rob Mulla's videos on time series. His Kaggle notebooks can be found here:
Prophet: https://www.kaggle.com/code/robikscube/time-series-forecasting-with-prophet-yt
XGBoost(Part 1): https://www.kaggle.com/code/robikscube/time-series-forecasting-with-machine-learning-yt
XGBoost(Part 2): https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost/notebook

