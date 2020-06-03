## Import Section
# Importing libraries required for the project
import pandas as pd
import numpy as np
import os
import re
import datetime
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
import pickle
from joblib import dump, load
from pathlib import Path
from tkinter import filedialog
import tkinter
from pandas.tseries.holiday import USFederalHolidayCalendar

def folderSelect():
    root = tkinter.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

def generating_new_features(hour):
    hour["hr2"] = hour["hr"]
    hour["season2"] = hour["season"]
    hour["temp2"] = hour["temp"]
    hour["hum2"] = hour["hum"]
    hour["weekday2"] = hour["weekday"]
    # Change dteday to date time
    hour["dteday"] = pd.to_datetime(hour["dteday"])
    # Convert the data type to eithwe category or to float
    int_hour = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    for col in int_hour:
        hour[col] = hour[col].astype("category")
    # A logarithmic transformation is applied deal with skewness
    hour["windspeed"] = np.log1p(hour.windspeed)
    # Feature Engineering
    # Rentals during average workingday assumed to be 09h00 to 17h00
    hour["IsOfficeHour"] = np.where(
        (hour["hr2"] >= 9) & (hour["hr2"] < 17) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsOfficeHour"] = hour["IsOfficeHour"].astype("category")
    # Rentals during daytime assumed to be 06h00 to 22h00
    hour["IsDaytime"] = np.where((hour["hr2"] >= 6) & (hour["hr2"] < 22), 1, 0)
    hour["IsDaytime"] = hour["IsDaytime"].astype("category")
    # Rentals during morning rush hour assumed to be 06h00 to 10h00
    hour["IsRushHourMorning"] = np.where(
        (hour["hr2"] >= 6) & (hour["hr2"] < 10) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushHourMorning"] = hour["IsRushHourMorning"].astype("category")
    # Rented during evening rush hour assumed to be 15h00 to 19h00
    hour["IsRushHourEvening"] = np.where(
        (hour["hr2"] >= 15) & (hour["hr2"] < 19) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushHourEvening"] = hour["IsRushHourEvening"].astype("category")
    # Rentals during busiest season
    hour["IsHighSeason"] = np.where((hour["season2"] == 3), 1, 0)
    hour["IsHighSeason"] = hour["IsHighSeason"].astype("category")
    # Binning variables temp, atemp, hum in 5 equally sized bins
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    hour["temp_binned"] = pd.cut(hour["temp2"], bins).astype("category")
    hour["hum_binned"] = pd.cut(hour["hum2"], bins).astype("category")
    # Dropping duplicated rows used for feature engineering
    hour = hour.drop(columns=["hr2", "season2", "temp2", "hum2", "weekday2"])
    return hour

def get_season(mydate):
    """
    This function obtains the
    season of the year from
    the date parameter.
    """
    season = 0
    dayofyear = mydate.timetuple().tm_yday
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    if dayofyear in spring:
        season = 1
    elif dayofyear in summer:
        season = 2
    elif dayofyear in fall:
        season = 3
    else:
        season = 4
    return season

def fill_missing_features(myTestdf, train_features):
    """
    This function checks for the missing
    columns in the test dataset and fills the missing
    Columns with default value 0 
    """
    # Get missing columns in the training test
    missing_cols = set(train_features) - set(myTestdf.columns.values)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        myTestdf[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    return myTestdf

def prepare_train_data():
    """
    This function preprocess the data (hour and day)
    It also generates new features and prepare the train and test data
    """
    day = pd.read_csv(
        "./src/ie_bike_model/day.csv", index_col="instant", parse_dates=True
    )
    hour = pd.read_csv(
        "./src/ie_bike_model/hour.csv", index_col="instant", parse_dates=True
    )
    hour = generating_new_features(hour)
    hour["cnt"] = np.sqrt(hour.cnt)
    # creating duplicate columns for feature engineering
    hour = pd.get_dummies(hour)
    ## Set the Training Data and Test Data
    hour_train = hour.iloc[0:15211]
    hour_test = hour.iloc[15212:17379]
    # Modelling Stage
    train = hour_train.drop(columns=["dteday", "casual", "atemp", "registered"])
    test = hour_test.drop(columns=["dteday", "casual", "registered", "atemp"])
    # Separate the independent and target variable on the training data
    train_X = train.drop(columns=["cnt"], axis=1)
    train_y = train["cnt"]
    # Separate the independent and target variable on the test data
    test_X = test.drop(columns=["cnt"], axis=1)
    test_y = test["cnt"]
    return train_X, train_y, test_X, test_y


def train_and_persist():
    """
    This function engineers
    the features and trains
    the RandomForestRegressor.
    """
    train_X, train_y, test_X, test_y = prepare_train_data()
    ## Building RandomForestRegressor Model Using Best Parameters from GridSearch
    rf = RandomForestRegressor(
        max_depth=40,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42,
    )
    rf.fit(train_X, train_y)
    folder_selected = folderSelect()
    filename = os.path.join(folder_selected, "model.joblib")
    dump(rf, filename)

def predict(dict):
    """
    This function pickles the
    model and generates the test
    features by passing the dict
    parameter.
    """

    folder_selected = folderSelect()
    filename = os.path.join(folder_selected, "model.joblib")
    if filename is None:
        train_and_persist()
    else:
        with open(filename, "rb") as f:
            model = load(f)
    testDf = pd.DataFrame(dict, index=[0])
    testDf["hr"] = testDf["date"].dt.hour
    testDf["yr"] = testDf["date"].dt.year - testDf["date"].dt.year.min()
    testDf["mnth"] = testDf["date"].dt.month
    testDf["season"] = testDf["date"].map(get_season)
    testDf["weekday"] = testDf["date"].dt.weekday
    testDf["dteday"] = testDf["date"].dt.day
    testDf["dteday"] = pd.to_datetime(testDf["dteday"])
    cal = USFederalHolidayCalendar()
    holidays = pd.to_datetime(cal.holidays(start="2011-01-01", end="2011-06-30"))
    testDf["holiday"] = pd.to_datetime(testDf["date"]).dt.date in holidays
    testDf["workingday"] = pd.to_datetime(testDf["date"]).dt.date not in holidays
    testDf["holiday"] = testDf["holiday"].map(lambda x: 1 if x else 0)
    testDf["workingday"] = testDf["workingday"].map(lambda x: 1 if x else 0)
    t_max , t_min = 50, -8
    # This is for tempretaure normalization
    testDf["temp"] = (testDf["temperature_C"] - t_min)/(t_max-t_min)
    # We divide humidity by 100 to scale it between 0 and 1
    testDf["hum"] = testDf["humidity"]/100
    testDf = testDf.drop(columns=["temperature_C", "humidity"])
    # Convert the data type to eithwe category or to float
    testDf = generating_new_features(testDf)
    testDf = pd.get_dummies(testDf)
    # Finally start with Machine Learning
    test = testDf.drop(columns=["date", "dteday", "feeling_temperature_C"])
    # savedir = Path.home()
    # filename = os.path.join(savedir, "model.joblib")
    # with open(filename, "rb") as f:
    #     model = load(f)
    # print("done!")
    # f.close()
    train_X, train_y, test_X, test_y = prepare_train_data()
    train_features = train_X.columns.values
    test = fill_missing_features(test, train_features)
    pred = model.predict(test)
    pred = pred.astype(int)
    print(" Rounded predictions:\n", pred)
