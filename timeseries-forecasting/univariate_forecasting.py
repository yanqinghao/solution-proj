#!/usr/bin/python
import pandas as pd
import numpy as np
import fbprophet as fbpro
import sklearn.metrics as skm
import math
import dill as pickle
import os

# Abhishek - TODO list:
    # 1. Add capability to set change point dates manually: m = Prophet(changepoints=['2014-01-01'])
    # 2. Add ability to configure outlier dates/date intervals per the method in
    #    https://facebookincubator.github.io/prophet/docs/outliers.html
    # 3. Add ability to manipulate with uncertainty intervals and do highly computationally-intencive Markov Chain
    #    sampling per https://facebookincubator.github.io/prophet/docs/uncertainty_intervals.html
    # 4. Add capablity for Multivariate timeseries prediction - This can be done By RNN LSTM or by adding regressors for fbprophet model or by using ArimaX
    #    https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#additional-regressors
    # 5. Add AutoML capablity for choosing whether to do log transformation or not, Parameters :do_log_transform = 1/0. Setting Change Point Value(_changepoint_prior_scale)
    # 6. Add Capablity to check whether Weekly and yearly seasonalities are there or not 
    #    By default it is set to True,needs change 
    # 7. Use rmse,mae,smape for fitness function to optimize for evolutionary search for autoML

def save_model(model, path):
    '''Saving ML model for later use
    '''
    filename = os.path.join(path, 'model_v1.pk')
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(path):
    '''Loading ML model for later use
    '''
    filename = os.path.join(path, 'model_v1.pk')
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

class ProphetModeller(object):
    """
        This class is a wrapper over TS prediction algorithms provided by Prophet
        Prophet is Facebook-developed open-source lib for producing high quality forecasts for
        time series data that has multiple seasonality with linear or non-linear growth
        (https://github.com/facebookincubator/prophet)

        Set of product documentation on Prophet is available at
        https://facebookincubator.github.io/prophet/docs/quick_start.html
    """    
    def __init__(self, do_log_transform, frequency, turn_negative_forecasts_to_0=1, capacity_used=0,
                 changepoint_prior_scale=0.05, holidays_prior_scale=10.0, seasonality_prior_scale=10.0,
                 yearly_seasonality='auto', weekly_seasonality='auto', holidays=pd.DataFrame([])):
        """ Constructor for the class
            :param df_data a pandas data frame with TS data - it must have exactly two columns with
                   constant names - ['ds', 'y']
            :param future_periods - the int value indicating the number of TS data points to forecast ahead
            :param do_log_transform - 0/1 switch indicating if df_data has to be log transformed before prediction- Needs AutoML
        """
        # self._df_data = df_data
        # self._future_periods = future_periods
        self._do_log_transform = do_log_transform

        # fake empty placeholder forecast DF, to be properly populated at predict time
        self._forecast = pd.DataFrame(columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

        # fake empty placeholder holidays DF, to be set separately via respective property
        # before predict() call, if needed
        # see more details on holidays at https://facebookincubator.github.io/prophet/docs/holiday_effects.html
        self._holidays = holidays
        if len(self._holidays)!=0:
            self._holidays_set = 1
        else:
            self._holidays_set = 0# if changed to 1 in the holidays setter, it will trigger model setup with holidays

        self._capacity_used = capacity_used # if changed to 1 in set_capacity, it will result in "logistic" growth set in the model

        # below are default values for other class members that can be later altered via read-write properties
        self._turn_negative_forecasts_to_0 = turn_negative_forecasts_to_0 # this is 0/1 flag
        self._changepoint_prior_scale = changepoint_prior_scale   # default value is set by Facebook team

        # default value set by Facebook team (see https://facebookincubator.github.io/prophet/docs/holiday_effects.html
        self._holidays_prior_scale = holidays_prior_scale

        self._seasonality_prior_scale = seasonality_prior_scale #default value set by Facebook team

        self._yearly_seasonality = yearly_seasonality # set False via property if the TS does not have yearly seasonality
        self._weekly_seasonality = weekly_seasonality # set False via property if the TS does not have weekly seasonality

        # freq: Any for pd.date_range, such as 'D' or 'M'
        # see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases for more info
        self._frequency = frequency # default set by Facebook team to Day

        self._forecast_model = ''

    def set_capacity(self, df_data, list_of_capacity_modifiers):
        """ this will set capacity modifier, assuming the y values in the df_data could be bigger if all
            potential capacity (like market size, physical limit of generated power, etc. ) could be exhausted
            We must specify the carrying capacity in a column cap.
            It would usually be set using data or expertise about the market size or other limits related to y values.

            Note: if you like to set capacity modifiers, it should be done before you call predict()
        """
        self.df_data['cap'] = list_of_capacity_modifiers
        self._capacity_used = 1

    def fit(self,df_data):
        """ This is the fitness function 
        """
        if np.mean(df_data['y'].values) == 0:
            print("[ProphetModeller.predict] Waring: Input for Prophet is ts with zero values - no forecast will be created")
            #Time series with zero values result in no forecast
        else:
            df = df_data # make a local copy of input data ...

            if self._do_log_transform:
                df['y'] = np.log(df['y'])
            if self._capacity_used:
                if self._holidays_set:
                    model = fbpro.Prophet(growth='logistic',
                                      changepoint_prior_scale=self._changepoint_prior_scale,
                                      holidays=self._holidays,
                                      holidays_prior_scale=self._holidays_prior_scale,
                                      yearly_seasonality=self._yearly_seasonality,
                                      weekly_seasonality=self._weekly_seasonality,
                                      seasonality_prior_scale=self._seasonality_prior_scale)
                else:
                    model = fbpro.Prophet(growth='logistic',
                                      changepoint_prior_scale=self._changepoint_prior_scale,
                                      yearly_seasonality=self._yearly_seasonality,
                                      weekly_seasonality=self._weekly_seasonality,
                                      seasonality_prior_scale=self._seasonality_prior_scale)
            else:
                if self._holidays_set:
                    model = fbpro.Prophet(changepoint_prior_scale=self._changepoint_prior_scale,
                                      holidays=self._holidays,
                                      holidays_prior_scale=self._holidays_prior_scale,
                                      yearly_seasonality=self._yearly_seasonality,
                                      weekly_seasonality=self._weekly_seasonality,
                                      seasonality_prior_scale=self._seasonality_prior_scale)
                else:
                    model = fbpro.Prophet(changepoint_prior_scale=self._changepoint_prior_scale,
                                      yearly_seasonality=self._yearly_seasonality,
                                      weekly_seasonality=self._weekly_seasonality,
                                      seasonality_prior_scale=self._seasonality_prior_scale)
            model.fit(df)
            self._forecast_model=model

    def get_model(self):
        return self

    def make_future_dataframe(self, start, end, freq='D'):
        """Simulate the trend using the extrapolated generative model.

        Parameters
        ----------
        periods: Int number of periods to forecast forward.
        freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
        include_history: Boolean to include the historical dates in the data
            frame for predictions.

        Returns
        -------
        pd.Dataframe that extends forward from the end of self.history for the
        requested number of periods.
        """
        last_date = start
        dates = pd.date_range(start=last_date, end=end, freq=freq)
        return pd.DataFrame({'ds': dates})

    def do_forecast(self, start, end, debug=0):
        '''Forecasting happens here
            a.Reloading the mode(this function has issues)
            b.Forecasting
        '''
        #Testing on test data to find the accuracy by reloading the model
        # filename = self.filepath
        # with open(filename ,'rb') as f:
        #     prophet_wrapper_object = pickle.load(f)

        future = self.make_future_dataframe(start, end, freq=self._frequency)

        # this will return data for columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper'] always
        # ds - datetime stamp of the point in observations
        # yhat - prediction
        # 'yhat_lower' and 'yhat_upper' - uncertainty intervals
        #
        # optionally, additional cols could be added with the impact of each holiday season, if holidays configured
        # in this case, value of 'holiday' col in each holiday df row will be the name of the impact col for such
        # a holiday
        self._forecast = self._forecast_model.predict(future)

        if self._turn_negative_forecasts_to_0:
            self._forecast.loc[self._forecast.yhat < 0, 'yhat'] = 0
            self._forecast.loc[self._forecast.yhat_lower < 0, 'yhat_lower'] = 0
            self._forecast.loc[self._forecast.yhat_upper < 0, 'yhat_upper'] = 0

        if self._do_log_transform:
            self._forecast['yhat'] = np.exp(self._forecast['yhat'])
            self._forecast['yhat_lower'] = np.exp(self._forecast['yhat_lower'])
            self._forecast['yhat_upper'] = np.exp(self._forecast['yhat_upper'])

        if debug:
            print("Forecasted values:")
            print(self._forecast.tail())

    def get_forecast_only(self):
        """ this will return the subset of self._forecast without historic data """
        df_forecast_only = self._forecast
        print(df_forecast_only)
        return df_forecast_only

    # validation methods / metrics

    def rmse(self, pred, targets):
        """ This method validates RMSE of the predicted values vs. the targets from the validation set

        :param targets - a list of target (true) values from the validation set

        :return calculated RMSE or -1 in case the length of targets list does not equal to self._future_periods
        """

        rmse = -1

        # if len(targets) != self._future_periods:
        #     print("[ProphetModeller.rmse] invalid target length: ", len(targets),
        #           ", expected length: ", self._future_periods)
        # else:
        y_pred = pred

        rmse = math.sqrt(skm.mean_squared_error(targets, y_pred))

        return rmse

    def mean_absolute_percentage_error(self, pred, targets):
        """ This method validates MAPE of the predicted values vs. the targets from the validation set

            :param targets - a list of target (true) values from the validation set

            :return calculated MAPE or -1 in case the length of targets list does not equal to self._future_periods
        """
        mape = -1

        # if len(targets) != self._future_periods:
        #     print("[ProphetModeller.mean_absolute_percentage_error] invalid target length: ", len(targets),
        #           ", expected length: ", self._future_periods)
        #
        # else:
        y_pred = pred
        y_true = targets

            ## Note: does not handle mix 1d representation
            # if _is_1d(y_true):
            #    y_true, y_pred = _check_1d_array(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mape

    def smape(self, pred, targets):
        """ This method validates SMAPE of the predicted values vs. the targets from the validation set

            :param targets - a list of target (true) values from the validation set

            :return calculated SMAPE or -1 in case the length of targets list does not equal to self._future_periods
        """
        smape = -1
        # if len(targets) != self._future_periods:
        #     print("[ProphetModeller.smape] invalid target length: ", len(targets),
        #           ", expected length: ", self._future_periods)
        #
        # else:
        y_pred = pred
        y_true = targets

        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0
        smape = np.mean(diff)

        return smape

    # properties

    @property
    def forecast(self):
    
        # this returns the entire forecast df, which contains the origianl TS plus self._future_periods forecasted
        # periods in future - if you need to get the forecasted new values, there is a separate method for that above
        print("getting forecast")
        return self._forecast

    @property
    def turn_negative_forecasts_to_0(self):
        return self._turn_negative_forecasts_to_0

    @turn_negative_forecasts_to_0.setter
    def turn_negative_forecasts_to_0(self, value):
        if value != True:
            self._turn_negative_forecasts_to_0 = False
        else:
            self._turn_negative_forecasts_to_0 = True

    # Parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many
    # changepoints, small values will allow few changepoints.
    # see more on the statistical meaning of this parameter at
    # https://facebookincubator.github.io/prophet/docs/trend_changepoints.html, Adjusting trend flexibility
    @property
    def changepoint_prior_scale(self):
        return self._changepoint_prior_scale

    @changepoint_prior_scale.setter
    def changepoint_prior_scale(self, value):
        self._changepoint_prior_scale = value # should be float value between 0 and 1

    @property
    def holidays(self):
        return self._holidays

    @holidays.setter
    def holidays(self, df_holidays):
        # df_holidays must be created in adherance to the convention explained at
        # https://facebookincubator.github.io/prophet/docs/holiday_effects.html, Modelling holidays
        self._holidays = df_holidays
        self._holidays_set = 1  # if changed to 1 in the holidays setter, it will trigger model setup with holidays

    # Parameter modulating the strength of the holiday components model.
    @property
    def holidays_prior_scale(self):
        return self._holidays_prior_scale

    @holidays_prior_scale.setter
    def holidays_prior_scale(self, value):
        # value is non-negative float
        # detailed statistical meaning of this parameter is explained in
        # https://facebookincubator.github.io/prophet/docs/holiday_effects.html,
        # 'Prior scale for holidays and seasonality' section
        self._holidays_prior_scale = value

    # seasonality flags
    @property
    def yearly_seasonality(self):
        return self._yearly_seasonality
    @yearly_seasonality.setter
    def yearly_seasonality(self, boolean_value):
        self._yearly_seasonality = boolean_value

    @property
    def weekly_seasonality(self):
        return self._weekly_seasonality

    @weekly_seasonality.setter
    def weekly_seasonality(self, boolean_value):
        self._weekly_seasonality = boolean_value

    # Parameter modulating the strength of the
    # seasonality model. Larger values allow the model to fit larger seasonal
    # fluctuations, smaller values dampen the seasonality.
    @property
    def seasonality_prior_scale(self):
        return self._seasonality_prior_scale

    @seasonality_prior_scale.setter
    def seasonality_prior_scale(self, value):
        self._seasonality_prior_scale = value

    # Any valid frequency for pd.date_range, such as 'D' or 'M'.
    # Full list of valid frequencies (or offset aliases) is available at
    # http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value

    @property
    def forecast_model(self):
        return self._forecast_model

    @forecast_model.setter
    def forecast_model(self,value):
        self._forecast_model =value
def main():
    '''
    This is main module modeule which acts as a driver for the program. 
    This has to be updated to include AutoML components. Refer above for TODO list
    It currently involves:
        a. Reading in the data from the source
        b. Creating an object of FB Phophet wrapper class ,Parameters(data, future prediction count, do_log_transform)
        c. Calling Pridict function to fit the model
        d. Saving the model built
        e. Obtaining the forecast for 3 data
        f. Calculation of Errors for prediction.
    '''
    
    print("Reading the data and doing ETL")
    df_raw = pd.read_csv('ts_data.csv', nrows = 11856)
    df_raw=df_raw.rename( columns={"Datetime": "ds", "Count": "y"})
    #creating time index
    df_raw.index = pd.to_datetime(df_raw.ds,format='%d-%m-%Y %H:%M')
    #grouping records of a day and calculating and using mean of counts for that day
    df_raw=df_raw.resample('D').mean()
    #getting rid of unnecessary columns
    df_raw=df_raw.drop(columns=['ID'])
    #creating 'ds' column out of index
    df_raw['ds'] = df_raw.index
    train=df_raw[0:433]
    test=df_raw[433:]
    
    
    prop = ProphetModeller(train,3,0)
    print("Fitting the forecasting model..")
    prop.fit()
    print("Saving forcasting model..")
    prop.save_model()
    print("Forecasting future values..")
    prop.do_forecast(3)
    
    prop.get_forecast_only()

    #Errors
    print("calculationg forecasting errors..")
    
    #extracting 3 values from testing data
    future_values =list(test[:3]['y'].values)
    mae=prop.mean_absolute_percentage_error(future_values)
    rmse=prop.rmse(future_values)
    smape=prop.smape(future_values)

    print("ERRORS: MAE = %s, RMSE = %s, SMAPE = %s"%(mae,rmse,smape))

if __name__ == "__main__":
    print("Starting the program..")
    main()
    print("Ending the program..")
