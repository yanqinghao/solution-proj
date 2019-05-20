# -*- coding: utf-8 -*-
"""
@author: yanqh2 & Abhishek
"""

import mlstudiosdk.modules
import mlstudiosdk.solution_gallery
from mlstudiosdk.modules.components.component import LComponent
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import df2table
from mlstudiosdk.modules.algo.data import Domain, Table
from mlstudiosdk.modules.algo.evaluation import Results
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable
from mlstudiosdk.modules.utils.itemlist import MetricFrame
from mlstudiosdk.modules.utils.metricType import MetricType
from sklearn.model_selection import train_test_split
from mlstudiosdk.exceptions.exception_base import Error
from mlstudiosdk.modules.components.settings import Setting
import random
import string
import shutil
import os
import glob #find file directories and files
import pandas as pd
import numpy as np
from univariate_forecasting import ProphetModeller,load_model,save_model
from workalendar.usa import UnitedStates
from workalendar.asia import HongKong
from tempfile import NamedTemporaryFile, TemporaryDirectory


class TimeSeries(LComponent):
    category = 'Algorithm'
    name = "TimeSeries"
    #create the current file path
    inputs = [("Train Data", mlstudiosdk.modules.algo.data.Table, "set_traindata"),
              ("Test Data", mlstudiosdk.modules.algo.data.Table, "set_testdata")]
    outputs = [("News", mlstudiosdk.modules.algo.data.Table),
               ("Predictions", Table),
               ("Evaluation Results", mlstudiosdk.modules.algo.evaluation.Results),
               ("Columns", list),
               ("Metas", list),
               ("Metric Score", MetricFrame),
               ("Metric", MetricType)]
    # None : no holiday_effect China : HongKong holiday_effect  US : US holiday_effect
    holiday_effect = Setting("None", {"type": "string"})
    # label log trasform
    log_transform = Setting(False, {"type": "boolean"})
    # negative prediction to 0
    turn_negative_forecasts_to_0 = Setting(True, {"type": "boolean"})
    # todo make time_name as an attribute, instead of 0 col of data
    # time_name = Setting(None, {"type": "string"})

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None
        # self.file_path = None
        # self.temporary_dir = None

    def set_traindata(self, data):
        self.train_data = data

    def set_testdata(self, data):
        self.test_data = data
    
    # def set_file_path(self):
    #     ran_str = ''.join(random.sample(string.ascii_letters + string.digits, random.randint(5, 12)))
    #     self.file_path=os.path.join(os.getcwd(),ran_str)
    #     while os.path.exists(self.file_path):
    #         ran_str = ''.join(random.sample(string.ascii_letters + string.digits, random.randint(5, 12)))
    #         self.file_path=os.path.join(os.getcwd(),ran_str)
    #     os.makedirs(self.file_path)

    def files(self,curr_dir = '.', ext = 'fasttext_*.txt'):
        """Files in the current directory"""
        for i in glob.glob(os.path.join(curr_dir, ext)):
            yield (i)

    def remove_files(self,rootdir, ext, show = False):
        """Delete the matching files in the rootdir directory"""
        for i in self.files(rootdir, ext):
             # if show:
                # print('如下文件已被删除:',i)
            os.remove(i)

    def merge_data(self,origin, result):
        # result table to origin table's meta
        domain = origin.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        new_domain = Domain(attributes, classes+result.domain.variables, meta_attrs)
        new_table = Table.from_table(new_domain, origin)
        for attr in result.domain:
            new_table.get_column_view(attr)[0][:] = result.get_column_view(attr)[0][:]
        return new_table

    def run(self):
        # self.set_file_path()
        train_data = table2df(self.train_data)
        test_data = table2df(self.test_data)
        self.label_name = self.train_data.domain.class_var.name
        self.time_name = self.train_data.domain.attributes[0].name
        train_data = train_data[[self.time_name,self.label_name]]
        test_data = test_data[[self.time_name, self.label_name]]
        train_data = train_data.rename(index=str, columns={self.label_name: "y", self.time_name: "ds"})
        test_data = test_data.rename(index=str, columns={self.label_name: "y", self.time_name: "ds"})
        train_data = train_data.drop_duplicates('ds')
        if len(self.train_data.domain.variables) - len(self.train_data.domain.class_vars) > 1 \
                or len(self.train_data.domain.attributes) > 1:
            raise Error("too many features")
        try:
            train_data['ds'] = pd.to_datetime(train_data['ds'])
            test_data['ds'] = pd.to_datetime(test_data['ds'])
        except:
            raise Error("datetime type error")
        train_data = train_data.sort_values(by=['ds'])
        test_data = test_data.sort_values(by=['ds'])
        # train_data_time = pd.date_range(start=train_data[self.time_name].min(),
        #                            end=train_data[self.time_name].max(),freq='H')
        delta_t = pd.DataFrame(train_data[['ds']].iloc[1:].values-
                               train_data[['ds']].iloc[:-1].values,columns=['delta'])
        if delta_t.groupby('delta')['delta'].count()[0]/len(delta_t)>0.5:
            freq_train = pd.unique(delta_t['delta'])[0]
        else:
            raise Error('Too many time series missing values')
        freq_para = str(int(freq_train)) + 'N'
        test_data = test_data.dropna()

        do_log_transform = self.log_transform
        frequency = freq_para
        capacity_used = 0
        turn_negative_forecasts_to_0 = self.turn_negative_forecasts_to_0

        holtemp = []
        if self.holiday_effect == 'China':
            for i in list(set(train_data.ds.dt.year)):
                cal = HongKong()
                test_hld = cal.holidays(i)
                hol = pd.DataFrame(test_hld, columns=['ds','holiday'])
                hol['ds'] = pd.to_datetime(hol['ds'])
                holtemp.append(hol)
        elif self.holiday_effect == 'US':
            for i in list(set(train_data.ds.dt.year)):
                cal = UnitedStates()
                test_hld = cal.holidays(i)
                hol = pd.DataFrame(test_hld, columns=['ds','holiday'])
                hol['ds'] = pd.to_datetime(hol['ds'])
                holtemp.append(hol)
        if len(holtemp)==0:
            holidays = pd.DataFrame([])
        else:
            holidays = pd.concat(holtemp,axis=0)

        prop = ProphetModeller(do_log_transform=do_log_transform, frequency=frequency,
                               capacity_used=capacity_used, turn_negative_forecasts_to_0=turn_negative_forecasts_to_0,
                               holidays = holidays)

        print("Fitting the forecasting model..")
        train_data['ds'] = train_data['ds'].apply(lambda x: x.tz_convert(None))
        test_data['ds'] = test_data['ds'].apply(lambda x: x.tz_convert(None))
        prop.fit(train_data)
        print("Saving forcasting model..")
        # save_model(prop.get_model(), self.file_path)
        self.model = prop
        starttime = test_data['ds'].min()
        endtime = test_data['ds'].max()
        prop.do_forecast(start=starttime, end=endtime)
        pre_data = prop.get_forecast_only()
        result_df = pd.merge(test_data,pre_data,on='ds')

        mae_score = prop.mean_absolute_percentage_error(result_df['yhat'], result_df['y'])
        rmse_score = prop.rmse(result_df['yhat'], result_df['y'])
        smape = prop.smape(result_df['yhat'], result_df['y'])
        print("ERRORS: MAE = %s, RMSE = %s, SMAPE = %s" % (mae_score, rmse_score, smape))

        """-----------calculate probility-------------------------------"""
        metas = [ContinuousVariable('predict'),
                 ContinuousVariable('predict lower'),
                 ContinuousVariable('predict upper')]

        cols = result_df[['yhat', 'yhat_lower', 'yhat_upper']].values

        tbl = np.array(cols)
        res = Table.from_numpy(Domain(metas), tbl)
        final_result = self.merge_data(self.test_data,res)

        results = None
        N = len(final_result)  # note that self.test_data is orange Table
        results = Results(self.test_data, store_data=True)
        results.folds = None
        results.row_indices = np.arange(N)
        results.actual = np.array(test_data['y'])
        results.predicted = np.array([np.array(result_df['yhat'])])
        results.probabilities = np.array(result_df[['yhat_lower', 'yhat_upper']])

        print("predicted", results.predicted)
        print("actual", results.actual)
        results.learner_names = ['Multiple_text_Classifier']
        # self.send("Predictions", predictions)
        self.send("Evaluation Results", results)
        metric_frame = MetricFrame([[mae_score], [rmse_score]], index=["MAE", "RMSE"],
                                   columns=[MetricType.MEAN_ABSOLUTE_ERROR.name])
        self.send("Metric Score", metric_frame)
        self.send("Columns", cols)
        self.send("Metas", metas)
        self.send("Metric", MetricType.ROOT_MEAN_SQUARED_ERROR)
        # print(MetricType.ACCURACY)
        print("metric_frame",metric_frame)
        print("result:",results.predicted, results.predicted.shape)
        self.send('News', final_result)

    def rerun(self):
        # self.set_file_path()
        test_data = table2df(self.test_data)
        if self.time_name not in test_data.columns:
            raise Error("Can not find time column")
        if self.label_name in test_data.columns:
            test_data = test_data[[self.time_name, self.label_name]]
            test_data = test_data.rename(index=str, columns={self.label_name:
                                                                 "y", self.time_name: "ds"})
        else:
            test_data = test_data[[self.time_name]]
            test_data = test_data.rename(index=str, columns={self.time_name: "ds"})
        try:
            test_data['ds'] = pd.to_datetime(test_data['ds'])
        except:
            raise Error("datetime type error")
        test_data = test_data.sort_values(by=['ds'])
        # model = load_model(self.file_path)
        model = self.model
        test_data['ds'] = test_data['ds'].apply(lambda x: x.tz_convert(None))
        starttime = test_data['ds'].min()
        endtime = test_data['ds'].max()
        model.do_forecast(start=starttime, end=endtime)
        pre_data = model.get_forecast_only()
        result_df = pd.merge(test_data, pre_data, on='ds')
        """-----------calculate probility-------------------------------"""
        metas = [ContinuousVariable('predict'),
                 ContinuousVariable('predict lower'),
                 ContinuousVariable('predict upper')]

        cols = result_df[['yhat', 'yhat_lower', 'yhat_upper']].values

        tbl = np.array(cols)
        res = Table.from_numpy(Domain(metas), tbl)
        final_result = self.merge_data(self.test_data, res)

        self.send("Columns", cols)
        self.send("Metas", metas)
        self.send("Metric", None)
        self.send('News', final_result)
        # shutil.rmtree(self.file_path)

    def __getstate__(self):
        odict = super(TimeSeries, self).__getstate__()
        odict["train_data"] = None
        odict["test_data"] = None
        # odict["temporary_dir"] = None
        # odict["file_path"] = None
        return odict
