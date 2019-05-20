from mlstudiosdk.modules.components.component import LComponent
from mlstudiosdk.modules.algo.data import Table, Domain
from mlstudiosdk.modules.utils.itemlist import MetricFrame
from mlstudiosdk.modules.utils.metricType import MetricType
from mlstudiosdk.modules.algo.evaluation import Results
from mlstudiosdk.modules.components.utils.orange_table_2_data_frame import table2df
from mlstudiosdk.exceptions.exception_base import Error
from mlstudiosdk.modules.algo.data.variable import DiscreteVariable, ContinuousVariable
from mlstudiosdk.modules.components.settings import Setting

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class AnomalyDetector(LComponent):
    category = 'Algorithm'
    name = 'Anomaly Detection'
    title = 'Anomaly Detection'
    inputs = [("Train Data", Table, "set_traindata")]
    outputs = [("Annotated Data", Table)]

    outliers_fraction = Setting(0.1, {"type": "number", "minimum": 0, "exclusiveMinimum": True})

    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None
        self.num_anomaly = None
        self.features = None
        self.train_col = None

        
    def set_traindata(self, data):
        self.train_data = data
        
        
    def set_numanomaly(self, data):
        self.num_anomaly = data

    def run(self):
        df = table2df(self.train_data)
        
        outliers_fraction = self.outliers_fraction # num_anomalies/len(df)
        print('Outlier Fraction: ', outliers_fraction)

        model_df = df.copy()
        ### label encode
        enc = preprocessing.LabelEncoder()
        for col in model_df.columns:
            if model_df[col].dtype not in ['int64', 'float64']:
                model_df[col] = enc.fit_transform(model_df[col])

        ### standardization
        min_max_scaler = preprocessing.StandardScaler()
        X = min_max_scaler.fit_transform(model_df)

        ### Initialize algos
        anomaly_algorithms = [
            ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction, random_state=111)),
            ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                                 random_state=42)),
            ("Local Outlier Factor", LocalOutlierFactor(
                n_neighbors=35, contamination=outliers_fraction))]


        results = pd.DataFrame()
        for name, algo in anomaly_algorithms:
            if name == "Local Outlier Factor":
                y_pred = algo.fit_predict(X)
            else:
                y_pred = algo.fit(X).predict(X)

            results[name] = y_pred

        ### Do majority vote
        combined_result = results.apply(lambda row: row.value_counts().idxmax(), axis=1)
        combined_result[combined_result==1] = 0
        combined_result[combined_result == -1] = 1

        metas = DiscreteVariable('anomaly_index',values=('Normal','Noise'))

        domain = self.train_data.domain
        attributes, classes = domain.attributes, domain.class_vars
        meta_attrs = domain.metas
        classes = [metas]

        domain = Domain(attributes, classes, meta_attrs)
        new_table = Table.from_table(domain, self.train_data)

        predict_value = combined_result.tolist()
        new_table.get_column_view(metas)[0][:] = predict_value

        self.send("Annotated Data", new_table)

    def rerun(self):
        self.run()
        print('.................')
    