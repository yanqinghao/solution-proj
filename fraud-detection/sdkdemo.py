from solution_new import AnomalyDetector
from mlstudiosdk.modules.scheme.mode import Mode

solution = AnomalyDetector()

params = {
    'train_input':{
        'datapath': 'sample_data.csv'
    },
    'algorithm':{
        'outliers_fraction' : 0.1
    },
    'output':{
        "filename": 'anomaly.csv'
    }
}

solution.set_params(params)
solution.run()

solution.run(Mode.Test)
