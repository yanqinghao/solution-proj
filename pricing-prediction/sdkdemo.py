from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train input':{
        'datapath': 'data_samples.csv',
        'label': 'dol'
    },
    'test input':{
        'datapath': 'data_samples.csv',
        'label': 'dol'
    },
    'output':{
        "filename": 'prediction.csv'
    }
}

solution.set_params(params)
solution.run()


params = {
    'test input':{
        'datapath': 'data_samples2.csv'
    },
    'output':{
        "filename": 'prediction2.csv'
    }
}

solution.set_params(params)
solution.run(Mode.Test)
