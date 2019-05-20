from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train_input':{
        'datapath': 'PJME_hourly_train.csv',
        'label': 'PJME_MW'
    },
    'test_input':{
        'datapath': 'PJME_hourly_test.csv',
        'label': 'PJME_MW'
    },
    'ts':{

    },
    'output':{
        "filename": 'a.csv',
    }
}

solution.set_params(params)
solution.run()

params = {
    'train_input':{
        'datapath': 'PJME_hourly_train.csv',
        'label': 'PJME_MW'
    },
    'test_input':{
        'datapath': 'PJME_hourly_test_no_label.csv'
    },
    'ts':{

    },
    'output':{
        "filename": 'b.csv',
    }
}

solution.set_params(params)
solution.run(Mode.Test)