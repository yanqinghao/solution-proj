from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train_input':{
        'datapath': 'fasttext_training_1115.csv',
        'label': 'label',
        'column_id':'id'
    },
    'test_input':{
        'datapath': 'fasttext_test_1115.csv',
        'label': 'label',
        'column_id':'id'
    },
    'fasttext':{

    },
    'output':{
        "filename": 'a.csv',
    }
}

solution.set_params(params)
solution.run()

params = {
    'train_input':{
        'datapath': 'fasttext_training_1115.csv',
        'label': 'label',
        'column_id':'id'
    },
    'test_input':{
        'datapath': 'fasttext_test_unlabel.csv',
        'column_id':'id'
    },
    'fasttext':{

    },
    'output':{
        "filename": 'a.csv',
    }
}

solution.set_params(params)
solution.run(Mode.Test)