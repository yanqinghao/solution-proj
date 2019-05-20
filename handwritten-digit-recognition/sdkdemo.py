from solution import Solution
from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'train input':{
        'datapath': 'train3.csv',
        'label': 'label'
    },
    'test input':{
        'datapath': 'test3.csv',
        'label': 'label'
    },
    'algorithm':{

    },
    'output':{
        "filename": 'a.csv',
    }
}
solution.set_params(params)
solution.run()
params = {
    'train input':{
        'datapath': 'train3.csv',
        'label': 'label'
    },
    'test input':{
        'datapath': 'test3unlabel.csv',
    },
    'algorithm':{

    },
    'output':{
        "filename": 'a.csv',
    }
}
solution.set_params(params)
solution.run(Mode.Test)
