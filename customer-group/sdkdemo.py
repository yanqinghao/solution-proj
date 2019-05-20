from solution import Solution
# from mlstudiosdk.modules.scheme.mode import Mode

solution = Solution()

params = {
    'data_input':{
        'datapath': 'sample.csv'
    },
    'Customers_fcluster':{
    },
    'output':{
        "filename": 'result.csv'
    }
}

solution.set_params(params)
solution.run()

print('finish')
