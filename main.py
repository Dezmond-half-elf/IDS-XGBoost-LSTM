from experiments import common_pipeline_run, features_tweaking_run, dataset_tweaking_run

if __name__ == '__main__':
    common_pipeline_run('data\KDDTrain+.txt', 'data\KDDTest+.txt')

    #experiment_result = features_tweaking_run(list(range(15, 31, 1)), 'data\KDDTrain+.txt', 'data\KDDTest+.txt')

    #experiment_result = dataset_tweaking_run('data\KDDTrain+.txt', 'data\KDDTest+.txt')
