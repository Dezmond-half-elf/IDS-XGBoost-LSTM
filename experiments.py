from sklearn.model_selection import train_test_split

import torch.optim as optim
import torch.nn as nn

from preprocessing import preprocessing, modified_preprocessing, file_reading
from feature_selection import xgb_create, xgb_accuracy_count, feature_select
from LSTM_train_test import LSTMmodel, fit, test


def common_pipeline_run(train_path, test_path):
    full_df, train_size = file_reading(train_path, test_path)
    x_train, x_test, y_train, y_test = preprocessing(full_df, train_size)
    x_train_, y_train_ = modified_preprocessing(full_df, train_size, normal_=0.9, probe_=1, u2r_=100, r2l_=10)

    xgb = xgb_create(x_train, y_train, 10)
    train_accuracy, test_accuracy = xgb_accuracy_count(xgb, x_train, x_test, y_train, y_test)
    print(f'train accuracy: {train_accuracy}')
    print(f'test accuracy: {test_accuracy}')

    x_train_features_selected, x_test_features_selected = feature_select(21, xgb, x_train_, x_test)
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_features_selected, y_train_,
                                                                              test_size=0.25, random_state=0)

    model = LSTMmodel(input_size=21, output_size=5, lstm_layers=3)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.8)
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    train_acc, val_acc = fit(model, loss_function, optimizer, x_train_split, y_train_split, epochs_count=5,
                             batch_size=100,
                             val_data=(x_val_split, y_val_split))

    test_acc = test(model, loss_function, x_test_features_selected, y_test)


def num_features_tweaking(x_train, x_test, y_train, y_test, n_features_array, xgb):
    accuracies_train = []
    accuracies_val = []
    accuracies_test = []
    for n_features in n_features_array:
        print(f'{n_features} features:')
        x_train_features_selected, x_test_features_selected = feature_select(n_features, xgb, x_train, x_test)
        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_features_selected, y_train,
                                                                                  test_size=0.25, random_state=0)

        model = LSTMmodel(input_size=n_features)
        loss_function = nn.CrossEntropyLoss(label_smoothing=0.8)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_acc, val_acc = fit(model, loss_function, optimizer, x_train_split, y_train_split, epochs_count=5,
                                 batch_size=100,
                                 val_data=(x_val_split, y_val_split))
        test_acc = test(model, loss_function, x_test_features_selected, y_test)

        accuracies_train.append(train_acc)
        accuracies_val.append(val_acc)
        accuracies_test.append(test_acc)

    return accuracies_train, accuracies_val, accuracies_test


def features_tweaking_run(n_features_array, train_path, test_path):
    full_df, train_size = file_reading(train_path, test_path)
    x_train, x_test, y_train, y_test = preprocessing(full_df, train_size)

    xgb = xgb_create(x_train, y_train, 10)
    train_accuracy, test_accuracy = xgb_accuracy_count(xgb, x_train, x_test, y_train, y_test)
    print(f'train accuracy: {train_accuracy}')
    print(f'test accuracy: {test_accuracy}')

    accuracies_train, accuracies_val, accuracies_test = num_features_tweaking(x_train, x_test, y_train, y_test,
                                                                              n_features_array, xgb)

    return accuracies_train, accuracies_val, accuracies_test


def dataset_tweaking_run(train_path, test_path):
    full_df, train_size = file_reading(train_path, test_path)
    x_train, x_test, y_train, y_test = preprocessing(full_df, train_size)

    xgb = xgb_create(x_train, y_train, 10)
    train_accuracy, test_accuracy = xgb_accuracy_count(xgb, x_train, x_test, y_train, y_test)
    print(f'train accuracy: {train_accuracy}')
    print(f'test accuracy: {test_accuracy}')

    factors = {'normal': [0.9, 0.8, 0.7, 0.6], 'probe': [1, 2, 3, 4], 'u2r': [100, 150, 200, 250],
               'r2l': [10, 20, 30, 40]}

    test_accuracies = []
    parameters = []

    for j in range(len(factors['normal'])):
        for i in range(len(factors['normal'])):
            print(factors['normal'][j], factors['probe'][i], factors['u2r'][i], factors['r2l'][i], '\n')

            x_train_, y_train_ = modified_preprocessing(full_df, train_size, normal_=factors['normal'][j],
                                                        probe_=factors['probe'][i], u2r_=factors['u2r'][i],
                                                        r2l_=factors['r2l'][i])
            x_train_features_selected, x_test_features_selected = feature_select(21, xgb, x_train_, x_test)
            x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train_features_selected,
                                                                                      y_train_, test_size=0.25,
                                                                                      random_state=0)

            model = LSTMmodel(input_size=21, output_size=5, lstm_layers=3)
            loss_function = nn.CrossEntropyLoss(label_smoothing=0.8)
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_acc, val_acc = fit(model, loss_function, optimizer, x_train_split, y_train_split, epochs_count=5,
                                     batch_size=100, val_data=(x_val_split, y_val_split))

            test_acc = test(model, loss_function, x_test_features_selected, y_test)

            parameters.append([factors['normal'][j], factors['probe'][i], factors['u2r'][i], factors['r2l'][i]])
            test_accuracies.append(test_acc)

    return test_accuracies, parameters



