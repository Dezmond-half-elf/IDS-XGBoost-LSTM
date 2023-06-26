from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix_show(predict, y_test):
    _, indices = predict.max(2)
    y_pred = indices.reshape(predict.shape[0]).numpy()
    y_true = np.array(y_test)

    classes = ('normal', 'Dos', 'Probe', 'U2R', 'R2L')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
