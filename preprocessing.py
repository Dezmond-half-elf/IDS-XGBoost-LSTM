from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"]


DoS = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable',
       'smurf', 'teardrop', 'udpstorm', 'worm']
Probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
R2L = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf',
       'sendmail', 'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']


def file_reading(path_train, path_test):
    train_df = pd.read_csv(path_train, header=None, index_col=False, names=col_names, sep=',')
    test_df = pd.read_csv(path_test, header=None, index_col=False, names=col_names, sep=',')
    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    return full_df, train_df.shape[0]


def attack_type_sort(attack):
    if attack == 'normal':
        return 0
    elif attack in DoS:
        return 1
    elif attack in Probe:
        return 2
    elif attack in U2R:
        return 3
    elif attack in R2L:
        return 4


def attack_group(dataset):
    new_dataset = dataset.copy()
    new_dataset['label'] = new_dataset['label'].apply(attack_type_sort)

    return new_dataset


def balance_show(dataset):
    counter = [0, 0, 0, 0, 0]
    for i in range(dataset.shape[0]):
        counter[dataset['label'][i]]+=1

    return counter, counter[0]/dataset.shape[0]


def feature_encode(dataset):
    categorial_features = []
    for col_name in dataset.columns:
        if dataset[col_name].dtypes == 'object':
            categorial_features.append(col_name)

    new_dataset = dataset.copy()
    new_dataset[categorial_features] = new_dataset[categorial_features].apply(LabelEncoder().fit_transform)

    return new_dataset


def scaling(dataset):
    scaler = MinMaxScaler().set_output(transform="pandas")
    x = dataset.drop(['label', 'difficulty'], axis=1)
    y = dataset['label']
    scaled_x = scaler.fit_transform(x)

    return scaled_x, y


def preprocessing(dataset, train_size):
    dataset = attack_group(dataset)
    dataset = feature_encode(dataset)
    x, y = scaling(dataset)

    scaled_x_train = x[:train_size]
    scaled_x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    return scaled_x_train, scaled_x_test, y_train, y_test


def one_label_splitter(x, y, label):
    mask = y.isin([label])

    return x[mask], y[mask]


def train_data_create(x_train, y_train, normal_=0.8, probe_=2, u2r_=100, r2l_=30):
    x0, y0 = one_label_splitter(x_train, y_train, 0)
    x1, y1 = one_label_splitter(x_train, y_train, 1)
    x2, y2 = one_label_splitter(x_train, y_train, 2)
    x3, y3 = one_label_splitter(x_train, y_train, 3)
    x4, y4 = one_label_splitter(x_train, y_train, 4)

    x_u2rs = x3.copy()
    y_u2rs = y3.copy()
    for i in range(u2r_):
        x_u2rs = pd.concat([x_u2rs, x3])
        y_u2rs = pd.concat([y_u2rs, y3])

    x_r2ls = x4.copy()
    y_r2ls = y4.copy()
    for i in range(r2l_):
        x_r2ls = pd.concat([x_r2ls, x4])
        y_r2ls = pd.concat([y_r2ls, y4])

    x_probes = x2.copy()
    y_probes = y2.copy()
    for i in range(probe_):
        x_probes = pd.concat([x_probes, x2])
        y_probes = pd.concat([y_probes, y2])

    normal_size = int(x0.shape[0] * normal_)

    x = pd.concat([x_u2rs, x_r2ls, x_probes, x0[:normal_size], x1])
    y = pd.concat([y_u2rs, y_r2ls, y_probes, y0[:normal_size], y1])
    x['labels'] = y
    x.sample(frac=1)
    y = x['labels']

    return x.drop(['labels'], axis=1), y


def modified_preprocessing(data, train_size, normal_=0.8, probe_=2, u2r_=100, r2l_=30):
    dataset = attack_group(data)
    dataset = feature_encode(dataset)
    x, y = scaling(dataset)

    x_train = x[:train_size]
    y_train = y[:train_size]

    new_train_x, new_train_y = train_data_create(x_train, y_train, normal_, probe_, u2r_, r2l_)

    return new_train_x, new_train_y
