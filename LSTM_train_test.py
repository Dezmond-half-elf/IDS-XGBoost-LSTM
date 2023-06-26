import math
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

def data_transform(x, y):
    x_tensor = torch.tensor(x.values).type(torch.FloatTensor)
    y_tensor = torch.tensor(y.values).type(torch.LongTensor)

    return torch.reshape(x_tensor, (x_tensor.shape[0], 1, x_tensor.shape[1])), y_tensor


def iterate_batches(x, y, batch_size):
    n_samples = x.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]

        yield x[batch_indices], y[batch_indices]


class LSTMmodel(nn.Module):
    def __init__(self, input_size=17, output_size=5, hidden_units=150, lstm_layers=3):
         super(LSTMmodel, self).__init__()

         self.lstm_layers = lstm_layers
         self.hidden_units = hidden_units
         self.input_size = input_size

         self.lstm = nn.LSTM(input_size, hidden_units, num_layers=lstm_layers, batch_first=True)
         self.linear = nn.Linear(hidden_units, output_size)
         self.relu = nn.ReLU()


    def forward(self, intrusion):
        lstm_out, (hn, cn) = self.lstm(intrusion)
        relu_out = self.relu(lstm_out)
        linear_out = self.linear(relu_out)

        return F.softmax(linear_out, dim=2)


def accuracy_count(predict, gt):
    _, indices = torch.max(predict, 2)
    indices_vector = indices.reshape(indices.shape[0])

    return torch.sum(indices_vector==gt, dtype=int).item(), gt.size()[0]


def loss_count(predict, gt, loss_function):
    reshaped_predict = torch.reshape(predict, (predict.shape[0], predict.shape[2]))

    return loss_function(reshaped_predict, gt)


def do_epoch(model, loss_function, x_, y_, batch_size, optimizer=None, name=None):
    epoch_loss = 0
    correct_count = 0
    sum_count = 0

    x, y = data_transform(x_, y_)

    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)

    batches_count = math.ceil(x.shape[0] / batch_size)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, (x_batch, y_batch) in enumerate(iterate_batches(x, y, batch_size)):
                predict = model(x_batch)

                cur_correct_count, cur_sum_count = accuracy_count(predict, y_batch)

                loss = loss_count(predict, y_batch, loss_function)
                epoch_loss += loss.item()

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                correct_count += cur_correct_count
                sum_count += cur_sum_count

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
                    name, loss.item(), cur_correct_count / cur_sum_count)
                )

            progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format(
                name, epoch_loss / batches_count, correct_count / sum_count)
            )

    return epoch_loss / batches_count, correct_count / sum_count


def fit(model, criterion, optimizer, x, y, epochs_count=1, batch_size=32,
        val_data=None, val_batch_size=None):

    if not val_data is None and val_batch_size is None:
        val_batch_size = batch_size

    train_accuracies = []
    val_accuracies = []
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss, train_acc = do_epoch(model, criterion, x, y, batch_size, optimizer, name_prefix + 'Train:')
        train_accuracies.append(round(train_acc, 4) * 100)

        if not val_data is None:
            val_loss, val_acc = do_epoch(model, criterion, val_data[0], val_data[1], val_batch_size, None,
                                         name_prefix + '  Val:')
            val_accuracies.append(round(val_acc, 4) * 100)

    return train_accuracies[-1], val_accuracies[-1]


def test(model, criterion, x, y):
    test_loss, test_acc = do_epoch(model, criterion, x, y, batch_size=x.shape[0])

    return round(test_acc, 4)*100



