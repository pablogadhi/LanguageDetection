import re
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from pyonmttok import Tokenizer
from nltk.translate.bleu_score import sentence_bleu


class BLEUDataset(Dataset):
    def __init__(self, pd_dataframe):
        self.dataframe = pd_dataframe

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        inputs = self.dataframe.iloc[idx,
                                     self.dataframe.columns != 'origin'].values
        label = int(self.dataframe.iloc[idx]['origin'])
        return inputs, label

    def get_all_labels(self):
        return [int(x['origin']) for _, x in self.dataframe.iterrows()]


class Classifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(Classifier, self).__init__()
        self.hidden0 = nn.Linear(10, 1024, bias=True)
        self.hidden1 = nn.Linear(1024, 512, bias=True)
        self.hidden2 = nn.Linear(512, 128, bias=True)
        self.hidden3 = nn.Linear(128, 32, bias=True)
        self.hidden4 = nn.Linear(32, 8, bias=True)
        self.out = nn.Linear(8, 4, bias=True)
        self.l_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.l_relu(self.hidden0(x))
        # x = self.dropout(x)
        x = self.l_relu(self.hidden1(x))
        # x = self.dropout(x)
        x = self.l_relu(self.hidden2(x))
        x = self.l_relu(self.hidden3(x))
        x = self.l_relu(self.hidden4(x))
        x = self.dropout(x)
        x = self.softmax(self.out(x))
        return x


def train_classifier(classifier, dataloader, epochs, log_rate, device='cpu'):
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        start = time.time()
        total, correct = (0, 0)
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = classifier(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
            if i % log_rate == log_rate - 1:
                print('Epoch {}, iter {}: {:.4f} loss | {:.4f}% accuracy'.format(
                    epoch, i + 1, running_loss / log_rate, correct / total * 100))
                running_loss = 0.0

        end = time.time()
        print('Epoch {} took: {:.3f}s'.format(epoch, end - start))

    print('Finished Training')


def test_classifier(classifier, dataloader):
    classifier.eval()

    predictions = []
    total, correct = (0, 0)
    for data in dataloader:
        inputs, labels = data
        outputs = classifier(inputs.float())

        _, predicted = torch.max(outputs.data, 1)
        predictions = predictions + predicted.tolist()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test data: {:.2f}%'.format(
        100 * correct / total))
    return predictions


class Detector:
    def __init__(self, model_path):
        self.tokenizer = Tokenizer('conservative')
        self.classifier = self.load_classifier(model_path)

    def load_classifier(self, model_path):
        classifier = Classifier()
        classifier.load_state_dict(torch.load(model_path))
        classifier.eval()
        return classifier
