import numpy as np
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


class Classifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(17, 9, bias=True)
        self.relu = nn.ReLU()
        self.out = nn.Linear(9, 4, bias=True)
        self.softmax = nn.Softmax(4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.softmax(self.out(x))
        return x


def train_classifier(classifier, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


class Detector:
    def __init__(self):
        self.tokenizer = Tokenizer('conservative')
        self.classifier = self.load_classifier()

    def load_classifier(self):
        # TODO Load a trained model
        return Classifier()
