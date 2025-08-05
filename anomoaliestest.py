import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from arff2pandas import a2p
import scipy.io.arff as arff 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

trainset_file = 'ECG5000_TRAIN.arff'
testset_file = 'ECG5000_TEST.arff'

traindata, trainmeta = arff.loadarff(trainset_file)
testdata, testmeta = arff.loadarff(testset_file)
train = pd.DataFrame(traindata, columns=trainmeta.names())
test = pd.DataFrame(testdata, columns=testmeta.names())
# df = train.append(test)
df = pd.concat([train, test])

print(train.shape)
print(test.shape)
print(df.shape)

print(df.head())

fig = plt.figure(figsize=(12, 6))

new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns
df_norm = df[df.target == b'1']
df_abnorm = df[df.target != b'1']

plt.plot(df_norm.iloc[0, :-1])
plt.plot(df_norm.iloc[10, :-1])
plt.plot(df_norm.iloc[200, :-1])

plt.show()

fig = plt.figure(figsize=(12, 6))

plt.plot(df_abnorm.iloc[0, :-1])
plt.plot(df_abnorm.iloc[10, :-1])
plt.plot(df_abnorm.iloc[200, :-1])

plt.show()


print(df_norm.shape)
print(df_abnorm.shape)

class ECG5000(Dataset):

    def __init__(self, mode):

        assert mode in ['normal', 'anomaly']

        trainset_file = 'ECG5000_TRAIN.arff'
        testset_file = 'ECG5000_TEST.arff'

        traindata, trainmeta = arff.loadarff(trainset_file)
        testdata, testmeta = arff.loadarff(testset_file)
        train = pd.DataFrame(traindata, columns=trainmeta.names())
        test = pd.DataFrame(testdata, columns=testmeta.names())
        # df = train.append(test)
        df = pd.concat([train, test])

        # split in normal and anomaly data, then drop label
        CLASS_NORMAL = 1
        new_columns = list(df.columns)
        new_columns[-1] = 'target'
        df.columns = new_columns


        if mode == 'normal':
            df = df[df.target == b'1'].drop(labels='target', axis=1)
        else:
            df = df[df.target != b'1'].drop(labels='target', axis=1)

        print(df.shape)
        # train_df, val_df = train_test_split(
        #     normal_df,
        #     test_size=0.15,
        #     random_state=random_seed
        # )
        #
        # val_df, test_df = train_test_split(
        #     val_df,
        #     test_size=0.33,
        #     random_state=random_seed
        # )

        self.X = df.astype(np.float32).to_numpy()

    def get_torch_tensor(self):
        return torch.from_numpy(self.X)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]).reshape(-1, 1)

    # return len of dataset
    def __len__(self):
        return self.X.shape[0]
    
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'ENCODER input dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        # print(f'ENCODER reshaped dim: {x.shape}')
        x, (_, _) = self.rnn1(x)
        # print(f'ENCODER output rnn1 dim: {x.shape}')
        x, (hidden_n, _) = self.rnn2(x)
        # print(f'ENCODER output rnn2 dim: {x.shape}')
        # print(f'ENCODER hidden_n rnn2 dim: {hidden_n.shape}')
        # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, self.embedding_dim)}')
        return hidden_n.reshape((batch_size, self.embedding_dim))
    
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(f'DECODER input dim: {x.shape}')
        x = x.repeat(self.seq_len, self.n_features) # todo testare se funziona con più feature
        # print(f'DECODER repeat dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER reshaped dim: {x.shape}')
        x, (hidden_n, cell_n) = self.rnn1(x)
        # print(f'DECODER output rnn1 dim:/ {x.shape}')
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)
    
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, device='cuda', batch_size=32):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
dataset_normal = ECG5000(mode='normal')
dataset_anomaly = ECG5000(mode='anomaly')

print(f'Normal dataset: {len(dataset_normal.X)}'
        f'Anomaly dataset: {len(dataset_anomaly.X)}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_len, n_features = 140, 1
batch_size = 512

################################
validation_split = test_split = 0.15
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset_normal)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
print(f'dataset_size: {dataset_size}')
print(f'split: {split}')

# suffling
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]
train_indices, val_indices = train_indices[split:], train_indices[:split]

print('train_indices: ', len(train_indices))
print('val_indices: ', len(val_indices))
print('test_indices: ', len(test_indices))

# check all splits have no intersections
assert not [value for value in train_indices if value in test_indices]
assert not [value for value in train_indices if value in val_indices]
assert not [value for value in val_indices if value in test_indices]

model = RecurrentAutoencoder(seq_len, n_features=n_features, embedding_dim=128, device=device, batch_size=batch_size)

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(dataset_normal, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset_normal, batch_size=batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(dataset_normal, batch_size=batch_size, sampler=test_sampler)
anomaly_loader = torch.utils.data.DataLoader(dataset_anomaly, batch_size=batch_size)

# start training
n_epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='mean').to(device) # todo article use L1Loss
history = dict(train=[], val=[])
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 10000.0

for epoch in tqdm(range(1, n_epochs + 1)):
    model = model.train() 

    train_losses = []
    val_losses = []
    test_losses = []
    anomaly_losses = []

    for i, seq_true in enumerate(train_loader):
        optimizer.zero_grad()
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model = model.eval()
    with torch.no_grad():

        # validation steps
        for i, seq_true in enumerate(validation_loader):
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            val_losses.append(loss.item())

        # normal_test steps
        for i, seq_true in enumerate(test_loader):
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            test_losses.append(loss.item())

        # anomaly_test steps
        for i, seq_true in enumerate(anomaly_loader):
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            anomaly_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    test_loss = np.mean(test_losses)
    anomaly_loss = np.mean(anomaly_losses)
    history['train'].append(train_loss)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: train loss {train_loss} {" "*6} val loss {val_loss} {" "*6} test loss {test_loss} {" "*6} anomaly loss {anomaly_loss}')
    # print(f'Epoch {epoch}: train loss {train_loss} {" "*6} val loss {val_loss} {" "*6} test loss {test_loss} {" "*6} anomaly loss {anomaly_loss}')

model.load_state_dict(best_model_wts)