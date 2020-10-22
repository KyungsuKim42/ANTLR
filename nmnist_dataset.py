import torch
import torch.utils.data as data
from pathlib import Path
import csv
import pdb

class NMNIST(data.Dataset):
    def __init__(self, train=True, time_length=200, truncate_time_in_ms=300):
        self.time_length = time_length
        self.truncate_time_in_ms = truncate_time_in_ms
        if train:
            self.data_path_binary = Path("./dataset/N-MNIST/Train/")
        else:
            self.data_path_binary = Path("./dataset/N-MNIST/Test/")
        label_fname = self.data_path_binary / "label.csv"
        with open(label_fname) as label_file:
            reader = csv.reader(label_file)
            self.label = [int(item[0]) for item in list(reader)]

    def __len__(self):
        return 64
        return len(self.label)

    def __getitem__(self, index):
        x_data_bin = torch.zeros(self.time_length, 2, 34, 34, device='cpu')
        with open(self.data_path_binary / f"{index+1:05d}.bin", "rb") as f:
            spikes = f.read()
        f_length = len(spikes)
        assert f_length % 5 == 0
        n_spike = int(f_length / 5)
        for i in range(n_spike):
            # x = spikes[i*5]
            # y = spikes[i*5+1]
            channel = int(spikes[i*5+2] / 128)
            time = ((spikes[i*5+2] - 128 * channel) << 16) + (spikes[i*5+3] << 8) + spikes[i*5+4]
            if (time // 1000) >= self.truncate_time_in_ms:
                break
            time_step = int( time // (1000 * (self.truncate_time_in_ms / self.time_length)) )
            x_data_bin[time_step, channel, spikes[i*5], spikes[i*5+1]] = 1
        x_data = x_data_bin
        y_data = self.label[index]

        return x_data, y_data

def load_loader(config, num_workers, batch_size, test_batch_size, valid_size=10000, time_length=300):
    nmnist = NMNIST(train=True, time_length=time_length)
    nmnist_test = NMNIST(train=False, time_length=time_length)
    if config.multi_model:
        train_loader = []
        valid_loader = []
        for m in range(config.num_models):
            # nmnist_train, nmnist_valid = torch.utils.data.random_split(nmnist, [60000-valid_size, valid_size])
            nmnist_train, nmnist_valid = torch.utils.data.random_split(nmnist, [32, 32])

            train_loader.append(data.DataLoader(nmnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True))
            valid_loader.append(data.DataLoader(nmnist_valid, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True))
        test_loader = data.DataLoader(nmnist_test, batch_size=test_batch_size, num_workers=num_workers)
    else:
        # nmnist_train, nmnist_valid = torch.utils.data.random_split(nmnist, [60000-valid_size, valid_size])
        nmnist_train, nmnist_valid = torch.utils.data.random_split(nmnist, [32, 32])
        train_loader = data.DataLoader(nmnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = data.DataLoader(nmnist_valid, batch_size=test_batch_size, num_workers=num_workers)
        test_loader = data.DataLoader(nmnist_test, batch_size=test_batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
