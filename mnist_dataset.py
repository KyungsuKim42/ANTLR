import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from pathlib import Path

def load_loader(config, num_workers, batch_size, test_batch_size):
    mnist = datasets.MNIST(f"./dataset/mnist/",
                                 download=True,
                                 train=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    mnist_test = datasets.MNIST(Path.home() / f"./dataset/mnist/",
                                 download=True,
                                 train=False,
                                 transform= transforms.Compose([transforms.ToTensor()]))
    if config.multi_model:
        train_loader = []
        valid_loader = []
        for m in range(config.num_models):
            mnist_train, mnist_valid = torch.utils.data.random_split(mnist, [50000, 10000])
            # mnist_train, mnist_valid = torch.utils.data.random_split(mnist[:50], [25, 25])

            train_loader.append(data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers))
            valid_loader.append(data.DataLoader(mnist_valid, batch_size=test_batch_size, num_workers=num_workers))
            test_loader = data.DataLoader(mnist_test, batch_size=test_batch_size, num_workers=num_workers)
    else:
        mnist_train, mnist_valid = torch.utils.data.random_split(mnist, [50000, 10000])
        # mnist_train, mnist_valid = torch.utils.data.random_split(mnist[:50], [25, 25])
        train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = data.DataLoader(mnist_valid, batch_size=test_batch_size, num_workers=num_workers)
        test_loader = data.DataLoader(mnist_test, batch_size=test_batch_size, num_workers=num_workers)
    return train_loader, valid_loader, test_loader
