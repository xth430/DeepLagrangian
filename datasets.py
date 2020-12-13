import torch

class Datasets(torch.utils.data.Dataset):
    def __init__(self, states, targets):
        assert len(states) == len(targets)
        self.data_num = len(targets)
        self.states = states
        self.targets = targets

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        return self.states[idx], self.targets[idx]


def gen_loader(states, targets, batch_size=1, shuffle=False):
    states, targets = torch.tensor(states), torch.tensor(targets)
    datasets = Datasets(states, targets)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)

    return train_loader