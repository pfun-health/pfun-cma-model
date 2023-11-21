import torch
from sklearn.model_selection import ParameterGrid
from torch.utils.data import Dataset, DataLoader
import json

class CMATrainingSampleDataset(Dataset):
    """
    A PyTorch Dataset class to generate training samples for CMA model parameters.
    """
    def __init__(self, parameter_grid):
        self.samples = list(ParameterGrid(parameter_grid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def generate_training_samples(n_samples, parameter_grid, use_cuda=False):
    """
    Generates N training samples based on the provided parameter grid.
    
    :param n_samples: Number of samples to generate.
    :param parameter_grid: Grid of parameters to sample from.
    :param use_cuda: Boolean flag to use CUDA if available.
    :return: A list of training samples.
    """
    dataset = CMATrainingSampleDataset(parameter_grid)
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    samples = []
    for batch in dataloader:
        for params in batch:
            generator = CMATrainingSampleGenerator(params)
            sample = generator.generate_sample()
            samples.append(sample)
            if len(samples) >= n_samples:
                return samples

# Example usage
parameter_grid = {
    'd': [-0.5, 0, 0.5],
    'taup': [1, 4],
    'taug': [0.8, 1.2],
    'B': [0.05, 0.15],
    'Cm': [0, 0.7],
    'toff': [-0.5, 0.5]
}

samples = generate_training_samples(10, parameter_grid, use_cuda=True)
print(json.dumps(samples, indent=2))
