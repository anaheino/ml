import torch

weights = torch.tensor([0.1, 0.3, 0.4, 0.2])
samples = torch.multinomial(weights, num_samples=8, replacement=False)
print(samples)