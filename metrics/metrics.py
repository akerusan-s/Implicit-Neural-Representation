import torch


def metric(ground_truth, predicted, device='cpu') -> float:
    x = torch.as_tensor(ground_truth, dtype=torch.float32).to(device)
    y = torch.as_tensor(predicted, dtype=torch.float32).to(device)
    return (torch.linalg.norm(x - y) / torch.linalg.norm(x)).item()
