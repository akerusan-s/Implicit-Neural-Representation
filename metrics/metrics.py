import torch


def metric(input_x, output_y, device='cpu') -> float:
    x = torch.as_tensor(input_x, dtype=torch.float32).to(device)
    y = torch.as_tensor(output_y, dtype=torch.float32).to(device)
    return (torch.linalg.norm(x - y) / torch.linalg.norm(x)).item()
