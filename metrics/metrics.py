import torch


def metric(input_x, output_y) -> float:
    x = torch.as_tensor(input_x, dtype=torch.float32)
    y = torch.as_tensor(output_y, dtype=torch.float32)
    return (torch.linalg.norm(x - y) / torch.linalg.norm(x)).item()


