import torch


def get_derivatives(model, inputs) -> torch.Tensor:
    inputs_tensor = torch.tensor(inputs, dtype=torch.float, requires_grad=True).reshape(-1, 1)
    outputs = model(inputs_tensor)

    grad_per_coord = [torch.autograd.grad(
        outputs[:, i],
        inputs_tensor,
        grad_outputs=torch.ones_like(outputs[:, i]),
        create_graph=True, retain_graph=True
    )[0] for i in range(outputs.shape[1])]

    return torch.cat(grad_per_coord, dim=1)


def get_outputs(model, inputs) -> torch.Tensor:
    inputs_tensor = torch.as_tensor(inputs, dtype=torch.float32).view(-1, 1)
    return model(inputs_tensor)


def get_grads(inputs, outputs):
    grad_per_coord = [torch.autograd.grad(
        outputs[i],
        inputs,
        grad_outputs=torch.ones_like(outputs[i]),
        create_graph=True, retain_graph=True
    )[0] for i in range(outputs.shape[0])]
    return torch.cat(grad_per_coord)


def save_model(model, optimizer, dir_path):
    torch.save(model, dir_path + "/model.pth")
    torch.save(optimizer.state_dict(), dir_path + "/optimizer.pth")


def infer_model(checkpoint_path, inputs):
    model = torch.load(checkpoint_path, weights_only=False)
    model.eval()
    return get_outputs(model, inputs).detach().numpy(), get_derivatives(model, inputs).detach().numpy()
