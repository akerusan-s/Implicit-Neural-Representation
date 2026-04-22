import torch

from data.utils.loader import get_data
from metrics.metrics import metric
from models.utils import get_grads_batch
from training.utils import Logger


def train(
    model,
    optimizer,
    config,
    timestamps,
    data,
    data_derivatives,
    data_noised,
    return_metric=False
):
    logger = Logger(config)
    model = model.to(config.device)
    c = config.c_hyperparam
    seq_len = timestamps.shape[0] - 1

    eval_loss_dot = float('inf')

    for epoch in range(config.epochs):
        model.train()

        inputs = torch.tensor(timestamps, requires_grad=True, dtype=torch.float).to(config.device)    # (100,)
        next_inputs = inputs.clone().roll(-1)[:seq_len].reshape(-1, 1)
        inputs = inputs[:seq_len].reshape(-1, 1)

        targets = torch.tensor(data_noised, dtype=torch.float).T.to(config.device)
        targets = targets[:, :seq_len]  # (2, 100)

        h = next_inputs - inputs
        half_inputs = h / 2 + inputs

        outputs = model(inputs).T     # (2, 100)
        half_outputs = model(half_inputs).T
        next_outputs = model(next_inputs).T

        jac = get_grads_batch(inputs, outputs.T).T
        half_jac = get_grads_batch(half_inputs, half_outputs.T).T
        next_jac = get_grads_batch(next_inputs, next_outputs.T).T

        hes = get_grads_batch(inputs, jac.T).T
        next_hes = get_grads_batch(next_inputs, next_jac.T).T

        loss1 = (torch.norm(outputs - targets, dim=0) ** 2).sum() / seq_len

        k1 = h.T * jac
        k2 = h.T * half_jac
        k3 = h.T * half_jac
        k4 = h.T * next_jac
        loss2 = (torch.norm(next_outputs - outputs - 1/6 * (k1 + 2 * k2 + 2 * k3 + k4), dim=0) ** 2).sum() / (seq_len - 1)

        loss3 = (torch.norm(next_hes - hes, dim=0) ** 2).sum() / (seq_len - 1)

        loss = c[0] * loss1 + c[1] * loss2 + c[2] * loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        infer_input = torch.tensor(timestamps, requires_grad=True, dtype=torch.float).to(config.device).reshape(-1, 1)
        infer_output = model(infer_input)
        eval_loss_state = metric(
            infer_output[1:-1, :],
            data[1:-1, :],
            config.device
        )
        eval_loss_dot = metric(
            get_grads_batch(infer_input, infer_output)[1:-1, :],
            data_derivatives[1:-1, :],
            config.device
        )
        logger.log(
            epoch + 1,
            eval_loss_state,
            eval_loss_dot,
            loss.item(),
            [loss1.item(), loss2.item(), loss3.item()]
        )
        if logger.check_save(epoch + 1):
            logger.save(epoch + 1, model, optimizer)

    if return_metric:
        return eval_loss_dot
    return model


def train_model(config, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    timestamps, data, data_derivatives, data_noised = get_data(config.data_path)

    model = train(
        model,
        optimizer,
        config,
        timestamps,
        data,
        data_derivatives,
        data_noised
    )

    return model
