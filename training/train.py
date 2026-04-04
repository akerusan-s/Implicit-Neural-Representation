import torch

from metrics.metrics import metric

from data.utils.loader import get_data, get_dataloader
from models.utils import get_outputs, get_derivatives, get_grads
from training.utils import Logger


def train(
    model,
    optimizer,
    dataloader,
    config,
    timestamps,
    data,
    data_derivatives
):
    epochs = config.epochs
    m = len(dataloader) + 1
    c = config.c_hyperparam

    logger = Logger(config)

    for epoch in range(epochs):
        model.train()

        loss1_total = 0.0
        loss2_total = 0.0
        loss3_total = 0.0

        for t1, t2, el1, el2 in dataloader:
            h = t2 - t1
            t_inter = t1 + h / 2
            el1 = el1.view(-1)
            el2 = el2.view(-1)

            output1 = model(t1)
            output2 = model(t2)
            output_inter = model(t_inter)

            loss1 = torch.norm(output1 - el1) ** 2 / m

            grad1 = get_grads(t1, output1)
            grad_inter = get_grads(t_inter, output_inter)
            grad2 = get_grads(t2, output2)

            k1 = h * grad1
            k2 = h * grad_inter
            k3 = h * grad_inter
            k4 = h * grad2

            loss2 = torch.norm(el2 - el1 - 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)) ** 2 / m

            grad2_1 = get_grads(t1, grad1)
            grad2_2 = get_grads(t2, grad2)

            loss3 = torch.norm(grad2_2 - grad2_1) ** 2 / m

            loss1_total = loss1_total + loss1
            loss2_total = loss2_total + loss2
            loss3_total = loss3_total + loss3

        losses = [loss1_total, loss2_total, loss3_total]
        total_loss = sum(losses[i] * c[i] for i in range(3))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # validation
        model.eval()
        eval_loss_state = metric(
            get_outputs(model, timestamps)[:-1, :],
            data[:-1, :]
        )
        eval_loss_dot = metric(
            get_derivatives(model, timestamps)[:-1, :],
            data_derivatives[:-1, :]
        )
        logger.log(
            epoch + 1,
            eval_loss_state,
            eval_loss_dot,
            total_loss.item(),
            [loss.item() for loss in losses]
        )

        if logger.check_save(epoch + 1):
            logger.save(epoch + 1, model, optimizer)

    return model


def train_model(config, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    timestamps, data, data_derivatives, data_noised = get_data(config.data_path)
    dataloader = get_dataloader(data_noised, timestamps)

    model = train(
        model,
        optimizer,
        dataloader,
        config,
        timestamps,
        data,
        data_derivatives
    )

    return model
