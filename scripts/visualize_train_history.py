import json
import matplotlib.pyplot as plt


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_losses(data):
    epochs = [entry['epoch'] for entry in data]
    eval_loss_state = [entry['eval_loss_state'] for entry in data]
    eval_loss_dot = [entry['eval_loss_dot'] for entry in data]
    total_loss = [entry['total_loss'] for entry in data]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('История обучения (Loss over epochs)', fontsize=16)

    axes[0].plot(epochs, eval_loss_state, linestyle='-', color='b')
    axes[0].set_ylabel('eval_loss_state')
    axes[0].grid(True)

    axes[1].plot(epochs, eval_loss_dot, linestyle='-', color='r')
    axes[1].set_ylabel('eval_loss_dot')
    axes[1].grid(True)

    axes[2].plot(epochs, total_loss, linestyle='-', color='g')
    axes[2].set_ylabel('total_loss')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


data = load_data("../models_checkpoints/model_3/500_epoch/iteration_info.json")
data.sort(key=lambda x: x['epoch'])
plot_losses(data)
