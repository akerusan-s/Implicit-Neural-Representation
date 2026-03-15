from pathlib import Path

import json

from models.utils import save_model


class Logger:

    def __init__(
        self,
        config
    ):
        self.log_interval = config.log_interval
        self.save_interval = config.save_interval
        self.need_save = config.save
        self.saver = Saver(config)
        self.iteration_info = []

    def log(
        self,
        epoch,
        eval_loss_state,
        eval_loss_dot,
        total_loss,
        losses
    ):
        if self.check_log(epoch):
            print(f"Epoch: {epoch}.")
            print(f"Loss: {eval_loss_state} states, {eval_loss_dot} derivatives, {total_loss} loss ({losses})")
            print()
        self.iteration_info.append({
            "epoch": epoch,
            "eval_loss_state": eval_loss_state,
            "eval_loss_dot": eval_loss_dot,
            "total_loss": total_loss,
            "losses": losses
        })

    def check_log(self, epoch):
        return epoch % self.log_interval == 0

    def check_save(self, epoch):
        return self.need_save and epoch % self.save_interval == 0

    def save(self, epoch, model, optimizer):
        path = self.saver.save(epoch, model, optimizer)
        with open(path + "/iteration_info.json", 'w', encoding='utf-8') as fp:
            json.dump(self.iteration_info, fp, ensure_ascii=False, indent=4)

        print("Saved to:", path)
        print()


class Saver:

    def __init__(self, config):
        self.save_path_checkpoint = config.save_path
        Path(config.save_path).mkdir(parents=True, exist_ok=True)

    def save(self, epoch, model, optimizer):
        path = self.save_path_checkpoint + "/" + f"{epoch}_epoch"

        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        save_model(model, optimizer, path)

        return path
