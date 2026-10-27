import csv
import os

# tensorboard n'est pas installe dans l'environnement pyOR : les pertes sont ecrites
# dans deux CSV du dossier d'experience, lus ensuite par la page Quarto.
#   losses_train.csv : epoch, global_step, step, <une colonne par perte>  (chaque iteration)
#   losses_val.csv   : epoch, <une colonne par perte>                     (chaque epoque)


def _value(v):
    return float(v.detach().cpu()) if hasattr(v, "detach") else float(v)


class TrainLogger:
    def __init__(self, experiment_dir, arg_obj, print_iter=200):
        self.experiment_dir = experiment_dir
        self.arg_obj = arg_obj
        self.print_iter = print_iter
        self.train_path = os.path.join(experiment_dir, 'losses_train.csv')
        self.val_path = os.path.join(experiment_dir, 'losses_val.csv')
        self._train_cols = None
        self._val_cols = None

    @staticmethod
    def _append(path, cols, row):
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            w.writerow([row.get(c, '') for c in cols])

    def log(self, epoch, global_step, step, current_losses):
        values = {k: _value(v) for k, v in current_losses.items()}
        if self._train_cols is None:
            self._train_cols = ['epoch', 'global_step', 'step', *values]
        self._append(self.train_path, self._train_cols,
                     {'epoch': epoch, 'global_step': global_step, 'step': step, **values})
        if global_step % self.print_iter == (self.print_iter - 1):
            for criterion, v in values.items():
                print(f'[{epoch}, {global_step:5d}] Train loss ({criterion}): {v:.6f}')

    def log_validation(self, val_loss, epoch):
        values = {k: _value(v) for k, v in val_loss.items()}
        if self._val_cols is None:
            self._val_cols = ['epoch', *values]
        self._append(self.val_path, self._val_cols, {'epoch': epoch, **values})

    def close(self):
        pass
