"""
Example integration of pytorch-lightning on a slurm cluster

Mike Peven (mpeven@gmail.com)
"""
import argparse
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from test_tube.hpc import SlurmCluster
from test_tube import HyperOptArgumentParser, Experiment

class MNISTClassifier(pl.LightningModule):
    def __init__(self, layer1_size, layer2_size, learning_rate, **kwargs):
        super().__init__()
        self.save_hyperparameters('layer1_size', 'layer2_size', 'learning_rate')
        self.setup_model()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--layer1_size', type=int, default=128)
        parser.add_argument('--layer2_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser

    def setup_model(self):
        self.layer_1 = torch.nn.Linear(28 * 28, self.hparams.layer1_size)
        self.layer_2 = torch.nn.Linear(self.hparams.layer1_size, self.hparams.layer2_size)
        self.layer_3 = torch.nn.Linear(self.hparams.layer2_size, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        acc = np.mean(torch.eq(torch.argmax(y_hat, 1), y).numpy())
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        acc = np.mean(torch.eq(torch.argmax(y_hat, 1), y).numpy())
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MNISTDataModule(pl.LightningDataModule):
    """
    Set up a DataModule with the correct train/val/test splits to make training simple

    For slurm: put any single-time processing in prepare_data() and per-node processing in setup()
    """

    def __init__(self, data_dir:str="./", batch_size:int=64, num_workers:int=8, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download only
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # transform
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_train = torchvision.datasets.MNIST(self.data_dir, train=True, download=False, transform=transform)
        mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, download=False, transform=transform)

        # train/val split
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# Version without slurm that works
def get_args():
    parser = argparse.ArgumentParser()

    # PROGRAM level args
    parser.add_argument('--notification_email', type=str)

    # Model args auto add
    parser = MNISTClassifier.add_model_specific_args(parser)

    # Dataset args auto add
    parser = MNISTDataModule.add_argparse_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


def get_hyperparameters_args():
    parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parser.add_argument('--test_tube_exp_name', default='sweep_test')
    parser.add_argument('--log_path', default='/home/map6/pytorch-slurm')
    parser.add_argument('--gpus', default=1)
    parser.opt_list(
        '--learning_rate', 
        default=0.001,
        type=float,
        options=[0.001, 0.01, 0.1],
        tunable=True
    )
    parser.opt_list(
        '--layer1_size',
        default=128, 
        type=float, 
        options=[64, 128, 256], 
        tunable=True
    )
    parser.opt_list(
        '--layer2_size',
        default=256, 
        type=float, 
        options=[64, 128, 256, 512], 
        tunable=True
    )
    parser.add_argument(
        '--data_dir',
        default="./", 
        type=str
    )
    parser.add_argument(
        '--batch_size',
        default=64, 
        type=int
    )
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int
    )

    # Model args auto add
    # parser = MNISTClassifier.add_model_specific_args(parser)

    # Dataset args auto add
    # parser = MNISTDataModule.add_argparse_args(parser)

    # # add all the available trainer options to argparse
    # # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    # parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    return args


def train(args, cluster):
    trainer = Trainer.from_argparse_args(args)
    model = MNISTClassifier(**vars(args))
    datamodule = MNISTDataModule.from_argparse_args(args)
    trainer.fit(model, datamodule)
    trainer.test(model)


def main_non_slurm():
    hyperparams = get_hyperparameters_args()
    train(hyperparams)


def main():
    hyperparams = get_hyperparameters_args()
    cluster = SlurmCluster(hyperparam_optimizer=hyperparams, log_path=hyperparams.log_path)
    cluster.notify_job_status(email='mpeven@gmail.com', on_done=True, on_fail=True)
    cluster.add_command('source activate recognition')
    # cluster.add_slurm_cmd(cmd='cpus-per-task', value='4', comment='CPUS per task.')
    cluster.add_slurm_cmd(cmd='gres', value='gpu:1', comment='gpus per task')
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.per_experiment_nb_cpus = 8
    cluster.job_time = '1:00:00'
    cluster.optimize_parallel_cluster_gpu(train, nb_trials=3*3*4, job_name='hyperparam_sweep')


if __name__ == "__main__":
    main()
