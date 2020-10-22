import torch
import matplotlib.pyplot as plt
import os
import sys
import time
from pathlib import Path
import argparse
import copy

from antlr import *
from trainer import *
import utils
import mnist_dataset
import nmnist_dataset


parser = argparse.ArgumentParser(description='ANTLR MNIST/N-MNIST Arguments', fromfile_prefix_chars='@')


parser.add_argument('--task', type=str, default='mnist', metavar='S',
                    help='which task to run (mnist or nmnist)')
parser.add_argument('-t', '--tag', type=str, default='untitled', metavar='S',
                    help='tag for the run')
parser.add_argument('-s', '--random-seed', type=int, metavar='N',
                    help='random seed. if not specified, new random seed is generated.')
parser.add_argument('--evaluation-mode', action='store_true', default=False,
                    help='enable evaluation mode')

# Training/Model configuration.
parser.add_argument('--epoch', type=int, default=20, metavar='S',
                    help='How many epochs to run.')

parser.add_argument('--optim-name', type=str, default='adam', metavar='S',
                    help='a type of optimizer (e.g. adam (default), sgd)')
parser.add_argument('--learning-rate', type=float, default=0.001, metavar='F',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0, metavar='F',
                    help='SGD momentum (default: 0))')
parser.add_argument('--weight-decay', type=float, default=0, metavar='F',
                    help='weight decay (default: 0))')
parser.add_argument('--lr-step-size', type=int, default=10, metavar='F',
                    help='learning rate step size for lr_scheduler. (default: 10)')
parser.add_argument('--lr-gamma', type=int, default=0.1, metavar='F',
                    help='learning rate scale down factor for lr_scheduler. (default: 0.1)')

parser.add_argument('--max_target-spikes', type=int, default=1, metavar='N',
                    help='number of spikes for true target (default: 1)')
parser.add_argument('--min_target-spikes', type=int, default=0, metavar='N',
                    help='number of spikes for false target (default: 0)')
parser.add_argument('--max-input-timing', type=int, default=0, metavar='N',
                    help='earliest input timing for maximum input value (default: 0)')
parser.add_argument('--min-input-timing', type=int, default=24, metavar='N',
                    help='latest input timing for minimum input value (default: 24)')

parser.add_argument('--gpu', action='store_true', default=False,
                    help="Whether to use GPU or not.")
parser.add_argument('--resume', action='store_true', default=False,
                    help="Whether to resume at the last point.")
parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                    help='(default: 1)')
parser.add_argument('--inf-speed-test', type=int, default=0, metavar='N',
                    help='(default: 0)')

model_args = parser.add_argument_group('model parameters')
parser.add_argument('--time-length', type=int, default=300, metavar='F',
                    help='simulation time length (default: 300))')
model_args.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
model_args.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
model_args.add_argument('--alpha-i', type=float, default=0.99, metavar='F',
                        help='alpha_i used for decaying current\
                        (default: 0.99)')
model_args.add_argument('--alpha-v', type=float, default=0.99, metavar='F',
                        help='alpha_v used for decaying voltage\
                        (default: 0.99)')
model_args.add_argument('--beta-i', type=float, default=1.0, metavar='F',
                        help='beta_i used for scaling current (default: 1.0)')
model_args.add_argument('--beta-v', type=float, default=1.0, metavar='F',
                        help='beta_v used for scaling voltage (default: 1.0)')
model_args.add_argument('--beta-bias', type=float, default=1.0, metavar='F',
                        help='beta_bias used for scaling voltage bias (default: 1.0)')
model_args.add_argument('--surr-alpha', type=float, default=1.0, metavar='F',
                        help='surr_alpha used for surrogate derivative \
                        (default: 1.0)')
model_args.add_argument('--surr-beta', type=float, default=3.0, metavar='F',
                        help='surr_beta used for surrogate derivative \
                        (default: 3.0)')
model_args.add_argument('-l', '--lrule', type=str, default='ANTLR', metavar='S',
                        help='learning rule type')
model_args.add_argument('--target-type', type=str, default='latency', metavar='S',
                        help='type of target values (e.g. \'count\' (default), \'train\', \'latency\')')
# Depricated.
model_args.add_argument('--lambda-nospike', type=float, default=0.1, metavar='F',
                        help='lambda for no_spike loss or latency target (default: 0.1)')
model_args.add_argument('--timing-penalty', type=float, default=100.0, metavar='F',
                        help='(default: 100.0)')
model_args.add_argument('--grad-clip', type=float, default=1.0, metavar='F',
                        help='(default: 1.0)')

model_args.add_argument('--multi-model', action='store_true', default=False,
                        help='Turn multi model mode on.')
model_args.add_argument('--num-models', type=int, default=1, metavar='N',
                        help='Default=1')
model_args.add_argument('--init-bias-center', type=int, default=0, metavar='N',
                        help='0')
model_args.add_argument('--beta-auto', type=int, default=1, metavar='N',
                        help='1')

# Beta in softmax function.
model_args.add_argument('--softmax-beta', type=float, default=0.166667, metavar='N',
                        help='Beta (1/temperature) in softmax')



def main():
    # Load configurations from json file.
    apargs = parser.parse_args()

    # Initialize configuration instance.
    if apargs.evaluation_mode:
        # apargs.config = Path(f"./logs/{apargs.tag}/config.json")
        config_dict = utils.read_json(apargs.config)
        config_dict['evaluation_mode'] = True
        # When reusing the parameters trained with multi-model settings,
        # the multi-model switch should be turned off before loading.
        config_dict['multi_model'] = False
        config_dict['num_models'] = 1
    else:
        config_dict = utils.read_json(Path(f"./configs/{apargs.task}.json"))
        config_dict.update({key: getattr(apargs, key) for key in vars(apargs)})
    config = utils.Config(config_dict)

    # Initialize logger instance.
    logger = utils.Logger(config.tag, config.resume, task=config.task)
    if logger.resume:
        config.random_seed = logger.config_resume.random_seed
    elif config.random_seed is None:
        config.random_seed = np.random.randint(1, 1000000)
    torch.manual_seed(config.random_seed)
    logger.save_config(config)

    cuda_enabled = config.gpu and torch.cuda.is_available()
    if cuda_enabled:
        try:
            torch.multiprocessing.set_start_method('spawn', force=True )
        except:
            pass
        torch.cuda.manual_seed(config.random_seed)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    assert config.task in ["mnist", "nmnist"]
    if config.task == "mnist":
        data_loaders = mnist_dataset.load_loader(config=config, num_workers=config.num_workers,
                                                 batch_size=config.batch_size, test_batch_size=config.test_batch_size)
    elif config.task == "nmnist":
        data_loaders = nmnist_dataset.load_loader(config=config, num_workers=config.num_workers,
                                                  batch_size=config.batch_size, test_batch_size=config.test_batch_size,
                                                  time_length=config.time_length)

    trainer = Trainer(config, data_loaders=data_loaders, logger=logger, gpu=cuda_enabled, task=config.task)

    # seed again to ensure the model parameters are sampled right after the
    # seed setting.
    torch.manual_seed(config.random_seed)
    if cuda_enabled:
        torch.cuda.manual_seed(config.random_seed)
    trainer.make_model(config)

    if apargs.evaluation_mode:
        param_path = Path(f"./logs/{apargs.tag}/m0_best_model.pt")
        if cuda_enabled:
            trainer.load_model(torch.load(param_path))
        else:
            trainer.load_model(torch.load(param_path, map_location=torch.device('cpu')))
        trainer.test()
    else:
        if logger.resume:
            param_path_model = logger.log_dir / "last_model.pt"
            param_path_optim = logger.log_dir / "last_optim.pt"
            try:
                if cuda_enabled:
                    trainer.load_model(torch.load(param_path_model))
                    trainer.load_optim(torch.load(param_path_optim))
                else:
                    trainer.load_model(torch.load(param_path_model, map_location=torch.device('cpu')))
                    trainer.load_optim(torch.load(param_path_optim, map_location=torch.device('cpu')))

                # for randomness in resuming
                seed_keep = np.random.randint(1, 100000)
                torch.manual_seed(seed_keep)
                if cuda_enabled:
                    torch.cuda.manual_seed(seed_keep)
            except:
                logger.resume = False
        else:
            trainer.save_model('init')

        trainer.make_scheduler()
        trainer.run(config)

if __name__ == "__main__":
    main()
