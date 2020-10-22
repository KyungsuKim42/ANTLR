import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import datetime
import sys
import shutil
from pathlib import Path
import os
import logging
import argparse
import json
import pickle
pickle.HIGHEST_PROTOCOL = 4
import pandas as pd

class Config():
    """
    Dummy class for configuration container.
    Ex) dict = {"abc":1, "bcd":3.5}
        cfg = Config(dict), then
        cfg.abc = 1, cfg.bcd=3.5.
    """
    def __init__(self, dict):
        """
        Update class object's properties as given dictionary.
        """
        self.__dict__.update(dict)
        return

class Logger():
    """
    A logger that logs following thigs.
        * Training Progress.
            * Model Parameters.
            * Training Loss.
        * Training Data.
            * Input spike train
            * Target spike train.
    """
    def __init__(self, exp_name, resume, task='nmnist'):
        """
        Initialize a logger. Generates a new log directory.
        Args
            exp_name : Name of the experiment.
        """
        self.resume = resume

        self.name_tag = exp_name
        time_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_tag = f"{self.name_tag}"
        if not os.path.isdir(f"./logs"):
            os.mkdir(f"./logs")
        self.log_dir = Path(f"./logs/{self.name_tag}")
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
            self.resume = False

        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(self.log_dir / "log.txt")
        print_targets = stream_handler, file_handler
        logging.basicConfig(format='%(message)s', level=logging.INFO,
                            handlers=print_targets)
        self.print_log(f"Logging at {exp_name}")
        self._init_train_df()
        self._init_valid_df()
        self._init_test_df()
        self._init_train_log_dict()
        self._init_valid_log_dict()
        self._init_test_log_dict()

        if self.resume:
            with open(self.log_dir / 'config.json', "rb") as f:
                self.config_resume = Config(json.load(f))

        self.tic = time.time()
        return

    def print_log(self, msg):
        logging.info(msg)
        return

    def _init_train_df(self):
        if self.resume:
            try:
                with open(self.log_dir / 'train.pkl', 'rb') as f:
                    train_df = pickle.load(f)
                self.train_df = train_df
            except:
                self.resume = False
                self.train_df = pd.DataFrame()
        else:
            self.train_df = pd.DataFrame()
        return

    def _init_valid_df(self):
        if self.resume:
            try:
                with open(self.log_dir / 'valid.pkl', 'rb') as f:
                    valid_df = pickle.load(f)
                self.valid_df = valid_df
            except:
                self.resume = False
                self.valid_df = pd.DataFrame()
        else:
            self.valid_df = pd.DataFrame()
        return

    def _init_test_df(self):
        if self.resume:
            try:
                with open(self.log_dir / 'test.pkl', 'rb') as f:
                    test_df = pickle.load(f)
                self.test_df = test_df
            except:
                self.resume = False
                self.test_df = pd.DataFrame()
        else:
            self.test_df = pd.DataFrame()
        return

    # def _init_train_log_dict(self):
    #     self.log_dict = dict()
    #     self.log_dict['epoch'] = list()
    #     self.log_dict['progress'] = list()
    #     self.log_dict['ratio'] = list()
    #     self.log_dict['loss'] = list()
    #     self.log_dict['num_spike_total'] = list()
    #     self.log_dict['num_spike_nec'] = list()
    #     self.log_dict['first_stime_min'] = list()
    #     self.log_dict['first_stime_mean'] = list()
    #     self.log_dict['acc'] = list()
    #     self.log_dict['acc_first'] = list()
    #     self.log_dict['batch_size'] = list()
    #     return

    def _init_train_log_dict(self):
        self.train_log_dict = dict()
        self.train_log_dict['epoch'] = list()
        self.train_log_dict['loss'] = list()
        self.train_log_dict['acc'] = list()
        self.train_log_dict['acc_first'] = list()
        self.train_log_dict['num_spike_total'] = list()
        self.train_log_dict['num_spike_nec'] = list()
        self.train_log_dict['first_stime_min'] = list()
        self.train_log_dict['first_stime_mean'] = list()
        return

    def _init_valid_log_dict(self):
        self.valid_log_dict = dict()
        self.valid_log_dict['epoch'] = list()
        self.valid_log_dict['loss'] = list()
        self.valid_log_dict['acc'] = list()
        self.valid_log_dict['acc_first'] = list()
        self.valid_log_dict['num_spike_total'] = list()
        self.valid_log_dict['num_spike_nec'] = list()
        self.valid_log_dict['first_stime_min'] = list()
        self.valid_log_dict['first_stime_mean'] = list()
        return

    def _init_test_log_dict(self):
        self.test_log_dict = dict()
        self.test_log_dict['epoch'] = list()
        self.test_log_dict['loss'] = list()
        self.test_log_dict['acc'] = list()
        self.test_log_dict['acc_first'] = list()
        self.test_log_dict['num_spike_total'] = list()
        self.test_log_dict['num_spike_nec'] = list()
        self.test_log_dict['first_stime_min'] = list()
        self.test_log_dict['first_stime_mean'] = list()
        return

    def save_spike_train_data(self, input, target, repeat):
        """
        Save input and target spike train as .npy file.
        """
        np.save(self.log_dir/f"input_{self.name_tag}_repeat{repeat}.npy",input)
        np.save(self.log_dir/f"target_{self.name_tag}_repeat{repeat}.npy",target)
        return

    def save_config(self, config):
        """
        Copy and paste the configuration file used for the experiment to the
        log_dir.
        """
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(config.__dict__, f)
        return

    def save_model(self, model, tag):
        """
        Save the best trained model so far.
        Args
            model : SNN model to save.
        """
        model_fname = self.log_dir / f"{tag}_model.pt"
        torch.save(model.state_dict(), model_fname)
        # self.print_log(f"{tag} model is saved.")
        return

    def save_optim(self, optim, tag):
        """
        Save the best trained model so far.
        Args
            model : SNN model to save.
        """
        optim_fname = self.log_dir / f"{tag}_optim.pt"
        torch.save(optim.state_dict(), optim_fname)
        # self.print_log(f"{tag} optim is saved.")
        return

    def save_multi_model(self, model, tag, model_id):
        """
        Save the best trained model so far.
        Args
            model : SNN model to save.
        """
        model_fname = self.log_dir / f"m{model_id}_{tag}_model.pt"
        state_dict = {key.replace(f'm{model_id}_', ''): model.state_dict()[key] for key in model.state_dict().keys() if f'm{model_id}_' in key}
        torch.save(state_dict, model_fname)
        self.print_log(f"m{model_id}: {tag} model is saved.")
        return

    def log_train(self, multi_model, epoch, progress, loss, num_spike_total, num_spike_nec, first_stime_min, first_stime_mean, num_correct, num_correct_first, batch_size, term_length, print_log = False):
        toc = time.time()
        len_train = 50000
        if multi_model:
            if print_log:
                loss_str = ([float(f'{item:.4f}') for item in loss])
                rateacc_str = [float(f'{(item/batch_size).mean()*100.0:5.2f}') for item in num_correct]
                firstacc_str = [float(f'{(item/batch_size).mean()*100.0:5.2f}') for item in num_correct_first]
                msg = f"{toc-self.tic:.3f}\te {epoch}\t{progress}\tp {progress/len_train:.4f}\tl {loss_str}\ttot {num_spike_total}\tnec {num_spike_nec}\trate {rateacc_str}\tfirst {firstacc_str}\tterm {term_length}"
                logging.info(msg)

            self.log_dict['epoch'].append(epoch)
            self.log_dict['progress'].append(progress)
            self.log_dict['ratio'].append(progress/len_train)
            self.log_dict['loss'].append(loss.cpu().tolist())
            self.log_dict['num_spike_total'].append(num_spike_total)
            self.log_dict['num_spike_nec'].append(num_spike_nec)
            self.log_dict['first_stime_min'].append(first_stime_min)
            self.log_dict['first_stime_mean'].append(first_stime_mean)
            self.log_dict['acc'].append((num_correct/batch_size).cpu().tolist())
            self.log_dict['acc_first'].append((num_correct_first/batch_size).cpu().tolist())
            self.log_dict['batch_size'].append(batch_size)
        else:
            if print_log:
                msg = f"{toc-self.tic:.3f}\te {epoch}\t{progress}\tp {progress/len_train:.4f}\tl {loss:.4f}\ttot {num_spike_total}\tnec {num_spike_nec}\tmin {first_stime_min:3.0f}\tmean {first_stime_mean:5.1f}\trate {(num_correct/batch_size)*100.0:5.2f}\tfirst {(num_correct_first/batch_size)*100.0:5.2f}\tterm {term_length}"
                logging.info(msg)

            self.log_dict['epoch'].append(epoch)
            self.log_dict['progress'].append(progress)
            self.log_dict['ratio'].append(progress/len_train)
            self.log_dict['loss'].append(loss)
            self.log_dict['num_spike_total'].append(num_spike_total)
            self.log_dict['num_spike_nec'].append(num_spike_nec)
            self.log_dict['first_stime_min'].append(first_stime_min)
            self.log_dict['first_stime_mean'].append(first_stime_mean)
            self.log_dict['acc'].append(float(num_correct)/batch_size)
            self.log_dict['acc_first'].append(float(num_correct_first)/batch_size)
            self.log_dict['batch_size'].append(batch_size)

        self.tic = time.time()
        return

    # def log_train_df(self):
    #     df_epoch = pd.DataFrame(self.log_dict)
    #     self.train_df = self.train_df.append(df_epoch.round(6), ignore_index=True)
    #     self.train_df.to_csv(self.log_dir / 'train.csv', index=True)
    #     self.train_df.to_pickle(self.log_dir / 'train.pkl')
    #     self._init_train_log_dict()
    #     return

    def log_train_df(self, epoch, loss, acc, acc_first, num_spike_total, num_spike_nec, first_stime_min, first_stime_mean, multi_model):
        self.train_log_dict['epoch'].append(epoch)
        if multi_model:
            self.train_log_dict['loss'].append(loss.tolist())
            self.train_log_dict['acc'].append(acc.tolist())
            self.train_log_dict['acc_first'].append(acc_first.tolist())
        else:
            self.train_log_dict['loss'].append(loss)
            self.train_log_dict['acc'].append(acc)
            self.train_log_dict['acc_first'].append(acc_first)
        self.train_log_dict['num_spike_total'].append(num_spike_total)
        self.train_log_dict['num_spike_nec'].append(num_spike_nec)
        self.train_log_dict['first_stime_min'].append(first_stime_min)
        self.train_log_dict['first_stime_mean'].append(first_stime_mean)
        self.train_df = self.train_df.append(pd.DataFrame(self.train_log_dict).round(6), ignore_index=True)
        self.train_df.to_csv(self.log_dir / 'train.csv', index=True)
        self.train_df.to_pickle(self.log_dir / 'train.pkl')
        self._init_train_log_dict()
        return

    def log_valid_df(self, epoch, loss, acc, acc_first, num_spike_total, num_spike_nec, first_stime_min, first_stime_mean, multi_model):
        self.valid_log_dict['epoch'].append(epoch)
        if multi_model:
            self.valid_log_dict['loss'].append(loss.tolist())
            self.valid_log_dict['acc'].append(acc.tolist())
            self.valid_log_dict['acc_first'].append(acc_first.tolist())
        else:
            self.valid_log_dict['loss'].append(loss)
            self.valid_log_dict['acc'].append(acc)
            self.valid_log_dict['acc_first'].append(acc_first)
        self.valid_log_dict['num_spike_total'].append(num_spike_total)
        self.valid_log_dict['num_spike_nec'].append(num_spike_nec)
        self.valid_log_dict['first_stime_min'].append(first_stime_min)
        self.valid_log_dict['first_stime_mean'].append(first_stime_mean)
        self.valid_df = self.valid_df.append(pd.DataFrame(self.valid_log_dict).round(6), ignore_index=True)
        self.valid_df.to_csv(self.log_dir / 'valid.csv', index=True)
        self.valid_df.to_pickle(self.log_dir / 'valid.pkl')
        self._init_valid_log_dict()
        return

    def log_test_df(self, epoch, loss, acc, acc_first, num_spike_total, num_spike_nec, first_stime_min, first_stime_mean, multi_model):
        self.test_log_dict['epoch'].append(epoch)
        if multi_model:
            self.test_log_dict['loss'].append(loss.tolist())
            self.test_log_dict['acc'].append(acc.tolist())
            self.test_log_dict['acc_first'].append(acc_first.tolist())
        else:
            self.test_log_dict['loss'].append(loss)
            self.test_log_dict['acc'].append(acc)
            self.test_log_dict['acc_first'].append(acc_first)
        self.test_log_dict['num_spike_total'].append(num_spike_total)
        self.test_log_dict['num_spike_nec'].append(num_spike_nec)
        self.test_log_dict['first_stime_min'].append(first_stime_min)
        self.test_log_dict['first_stime_mean'].append(first_stime_mean)
        self.test_df = self.test_df.append(pd.DataFrame(self.test_log_dict).round(6), ignore_index=True)
        self.test_df.to_csv(self.log_dir / 'test.csv', index=True)
        self.test_df.to_pickle(self.log_dir / 'test.pkl')
        self._init_test_log_dict()
        return

    def log_loss(self, loss, model_tag, epoch, repeat):
        """
        Log the loss.
        """
        logging.info(f"{repeat}th repeat, {epoch:03d}th epoch --"
                     f" {model_tag}'s loss : {loss:.6f}'")
        return

    def log_prog(self, mode, epoch, loss, acc, acc_first):
        """
        Log the progress.
        """
        str = f"{mode} epoch{epoch:02d} : loss = {loss}, acc = {acc}, acc_first = {acc_first}"

        logging.info(str)

        return

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--index', default=None)
    args = parser.parse_args()
    return args

def read_json(config_fname):
    with open(config_fname, "rb") as f:
        configs = json.load(f)
    return configs
