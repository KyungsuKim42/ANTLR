import torch.optim
import math
import pickle
from pathlib import Path
import utils
import time

import pkbar

from antlr import *

class Trainer():
    def __init__(self, config, data_loaders, logger, gpu=False, task='nmnist') :
        """
        Initialize Trainer class.
        Args
            config : Configuration object that contains all the configurations.
            data_loaders : A list of data_loaders containing train, test, test loader.
            plotter : A plotter that plots output/states/losses.
            logger : A logger that logs the progress of the training.
            task : Which task is being solved by this trainer. ['mnist', 'nmnist']
        """
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = data_loaders
        if config.multi_model:
            self.train_num_data = len(self.train_loader[0].dataset)
            self.valid_num_data = len(self.valid_loader[0].dataset)
            self.test_num_data = len(self.test_loader.dataset)
        else:
            self.train_num_data = len(self.train_loader.dataset)
            self.valid_num_data = len(self.valid_loader.dataset)
            self.test_num_data = len(self.test_loader.dataset)
        self.data_loaders = data_loaders
        self.logger = logger
        self.task = task
        self.cuda_enabled = gpu
        self.resume = self.logger.resume
        return

    def make_model(self, model_config):
        """
        Instantiate SNN models following configurations.
        Attribute of each models are set by using setattr().
        Args
            model_config_list : list of configuration dictionaries.
                Ex)model_config_list[0]['ste_type'] = "exp"
            same_init : A boolean variable that indicates wheter using same initial
                parameters for every models or not.
        """
        model = ListSNNMulti(model_config)
        self.model = model

        if model.optim_name == "sgd":
            optim = torch.optim.SGD(model.parameters(), lr=model.learning_rate,
                                    momentum=model.momentum, weight_decay=model.weight_decay)
        elif model.optim_name == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=model.learning_rate,
                                     weight_decay=model.weight_decay)
        self.optim = optim
        return

    def make_scheduler(self):
        """
        Make learning rate scheduler for each optimizers.
        """
        scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                    self.config.lr_step_size,
                                                    self.config.lr_gamma)
        self.scheduler = scheduler
        return

    def run(self, config):
        """
        Train each model in model_list.
        Args
            config : model configuration.
        """
        # Initialize result holder and plotter.
        loss_dict = dict()

        model = self.model
        optim = self.optim
        scheduler = self.scheduler

        if self.resume:
            if config.multi_model:
                best_valid_acc = np.array(self.logger.valid_df['acc'].tolist()).max(0)
                best_valid_acc_first = np.array(self.logger.valid_df['acc_first'].tolist()).max(0)
            else:
                best_valid_acc = self.logger.valid_df['acc'].to_numpy().max()
                best_valid_acc_first = self.logger.valid_df['acc_first'].to_numpy().max()

            current_epoch = self.logger.valid_df['epoch'].to_numpy().max() + 1
        else:
            if config.multi_model:
                best_valid_acc = np.zeros(config.num_models)
                best_valid_acc_first = np.zeros(config.num_models)
            else:
                best_valid_acc = 0
                best_valid_acc_first = 0

            current_epoch = 0

        for epoch in range(current_epoch, current_epoch + config.epoch):
            train_loss, train_acc, train_acc_first, kbar = self.run_epoch('train', model, optim, scheduler, epoch)
            train_nums_total = np.array(self.total_num_spike_total).sum() / self.train_num_data
            train_nums_nec = np.array(self.total_num_spike_nec).sum() / self.train_num_data
            scheduler.step()
            self.logger.log_train_df(epoch, train_loss, train_acc, train_acc_first, self.total_num_spike_total, self.total_num_spike_nec, self.min_first_stime_min, self.mean_first_stime_mean, model.multi_model)
            self.model.clean_state()

            valid_loss, valid_acc, valid_acc_first = self.run_epoch('valid', model)
            kbar.add(0, values=[("v_loss", valid_loss), ("v_acc", valid_acc), ("v_acc_first", valid_acc_first)])
            valid_nums_total = np.array(self.total_num_spike_total).sum() / self.valid_num_data
            valid_nums_nec = np.array(self.total_num_spike_nec).sum() / self.valid_num_data
            self.logger.log_valid_df(epoch, valid_loss, valid_acc, valid_acc_first, self.total_num_spike_total, self.total_num_spike_nec, self.min_first_stime_min, self.mean_first_stime_mean, model.multi_model)
            self.model.clean_state()

            if config.multi_model:
                for m in (valid_acc.cpu().numpy() > best_valid_acc).nonzero()[0].tolist():
                    self.save_multi_model('best', m)
                    best_valid_acc[m] = valid_acc[m]
                for m in (valid_acc_first.cpu().numpy() > best_valid_acc_first).nonzero()[0].tolist():
                    self.save_multi_model('best_first', m)
                    best_valid_acc_first[m] = valid_acc_first[m]
            else:
                if valid_acc > best_valid_acc:
                    self.save_model('best')
                    best_valid_acc = valid_acc
                if valid_acc_first > best_valid_acc_first:
                    self.save_model('best_first')
                    best_valid_acc_first = valid_acc_first

            self.save_model('last')
            self.save_optim('last')

            self.model.clean_state()

        return

    def run_epoch(self, mode, model, optim=None, scheduler=None, epoch=None):

        assert mode in ['train', 'valid', 'test']
        if mode == 'train':
            loader = self.train_loader
            num_data = self.train_num_data

            if model.multi_model:
                target_iter = len(loader[0])
            else:
                target_iter = len(loader)
            kbar = pkbar.Kbar(target=target_iter,  epoch=epoch,
                              num_epochs=self.config.epoch,
                              width=16, always_stateful=False)
        elif mode == 'valid':
            loader = self.valid_loader
            num_data = self.valid_num_data
        elif mode == 'test':
            loader = self.test_loader
            num_data = self.test_num_data

        progress = 0
        total_loss = 0
        total_correct = 0
        total_correct_first = 0
        # input.shape = [N, time_length, num_in_features]
        # target.shape = [N, time_length, num_target_features]

        if model.multi_model and not mode =='test':
            if self.config.evaluation_mode:
                loader = loader[0]
            else:
                loader = zip(*loader)

        for batch_idx, inp_tar in enumerate(loader):
            if model.multi_model:
                if mode == 'test' or self.config.evaluation_mode:
                    input = inp_tar[0].unsqueeze(0).repeat(model.num_models, *[1 for i in range(inp_tar[0].dim())])
                    target = inp_tar[1].unsqueeze(0).repeat(model.num_models, *[1 for i in range(inp_tar[1].dim())])
                else:
                    input = torch.stack([item[0] for item in inp_tar])
                    target = torch.stack([item[1] for item in inp_tar])
            else:
                input = inp_tar[0]
                target = inp_tar[1]

            if self.cuda_enabled:
                input = input.cuda()
                target = target.cuda()

            if self.task == "mnist":
                if model.multi_model:
                    input = input.reshape(input.shape[0] * input.shape[1], -1)
                    input = self.float2spikes(input, model.time_length, self.config.max_input_timing,
                                              self.config.min_input_timing, type = 'latency',
                                              stochastic=False, last=False, skip_zero=True)
                    input = input.reshape(model.num_models, int(input.shape[0] / model.num_models), *input.shape[1:])
                else:
                    input = input.reshape(input.shape[0], -1)
                    input = self.float2spikes(input, model.time_length, self.config.max_input_timing,
                                              self.config.min_input_timing, type = 'latency',
                                              stochastic=False, last=False, skip_zero=True)

            # Run forward pass.
            output = model(input)

            if mode == 'train':
                # Backward and update.
                optim.zero_grad()
                if self.config.target_type == 'latency':
                    model.backward_custom(target)
                else:
                    assert self.config.target_type == 'count'
                    target_spike = self.label2spikes(target.reshape(-1))
                    # model_batch x time x neuron
                    model.backward_custom(target_spike)
                optim.step()
            else:
                if self.config.target_type == 'latency':
                    model.calc_loss(target.reshape(-1))
                else:
                    assert self.config.target_type == 'count'
                    target_spike = self.label2spikes(target.reshape(-1))
                    # model_batch x time x neuron
                    model.calc_loss(target_spike)

            loss = model.loss

            batch_size = target.shape[-1]
            total_loss += loss * batch_size

            num_spike_total = model.num_spike_total
            num_spike_nec = model.num_spike_nec
            first_stime_min = model.first_stime_min
            first_stime_mean = model.first_stime_mean
            if batch_idx == 0:
                self.total_num_spike_total = num_spike_total
                self.total_num_spike_nec = num_spike_nec
                self.min_first_stime_min = first_stime_min
                self.mean_first_stime_mean = first_stime_mean
            else:
                if model.multi_model:
                    self.total_num_spike_total = [(np.array(num_spike_total[i]) + np.array(self.total_num_spike_total[i])).tolist() for i in range(len(num_spike_total))]
                    self.total_num_spike_nec = [(np.array(num_spike_nec[i]) + np.array(self.total_num_spike_nec[i])).tolist() for i in range(len(num_spike_nec))]
                    self.min_first_stime_min = [min(x, y) for x, y in zip(self.min_first_stime_min, first_stime_min)]
                    self.mean_first_stime_mean = ((np.array(self.mean_first_stime_mean) * progress + np.array(first_stime_mean) * batch_size) / (progress + batch_size)).tolist()
                else:
                    self.total_num_spike_total = [num_spike_total[i] + self.total_num_spike_total[i] for i in range(len(num_spike_total))]
                    self.total_num_spike_nec = [num_spike_nec[i] + self.total_num_spike_nec[i] for i in range(len(num_spike_nec))]
                    self.min_first_stime_min = min(self.min_first_stime_min, first_stime_min)
                    self.mean_first_stime_mean = (self.mean_first_stime_mean * progress + first_stime_mean * batch_size) / (progress + batch_size)

            pred_class = self.spikes2label(output, 'count')
            pred_class_first = self.spikes2label(output, 'first')
            if model.multi_model:
                num_correct = (pred_class.reshape(target.shape) == target).sum(1).float()
                num_correct_first = (pred_class_first.reshape(target.shape) == target).sum(1).float()
                total_correct += num_correct
                total_correct_first += num_correct_first
            else:
                num_correct = (pred_class == target).sum().float()
                num_correct_first = (pred_class_first == target).sum().float()
                total_correct += float(num_correct.item())
                total_correct_first += float(num_correct_first.item())

            current_acc_count = num_correct / batch_size
            current_acc_first = num_correct_first / batch_size

            progress += batch_size
            # if mode == 'train':
            #     self.logger.log_train(model.multi_model, epoch, progress, loss, num_spike_total, num_spike_nec, first_stime_min, first_stime_mean, num_correct, num_correct_first, batch_size, model.term_length, (batch_idx % self.config.log_interval == 0))

            if mode == "train":
                kbar.update(batch_idx+1, values=[("loss", loss),
                                               ("acc", current_acc_count),
                                               ("acc_first", current_acc_first)])

        if mode == "train":
            return (total_loss / progress), (total_correct / progress), (total_correct_first / progress), kbar
        else:
            return (total_loss / progress), (total_correct / progress), (total_correct_first / progress)

    def load_model(self, param_dict):
        self.model.load_state_dict(param_dict)
        return

    def save_model(self, tag):
        self.logger.save_model(self.model, tag)
        return

    def save_multi_model(self, tag, model_id):
            self.logger.save_multi_model(self.model, tag, model_id)
            return

    def load_optim(self, param_dict):
        self.optim.load_state_dict(param_dict)
        return

    def save_optim(self, tag):
        self.logger.save_optim(self.optim, tag)
        return

    def test(self):
        test_loss, test_acc_most, test_acc_earliest = self.run_epoch('test', self.model)
        print(f"test_loss : {test_loss:.4f} | test_acc_most : {test_acc_most:.4f} | " \
              f"test_acc_earliest : {test_acc_earliest}")
        return

    def spikes2label(self, spmat, decision_type = 'count'):
        """
        Args
            spmat : [batch x time x feature]
        Return
            label : [batch]
        """
        if decision_type == 'count':
            label = spmat.sum(1).max(1).indices
        elif decision_type == 'first':
            decreasing_output = spmat * torch.arange(spmat.shape[1], 0, -1).view(1, -1, 1)
            max_each_neuron = decreasing_output.max(dim=1).values
            # batch x feature
            label = max_each_neuron.max(dim=1).indices
        return label

    def label2spikes(self, label):
        """
        Generate target spike train based on the class label.
        Target spikes are evenly distributed according to the target spike number.
        Args:
            label : target label. shape = [batch]
        Return:
            spmat : shape = [batch, time, feature]
        """
        pos_num = self.model.max_target_spikes
        neg_num = self.model.min_target_spikes

        T = self.model.time_length
        spmat = torch.zeros([label.numel(), T, 10])
        spmat[:,0,:] = neg_num

        for i in range(label.numel()):
            spmat[i, 0, label[i]] = pos_num

        return spmat

    def float2spikes(self, flmat, time_length, num_max_spikes, num_min_spikes, type = 'stretch', stochastic=False, last=False, skip_zero=True):
        """
        Args
            flmat : float matrix [batch x feature] in [0, 1]
        Outputs
            spmat : spike matrix [batch x time x feature]
        """
        batch_size = flmat.size(0)
        if not stochastic:
            if type == 'stretch':
                flmat_min_to_max = (num_max_spikes - num_min_spikes) * flmat + num_min_spikes
                # batch x features
                if skip_zero:
                    flmat_min_to_max[flmat == 0] = 0
                flmat_min_to_max = flmat_min_to_max.view(batch_size, 1, -1)

                flmat_increasing_from_zero = flmat_min_to_max * torch.arange(0, 1 + 1/time_length, 1/time_length).view(1, -1, 1)

                if last:
                    flmat_step = flmat_increasing_from_zero.floor()
                else:
                    flmat_step = flmat_increasing_from_zero.ceil()
                spmat = flmat_step[:, 1:, :] - flmat_step[:, 0:-1, :]

            elif type == 'shift_back':
                flmat_min_to_max = (num_max_spikes - num_min_spikes) * flmat + num_min_spikes
                # batch x features
                if skip_zero:
                    flmat_min_to_max[flmat == 0] = 0
                flmat_min_to_max = flmat_min_to_max.view(batch_size, 1, -1)
                flmat_increasing_from_zero = (flmat_min_to_max - num_max_spikes) + num_max_spikes * torch.arange(0, 1 + 1/time_length, 1/time_length).view(1, -1, 1)
                flmat_increasing_from_zero = torch.clamp(flmat_increasing_from_zero, 0, num_max_spikes)

                if last:
                    flmat_step = flmat_increasing_from_zero.floor()
                else:
                    flmat_step = flmat_increasing_from_zero.ceil()
                spmat = flmat_step[:, 1:, :] - flmat_step[:, 0:-1, :]

            elif type == 'latency': # num_max_spikes -> max_timing, num_min_spikes -> min_timing
                flmat_min_to_max = (num_max_spikes - num_min_spikes) * flmat + num_min_spikes
                # batch x features
                if skip_zero:
                    flmat_min_to_max[flmat == 0] = -1
                flmat_min_to_max = flmat_min_to_max.view(batch_size, 1, -1)
                spmat = (torch.ones(flmat_min_to_max.shape) * torch.arange(0, time_length).view(1, -1, 1) == flmat_min_to_max.int()).float()

        return spmat

    def visualize_activity(self, model, input, target):
        """
        Visualize the internal activity during evaluation of the network.
        """
        batch_idxs = [0]

        batch_size = target.numel()
        state_v = [torch.stack(v).reshape(model.term_length,batch_size,-1) if len(v)>0 else None for v in model.state_v]
        len_state_v = [1 for sv in state_v if sv != None]
        state_s = [torch.stack(s).reshape(model.term_length,batch_size,-1) for s in model.state_s]

        fig, ax = plt.subplots(nrows = len(len_state_v) + len(state_s) + 1, ncols = len(batch_idxs), squeeze = False, num = 's, v each layer', figsize = (15,10))

        ## Reshape the nmnist input.
        if self.task == 'nmnist':
            input = input.reshape(batch_size, model.time_length, -1)
            input = input.permute(0,2,1)

        for i in range(len(batch_idxs)):
            bidx = batch_idxs[i]
            if self.task == 'nmnist':
                ax[0, i].scatter(input.cpu().reshape(batch_size,model.time_length,-1)[bidx].nonzero()[:, 0],
                                 input.cpu().reshape(batch_size,model.time_length,-1)[bidx].nonzero()[:, 1], s=1, c='r')
            else:
                ax[0, i].scatter(input.cpu().reshape(batch_size,model.time_length,-1)[bidx].nonzero()[:, 0],
                                 input.cpu().reshape(batch_size,model.time_length,-1)[bidx].nonzero()[:, 1], s=1, c='r')
            ax[0, i].set_xlim([-1, model.time_length + 1])
            # ax[0, i].set_ylim([-1, ])
            sidx = 0
            for l in range(len(state_v)):
                if state_v[l] is not None:
                    sidx += 1
                    _, batch_size, num_neuron = state_v[l].shape
                    ax[sidx, i].plot(state_v[l][:, bidx, :500].cpu())
                    ax[sidx, i].axhline(y=1, c='k')
                    # ax[1+2*l+0, i].set_ylim([state_v[l].min(), 2])
                    ax[sidx, i].set_ylim([-4, 4])
                    ax[sidx, i].set_xlim([-1, model.time_length + 1])

                sidx += 1
                _, batch_size, num_neuron = state_s[l].shape
                ax[sidx, i].set_title(state_s[l][:, bidx, :].sum().item())
                ax[sidx, i].scatter(state_s[l][:, bidx, :].cpu().nonzero()[:, 0], state_s[l][:, bidx, :].cpu().nonzero()[:, 1], s=1, c='r')
                ax[sidx, i].set_ylim([-1, num_neuron + 1])
                ax[sidx, i].set_xlim([-1, model.time_length + 1])

        fig.tight_layout()
        plt.show()
