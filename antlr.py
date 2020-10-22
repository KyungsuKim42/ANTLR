import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable, Function
from torch.nn import init
import torch.optim.sgd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import gc

class ListSNNMulti(nn.Module):
    """simple SNN model without SNNCell"""
    def __init__(self, model_config):
        super(ListSNNMulti, self).__init__()
        print("SNNmodel_parallel instantiated")
        if hasattr(model_config, '__dict__'):
            self.__dict__.update(model_config.__dict__)
        else:
            self.__dict__.update(model_config)
        if not self.multi_model:
            assert self.num_models == 1
        self._init_kernels()
        self._init_layers()

    def _init_kernels(self):
        assert self.target_type == 'train' or self.target_type == 'count' or self.target_type == 'latency', "target type should be either \'train\' or \'count\', or \'latency\'"
        if self.target_type == 'train' or self.target_type == 'count':
            if self.target_type == 'train':
                # self.alpha_exp = 0.9
                self.alpha_extend = 200
            elif self.target_type == 'count':
                self.alpha_exp = 1.0
                self.alpha_extend = 0
            kernel = torch.pow(self.alpha_exp, torch.arange(min(self.time_length, 1000)).float())
            kernel = kernel[kernel > 1e-06].view(1, 1, -1)

            kernel_shifted_front = torch.cat((kernel, torch.zeros(1,1,2)), 2)
            kernel_shifted_back = torch.cat((torch.zeros(1,1,2), kernel), 2)
            kernel_prime = (kernel_shifted_front - kernel_shifted_back) / 2

            self.alpha_kernel = kernel
            self.alpha_kernel_prime = kernel_prime

        # For calculating double-exponential kernel, we calculate each
        # timestep's synaptic current (exponentially decays toward future) and
        # accumulate the decayed effect (exponentially decays toward past) of
        # those synaptic currents to the current timestep.
        epsilon = torch.zeros(self.time_length)
        # epsilon = torch.zeros(1000)
        for t in range(epsilon.numel()):
            current_trace = torch.pow(self.alpha_i, torch.arange(t+1).float())
            trace_weight = torch.pow(self.alpha_v, torch.arange(t, -1, -1).float())
            epsilon[t] = (current_trace * trace_weight).sum()
        # epsilon = epsilon[epsilon.abs() > 1e-06]

        if self.beta_auto:
            self.beta_i = 1.0
            self.beta_v = 1.0 / epsilon.max()
            self.beta_bias = 1.0 / epsilon.max()
            print(f'calculated beta_v : {self.beta_v}')
            print(f'calculated beta_bias : {self.beta_bias}')
            epsilon *= self.beta_i * self.beta_v
        else:
            epsilon *= self.beta_i * self.beta_v

        epsilon_shifted_front = torch.cat((epsilon, torch.zeros(2)))
        epsilon_shifted_back = torch.cat((torch.zeros(2), epsilon))
        epsilon_prime = (epsilon_shifted_front - epsilon_shifted_back)/2

        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime
        return

    def _init_layers(self):
        self.num_layer = np.size(self.network_size) - 1

        self.state_v_bs = list()
        self.layers = list()
        self.fmap_shape_list = list()
        self.fmap_type_list = list()


        if self.multi_model:
            for m in range(self.num_models):
                if "x" in self.network_size[0]:
                    in_channels, height, width = [int(item) for item in self.network_size[0].split("x")]
                else:
                    in_channels = int(self.network_size[0])

                for l, layer_spec in enumerate(self.network_size[1:]):
                    if "conv" in layer_spec:
                        raise NotImplementedError
                    elif "fc" in layer_spec:
                        out_channels = int(layer_spec.strip("fc"))
                        layer = torch.nn.Linear(in_channels, out_channels, bias=False)
                        bias = Parameter(torch.Tensor(out_channels))
                        in_channels = out_channels
                        fmap_shape = [in_channels]
                        fmap_type = "fc"
                    elif "apool" in layer_spec:
                        raise NotImplementedError
                    elif "mpool" in layer_spec:
                        raise NotImplementedError
                    elif "flatten" in layer_spec:
                        layer = torch.nn.Flatten()
                        in_channels = in_channels * height * width
                        bias = None
                        height = 1
                        width = 1
                        fmap_shape = [in_channels]
                        fmap_type = "flatten"
                    else:
                        raise ValueError(f"Layer type {layer_spec} is invalid")

                    if m == 0:
                        self.layers.append([layer])
                        self.state_v_bs.append([bias])
                        self.fmap_shape_list.append(fmap_shape)
                        self.fmap_type_list.append(fmap_type)
                    else:
                        self.layers[l].append(layer)
                        self.state_v_bs[l].append(bias)
        else:
            if "x" in self.network_size[0]:
                in_channels, height, width = [int(item) for item in self.network_size[0].split("x")]
            else:
                in_channels = int(self.network_size[0])

            for layer_spec in self.network_size[1:]:
                if "conv" in layer_spec:
                    out_channels, kernel_size = [int(item) for item in layer_spec.strip("conv").split("c")]
                    padding = math.floor(kernel_size / 2)
                    layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                            padding=padding, bias=False)
                    bias = Parameter(torch.Tensor(out_channels))
                    in_channels = out_channels # Height and Width remains the same.
                    fmap_shape = [in_channels, height, width]
                    fmap_type = "conv"
                elif "fc" in layer_spec:
                    out_channels = int(layer_spec.strip("fc"))
                    layer = torch.nn.Linear(in_channels, out_channels, bias=False)
                    # print(layer.weight.mean())
                    bias = Parameter(torch.Tensor(out_channels))
                    in_channels = out_channels
                    fmap_shape = [in_channels]
                    fmap_type = "fc"
                elif "apool" in layer_spec:
                    pool_size = int(layer_spec.strip("apool"))
                    layer = torch.nn.AvgPool2d(pool_size)
                    bias = None
                    height = math.floor(height/pool_size)
                    width = math.floor(width/pool_size)
                    in_channels = out_channels
                    fmap_shape = [in_channels, height, width]
                    fmap_type = "apool"
                elif "mpool" in layer_spec:
                    pool_size = int(layer_spec.strip("mpool"))
                    layer = torch.nn.MaxPool2d(pool_size, return_indices=True)
                    layer.max_index_list = []
                    bias = None
                    height = math.floor(height/pool_size)
                    width = math.floor(width/pool_size)
                    in_channels = out_channels
                    fmap_shape = [in_channels, height, width]
                    fmap_type = "mpool"
                elif "flatten" in layer_spec:
                    layer = torch.nn.Flatten()
                    in_channels = in_channels * height * width
                    bias = None
                    height = 1
                    width = 1
                    fmap_shape = [in_channels]
                    fmap_type = "flatten"
                else:
                    raise ValueError(f"Layer type {layer_spec} is invalid")
                self.layers.append(layer)
                self.state_v_bs.append(bias)
                self.fmap_shape_list.append(fmap_shape)
                self.fmap_type_list.append(fmap_type)

        if self.multi_model:
            for m in range(self.num_models):
                setattr(self, f'm{m}_layers_module', nn.ModuleList([layers[m] for layers in self.layers]))
                setattr(self, f'm{m}_state_v_bs_param', nn.ParameterList([v_b[m] for v_b in self.state_v_bs]))
        else:
            self.layers_module = nn.ModuleList(self.layers)
            self.state_v_bs_param = nn.ParameterList(self.state_v_bs)

        self.reset_parameters()

    def _init_param_grads(self):
        self.weight_grad = list()
        self.bias_grad = list()
        for l, layers in enumerate(self.layers):
            if self.multi_model:
                layer = layers[0]
            else:
                layer = layers

            if self.fmap_type_list[l] in ["fc", "conv"]:
                if self.multi_model:
                    zeros_w = torch.zeros([self.num_models, *layer.weight.size()],
                                           requires_grad = False)
                    zeros_b = torch.zeros([self.num_models, *self.fmap_shape_list[l]])
                else:
                    zeros_w = torch.zeros(layer.weight.size(), requires_grad = False)
                    zeros_b = torch.zeros(self.fmap_shape_list[l])
                self.weight_grad.append(zeros_w)
                self.bias_grad.append(zeros_b)
            else:
                self.weight_grad.append(None)
                self.bias_grad.append(None)

    def reset_parameters(self):
        for l, layer in enumerate(self.layers):
            if self.multi_model:
                for m in range(self.num_models):
                    if isinstance(layer[m], nn.Conv2d) or isinstance(layer[m], nn.Linear):
                        init.constant_(self.state_v_bs[l][m], 0)
                        if hasattr(self, 'normal_weight_init'):
                            if self.normal_weight_init:
                                init.normal_(layer[m].weight, mean=0,
                                             std=self.weight_init_std)
                        if hasattr(self, 'weight_bias'):
                            layer[m].weight =  nn.Parameter(layer[m].weight + self.weight_bias)
            else:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    init.constant_(self.state_v_bs[l], 0)
                    if hasattr(self, 'normal_weight_init'):
                        if self.normal_weight_init:
                            init.normal_(layer.weight, mean=0,
                                         std=self.weight_init_std)
                    if hasattr(self, 'weight_bias'):
                        layer.weight =  nn.Parameter(layer.weight + self.weight_bias)

    def forward(self, input):
        # input: batch x time x neuron
        # state_*: layer x time x batch x neuron
        # output: batch x time x neuron
        with torch.no_grad():
            if self.multi_model:
                assert self.num_models == input.shape[0]
                # input = input.reshape(input.shape[0] * input.shape[1], input.shape[2], input.shape[3])
                self.input = input.reshape(input.shape[0] * input.shape[1], *input.shape[2:])
            else:
                self.input = input

            if self.target_type == 'latency':
                if self.multi_model:
                    self.output_s_cum = torch.zeros(self.num_models * input.shape[1], self.fmap_shape_list[-1][0])
                else:
                    self.output_s_cum = torch.zeros(input.shape[0], self.fmap_shape_list[-1][0])

            self.state_i = list()
            self.state_v = list()
            self.state_v_prime = list()
            self.state_s = list()
            for l, layers in enumerate(self.layers):
                if self.multi_model:
                    layer = layers[0]
                else:
                    layer = layers
                self.state_i.append(list())
                self.state_v.append(list())
                self.state_v_prime.append(list())
                self.state_s.append(list())

            # flush the max_index_list
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.max_index_list = []

            if self.multi_model:
                # Merging parameters
                self.weight_list = []
                self.bias_list = []
                for layer_list in self.layers:
                    if hasattr(layer_list[0], 'weight'):
                        weight = torch.stack([layer.weight for layer in layer_list])
                    else:
                        weight = None
                    self.weight_list.append(weight)
                for state_v_b_list in self.state_v_bs:
                    if state_v_b_list[0] is not None:
                        bias = torch.stack([state_v_b for state_v_b in state_v_b_list]).unsqueeze(1)
                    else:
                        bias = None
                    self.bias_list.append(bias)

            for t in range(self.time_length):
                for l, layers in enumerate(self.layers):
                    if self.multi_model:
                        layer = layers[0]
                    else:
                        layer = layers
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        if (l == 0):
                            if self.multi_model:
                                state_i = torch.bmm(input[:, :, t], self.weight_list[l].permute(0, 2, 1))
                                self.state_i[l].append(state_i.reshape(state_i.shape[0]*state_i.shape[1], *state_i.shape[2:]) * self.beta_i)
                            else:
                                self.state_i[l].append(layer(input[:, t]) * self.beta_i)
                        else:
                            if self.multi_model:
                                state_s = self.state_s[l-1][-1]
                                state_s_rs = state_s.reshape(self.num_models, int(state_s.shape[0]/self.num_models), *state_s.shape[1:])
                                state_i = torch.bmm(state_s_rs, self.weight_list[l].permute(0, 2, 1)) * self.beta_i
                                self.state_i[l].append(state_i.reshape(state_i.shape[0]*state_i.shape[1], *state_i.shape[2:]))
                            else:
                                self.state_i[l].append(layer(self.state_s[l-1][-1]) * self.beta_i)

                        if t != 0:
                            self.state_i[l][-1] += self.state_i[l][t-1] * (1-self.state_s[l][-1]) * self.alpha_i

                        if self.multi_model:
                            state_i = self.state_i[l][-1]
                            state_i_rs = state_i.reshape(self.num_models, int(state_i.shape[0]/self.num_models), *state_i.shape[1:])
                            state_v = state_i_rs * self.beta_v + self.bias_list[l] * self.beta_bias
                            self.state_v[l].append(state_v.reshape(state_v.shape[0]*state_v.shape[1], *state_v.shape[2:]))
                        else:
                            if len(self.state_i[l][-1].shape) == 4:
                                self.state_v[l].append(self.state_i[l][-1] * self.beta_v + self.state_v_bs[l].view(-1,1,1) * self.beta_bias)
                            elif len(self.state_i[l][-1].shape) == 2:
                                self.state_v[l].append(self.state_i[l][-1] * self.beta_v + self.state_v_bs[l] * self.beta_bias)
                            else:
                                raise ValueError("Something's wrong.")

                        if t != 0:
                            self.state_v[l][-1] += self.state_v[l][t-1] * (1-self.state_s[l][-1]) * self.alpha_v

                        if t != 0:
                            self.state_v_prime[l].append(self.state_v[l][-1] - self.state_v[l][-2] * (1 - self.state_s[l][-1]))
                        else:
                            self.state_v_prime[l].append(self.state_v[l][-1])
                        self.state_v_prime[l][-1] = torch.clamp(self.state_v_prime[l][-1], min=1e-2)

                        self.state_s[l].append(self.act(self.state_v[l][-1]))


                    # Average pooling or Max pooling.
                    elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.Flatten):
                        # It is assumed that the first layer is always conv or fc.
                        if l == 0:
                            self.state_s[l].append(layer(self.input[:, t]))
                        else:
                            self.state_s[l].append(layer(self.state_s[l-1][-1]))
                    elif isinstance(layer, nn.MaxPool2d):
                        # It is assumed that the first layer is always conv or fc.
                        pool_result, max_index = layer(self.state_s[l-1][-1])
                        layer.max_index_list.append(max_index)
                        self.state_s[l].append(pool_result)
                # Early stopping after every neuron spiked.
                if self.target_type == 'latency':
                    self.output_s_cum += self.state_s[-1][-1]
                    if (self.output_s_cum > 0).all():
                        break

            self.term_length = t+1

            # Reshape, permute, convert the output spike train.
            self.output = torch.stack(self.state_s[-1]).permute(1, 0, 2)

            if self.multi_model:
                self.output_each_model = self.output.reshape(self.num_models, int(self.output.shape[0]/self.num_models),
                                                            *self.output.shape[1:])

            # batch x time x feature
            self.calc_num_spike()

        return self.output

    def calc_loss(self, target, calc_spike_loss=False):

        self.batch_size = target.shape[0]
        if self.target_type == 'train':
            with torch.no_grad():
                self.diff = self.apply_alpha_kernel(self.output.float() - target.float())[:, :(self.time_length + self.alpha_extend), :]
                self.L = torch.pow(self.diff, 2) / (self.time_length * self.batch_size)
                if self.multi_model:
                    self.loss = self.L.reshape(self.num_models, -1).sum(1) * self.num_models
                else:
                    self.loss = self.L.sum()
        elif self.target_type == 'count':
            with torch.no_grad():
                self.diff = self.apply_alpha_kernel(self.output.float() - target.float())[:, :(self.time_length + self.alpha_extend), :]
                self.L = torch.pow(self.diff[:, -1, :], 2) / (self.time_length * self.batch_size)
                if self.multi_model:
                    self.loss = self.L.reshape(self.num_models, -1).sum(1) * self.num_models
                else:
                    self.loss = self.L.sum()
        elif self.target_type == 'latency':
            loss = nn.CrossEntropyLoss(reduction='none')
            self.tl_m_tf = (self.output * torch.arange(self.term_length, 0, -1).view(1, -1, 1)).max(dim=1).values # (term_length - t_first), =term_length for the first time step, =1 for the last time step, =0 for no spike
            self.sm_inp = self.tl_m_tf.float() * self.softmax_beta
            self.sm_inp.requires_grad = True
            self.celoss_per_batch = loss(self.sm_inp, target) / self.batch_size * self.num_models
            self.celoss = self.celoss_per_batch.sum()
            if self.multi_model:
                self.celoss_per_model = self.celoss_per_batch.reshape(self.num_models, -1).sum(1)
            self.celoss.backward()

            with torch.no_grad():
                self.L_nospike_per_batch = (torch.gather(self.output.sum(1), 1, target.view(-1, 1)) == 0).float() / self.batch_size * self.num_models * self.lambda_nospike
                self.loss_nospike = self.L_nospike_per_batch.sum()
                if self.multi_model:
                    self.loss_nospike_per_model = self.L_nospike_per_batch.reshape(self.num_models, -1).sum(1)

                if self.multi_model:
                    self.loss = self.celoss_per_model + self.loss_nospike_per_model
                else:
                    self.loss = self.celoss + self.loss_nospike


    def calc_first_stime(self):
        """
        Args
            self.output : [batch x time x feature]
            times : [batch]
        """
        decreasing_output = self.output * torch.arange(self.term_length, 0, -1).view(1, -1, 1)
        max_each_neuron = decreasing_output.max(dim=1).values
        # batch x feature
        max_whole_network = max_each_neuron.max(dim=1).values
        # batch
        first_stime = self.term_length - max_whole_network
        # batch
        if self.multi_model:
            first_stime_model = first_stime.reshape(self.num_models, -1)
            self.first_stime_min = first_stime_model.min(1).values.tolist()
            self.first_stime_mean = first_stime_model.mean(1).tolist()
        else:
            self.first_stime_min = first_stime.min().item()
            self.first_stime_mean = first_stime.mean().item()

        return first_stime.int()

    def calc_num_spike(self):
        """
        Add the number of total spikes except the last layer.
        Note that this value will increase with batch_size.
        """
        num_spike_total = []
        for l, layers in enumerate(self.layers):
            if self.multi_model:
                layer = layers[0]
            else:
                layer = layers
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if self.multi_model:
                    num_spike_total_each_model = torch.stack(self.state_s[l]).permute(1, 0, 2).reshape(self.num_models, int(self.output.shape[0] / self.num_models), self.term_length, -1).sum(1).sum(1).sum(1).int()
                    num_spike_total.append(num_spike_total_each_model)
                else:
                    num_spike_total.append(int(torch.stack(self.state_s[l]).sum().item()))
        if self.multi_model:
            num_spike_total = torch.stack(num_spike_total).t().tolist()
        self.num_spike_total = num_spike_total

        first_stime = self.calc_first_stime()
        num_spike_nec = []
        for l, layers in enumerate(self.layers):
            if self.multi_model:
                layer = layers[0]
            else:
                layer = layers
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                num_spike_nec_each_batch = torch.tensor([torch.stack(self.state_s[l]).int()[:(first_stime[b] + 1), b].sum().item()  for b in range(self.output.shape[0])])
                if self.multi_model:
                    num_spike_nec.append(num_spike_nec_each_batch.reshape(self.num_models, -1).sum(1))
                else:
                    num_spike_nec.append(num_spike_nec_each_batch.sum().item())
        if self.multi_model:
            num_spike_nec = torch.stack(num_spike_nec).t().tolist()
        self.num_spike_nec = num_spike_nec
        return

    def apply_alpha_kernel(self, input, padding=True, flip=True, prime=False):
        """
        Applies self.kernel or self.kernel_prime to given input.
        Arg
            input : input.shape = [batch_size, num_time_step, num_features]
            padding : Whether apply zero padding at both sides or not.
            flip : Whether apply flip to the kernel or not.
                Usually flip is used for inference and unflipped kernel is used
                for BP.
            prime : Whether to use kernel_prime or not.
        Return
            output : result.shape = [batch_size, num_time_step+alpha, num_features]
        """
        assert len(input.shape)==3, "Input shape must be rank-3"
        batch_size, num_time_step, num_features = input.shape

        kernel = self.alpha_kernel_prime if prime else self.alpha_kernel
        kernel = kernel.flip(2) if flip else kernel
        padding_length = kernel.numel()-1 if padding else 0

        input = input.float().permute(0, 2, 1)
        input = input.reshape(batch_size * num_features, 1, num_time_step)

        output = F.conv1d(input, kernel, padding=padding_length)
        output = output.reshape(batch_size, num_features, -1)
        output = output.permute(0, 2, 1)

        return output

    def clean_state(self):
        self.input = None
        self.target = None
        self.output = None
        self.output_each_model = None
        self.output_s_cum = None

        self.weight_list = None
        self.bias_list = None
        self.weight_grad = None
        self.bias_grad = None

        self.state_i = None
        self.state_v = None
        self.state_v_prime = None
        self.state_s = None

        self.state_i_grad = None
        self.state_v_grad = None
        self.state_v_dep_grad = None
        self.state_s_grad = None
        self.state_t_grad = None
        self.state_v_grad_epr_ef1 = None
        self.state_v_grad_epr_ef2 = None
        self.dLdS = None
        self.dLdT = None
        gc.collect()

    def _init_backward(self):
        self.state_i_grad = list()
        self.state_v_grad = list()
        if self.lrule != 'Timing':
            self.state_s_grad = list()
        else:
            self.state_s_grad = None
        if self.lrule != 'RNN':
            self.state_v_dep_grad = list()
        else:
            self.state_v_dep_grad = None
        if self.lrule in ['Timing', 'ANTLR']:
            self.state_t_grad = list()
            self.state_v_grad_epr_ef1 = list()
            self.state_v_grad_epr_ef2 = list()
        else:
            self.state_t_grad = None
            self.state_v_grad_epr_ef1 = None
            self.state_v_grad_epr_ef2 = None

        for l in range(self.num_layer):
            # state_*_grad[l]: time x batch x neuron
            zeros = torch.zeros(self.term_length, self.batch_size,
                                *self.fmap_shape_list[l], requires_grad=False)
            if self.state_s_grad is not None:
                self.state_s_grad.append(zeros.clone())
            if self.state_t_grad is not None:
                self.state_t_grad.append(zeros.clone())

            if self.fmap_type_list[l] in ["conv", "fc"]:
                self.state_i_grad.append(zeros.clone())
                self.state_v_grad.append(zeros.clone())
                if self.state_v_dep_grad is not None:
                    self.state_v_dep_grad.append(zeros.clone())
                if self.state_v_grad_epr_ef1 is not None:
                    self.state_v_grad_epr_ef1.append(zeros.clone())
                if self.state_v_grad_epr_ef2 is not None:
                    self.state_v_grad_epr_ef2.append(zeros.clone())
            else:
                self.state_i_grad.append(None)
                self.state_v_grad.append(None)
                if self.state_v_dep_grad is not None:
                    self.state_v_dep_grad.append(None)
                if self.state_v_grad_epr_ef1 is not None:
                    self.state_v_grad_epr_ef1.append(None)
                if self.state_v_grad_epr_ef2 is not None:
                    self.state_v_grad_epr_ef2.append(None)

    def backward_custom(self, target, epoch=0):
        """
        Calculate gradient of each parameters.
        Args
            output : output value.
                output.shape = [batch_size, num_time_step, num_features]
            target : target value.
                for train and count target:
                    target.shape = [batch_size, num_time_step, num_features]
                for latency target:
                    target.shape = [batch_size]
        """
        # Initialize backward variables when batch_size is changed.
        if self.multi_model:
            if self.target_type == 'latency':
                assert target.dim() == 2
                target = target.reshape(target.shape[0] * target.shape[1])
            else:
                assert target.dim() == 4
                target = target.reshape(target.shape[0] * target.shape[1], *target.shape[2:])

        self.batch_size = target.shape[0]
        assert self.input.shape[0] == self.batch_size

        self._init_param_grads()
        self._init_backward()

        # batch_size, num_time_step, num_features = self.output.shape
        if self.target_type == 'train' or self.target_type == 'count':
            assert self.output.shape == target.shape, "Output and target should be in the same shape"
            with torch.no_grad():
                # batch_size, num_time_step, num_features = self.output.shape
                self.calc_loss(target)

                ### Getting dLdS.
                if self.target_type == 'train':
                    dLdS = (self.apply_alpha_kernel(self.diff, padding=False, flip=False)
                            * 2 / (self.time_length * self.batch_size))
                elif self.target_type == 'count':
                    dLdS = self.diff[:, -1, :].view(self.batch_size, 1, -1).repeat(1, self.time_length, 1) * 2 / (self.time_length * self.batch_size)     # since alpha_exp = 1

                if self.multi_model:
                    dLdS *= self.num_models

                # Reshape to [time, batch, neuron]
                self.dLdS = dLdS.permute(1, 0, 2)

                ### Getting dLdT.
                if self.target_type == 'train':
                    dLdT_raw = self.apply_alpha_kernel(self.diff, flip=False, prime=True)
                    dLdT_raw = (dLdT_raw[:, (self.alpha_kernel_prime.numel()-2):(self.time_length + self.alpha_kernel_prime.numel()-2), :]
                                * (-2) / (self.time_length * self.batch_size))
                    # time x batch x neuron
                elif self.target_type == 'count':
                    dLdT_raw = torch.zeros(self.output.shape)

                if self.multi_model:
                    dLdT_raw *= self.num_models

                dLdT = self.output * dLdT_raw
                # Reshape to [time, batch, neuron]
                self.dLdT_raw = dLdT_raw.permute(1, 0, 2)
                self.dLdT = dLdT.permute(1, 0, 2)

        elif self.target_type == 'latency':
            assert self.output.shape[0] == target.shape[0], "Output and target should be in appropriate shape"
            self.calc_loss(target)
            with torch.no_grad():
                self.dLdS = torch.zeros(self.batch_size, *self.fmap_shape_list[-1]).scatter_(1, target.view(-1, 1), -self.L_nospike_per_batch).repeat(self.term_length, 1, 1)
                # time x batch x neuron

                dLdT = torch.zeros(self.term_length, self.batch_size, *self.fmap_shape_list[-1])
                self.sm_grad = self.sm_inp.grad
                self.sm_grad[self.tl_m_tf == 0] = 0
                idxs = torch.clamp(self.term_length - self.tl_m_tf, 0, self.term_length-1).long()
                self.dLdT = dLdT.scatter_(0, idxs.unsqueeze(0), self.sm_grad.unsqueeze(0) * (-self.softmax_beta))
                # time x batch x neuron

        with torch.no_grad():
            if self.lrule == 'Activation':
                self.gradAdd(self.dLdS, self.lrule)
            elif self.lrule == 'Timing':
                self.gradAdd(self.dLdT, 'Timing')
            elif self.lrule == 'ANTLR':
                self.lambda_timing = 1
                self.lambda_act = 1
                self.gradAdd((self.dLdT, self.dLdS), 'ANTLR')
            else:
                raise ValueError("Wrong lrule name.")

            for l, layer in enumerate(self.layers):
                if self.multi_model:
                    for m in range(self.num_models):
                        if type(self.grad_clip) == list:
                            assert self.num_models % len(self.grad_clip) == 0, f"{self.grad_clip}"
                            grad_clip = self.grad_clip[m // (self.num_models // len(self.grad_clip))]
                        else:
                            grad_clip = self.grad_clip

                        if isinstance(layer[m], nn.Conv2d) or isinstance(layer[m], nn.Linear):
                            layer[m].weight.grad = torch.clamp(layer[m].weight.grad, -abs(grad_clip), abs(grad_clip))
                            self.state_v_bs[l][m].grad = torch.clamp(self.state_v_bs[l][m].grad, -abs(grad_clip), abs(grad_clip))
                else:
                    if type(self.grad_clip) == list:
                        grad_clip = self.grad_clip[0]
                    else:
                        grad_clip = self.grad_clip

                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        layer.weight.grad = torch.clamp(layer.weight.grad, -abs(grad_clip), abs(grad_clip))
                        self.state_v_bs[l].grad = torch.clamp(self.state_v_bs[l].grad, -abs(grad_clip), abs(grad_clip))

    def gradAdd(self, output_grad_extrn, lrule, scale=1.0):
        # output_grad_extrn: time x batch x neuron
        with torch.no_grad():

            if lrule == 'Activation':
                self.bpAct(output_grad_extrn, 'SRM')
            elif lrule == 'Timing':
                self.bpTiming_recurrent(output_grad_extrn)
            elif lrule == 'ANTLR':
                self.bpANTLR(output_grad_extrn)
            for l, layers in enumerate(self.layers):
                if self.multi_model:
                    for m, layer in enumerate(layers):
                        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                            if type(layer.weight.grad) == type(None):
                                layer.weight.grad = self.weight_grad[l][m] * scale
                            else:
                                layer.weight.grad += self.weight_grad[l][m] * scale

                            if hasattr(self, 'bias_grad'):
                                if type(self.state_v_bs[l][m].grad) == type(None):
                                    self.state_v_bs[l][m].grad = self.bias_grad[l][m] * scale
                                else:
                                    self.state_v_bs[l][m].grad += self.bias_grad[l][m] * scale
                else:
                    layer = layers
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        if type(layer.weight.grad) ==  type(None):
                            layer.weight.grad = self.weight_grad[l] * scale
                        else:
                            layer.weight.grad += self.weight_grad[l] * scale

                        if hasattr(self, 'bias_grad'):
                            if type(self.state_v_bs[l].grad) == type(None):
                                self.state_v_bs[l].grad = self.bias_grad[l] * scale
                            else:
                                self.state_v_bs[l].grad += self.bias_grad[l] * scale

    def bpAct(self, output_grad_extrn, lrule):
        # output_grad_extrn: time x batch x neuron
        with torch.no_grad():
            for t in range(self.term_length-1, -1, -1):
                for l in range(self.num_layer-1, -1, -1):
                    ###### dL/dS[t] from upper layer
                    if l == self.num_layer-1:
                        self.state_s_grad[l][t] = output_grad_extrn[t]
                    else:
                        if self.fmap_type_list[l+1] in ["fc", "conv"]:
                            self.prop_dLdI_to_dLdX(t, l, X='S')
                        # For pooling & flatten.
                        else:
                            self.prop_dLdX_to_dLdX(t, l, X='S')


                    if self.fmap_type_list[l] in ["fc", "conv"]:
                        ###### dL/dV[t] from dL/dS[t]
                        self.state_v_grad[l][t] = self.surr_deriv(self.state_v[l][t]) * self.state_s_grad[l][t]
                        ###### dL/dI[t] from dL/dV[t]
                        self.prop_dLdV_to_dLdI(lrule, t, l)
            self.prop_dLdI_to_dLdW(lrule)

    def bpTiming_recurrent(self, output_grad_extrn):
        # output_grad_extrn: time x batch x neuron
        with torch.no_grad():
            for t in range(self.term_length-1, -1, -1):
                for l in range(self.num_layer-1, -1, -1):
                    ###### dL/dT[t] from upper layer
                    if l == self.num_layer-1:
                        self.state_t_grad[l][t] = output_grad_extrn[t]
                    else:
                        ###### dL/dT[t] from upper layer dL/dV[t]
                        if self.fmap_type_list[l+1] in ["fc", "conv"]:
                            self.tprop_dLdV_to_dLdT(t, l)
                        else:
                            self.prop_dLdX_to_dLdX(t, l, X='T')

                    if self.fmap_type_list[l] in ["fc", "conv"]:
                        ###### dL/dV[t] from dL/dT[t]
                        effective_input = (self.state_s[l][t] == 1)
                        self.state_v_grad[l][t] = 0
                        self.state_v_grad[l][t][effective_input] = self.state_t_grad[l][t][effective_input] / -self.state_v_prime[l][t][effective_input]

                        ###### dL/dI[t] from dL/dV[t]
                        self.prop_dLdV_to_dLdI('SRM', t, l)
            self.prop_dLdI_to_dLdW('SRM', True)

    def bpANTLR(self, output_grad_extrn): # timing + SRM (not RNN)
        # output_grad_extrn: 2 (dLdT, dLdS) x time x batch x neuron
        with torch.no_grad():
            for t in range(self.term_length-1, -1, -1):
                for l in range(self.num_layer-1, -1, -1):
                    ###### dL/dT[t], dL/dS[t] from upper layer
                    if l == self.num_layer-1:
                        self.state_t_grad[l][t] = output_grad_extrn[0][t]
                        self.state_s_grad[l][t] = output_grad_extrn[1][t]
                    else:
                        if self.fmap_type_list[l+1] in ["fc", "conv"]:
                            ###### dL/dT[t] from upper layer dL/dV[t]
                            self.tprop_dLdV_to_dLdT(t, l)
                            ###### dL/dS[t] from upper layer dL/dV[t]
                            self.prop_dLdI_to_dLdX(t, l, X='S')
                        else:
                            # For pooling & flatten
                            self.prop_dLdX_to_dLdX(t, l, X='T')
                            self.prop_dLdX_to_dLdX(t, l, X='S')

                    if self.fmap_type_list[l] in ["fc", "conv"]:
                        ###### dL/dV[t] from dL/dT[t], dLdS[t]
                        effective_input = (self.state_s[l][t] == 1)

                        act_vgrad = self.surr_deriv(self.state_v[l][t]) * self.state_s_grad[l][t]
                        tim_vgrad = torch.zeros(self.state_t_grad[l][t].shape)
                        tim_vgrad[effective_input] = self.state_t_grad[l][t][effective_input] / -self.state_v_prime[l][t][effective_input]
                        self.state_v_grad[l][t] = self.lambda_act * act_vgrad
                        self.state_v_grad[l][t][effective_input] += self.lambda_timing * tim_vgrad[effective_input]

                        ###### dL/dI[t] from dL/dV[t]
                        self.prop_dLdV_to_dLdI('SRM', t, l)
            self.prop_dLdI_to_dLdW('SRM')
            ###### timing weight not used

    def prop_dLdI_to_dLdX(self, time, layer, X):
        t = time; l = layer

        if X == 'S':
            state_i_grad = self.state_i_grad
            state_x_grad = self.state_s_grad
        elif X == 'T':
            state_i_grad = self.state_v_grad_epr_ef2
            state_x_grad = self.state_t_grad
        else:
            raise ValueError("Invalid value for X.")

        with torch.no_grad():
            if self.fmap_type_list[l+1] == "fc":
                if self.multi_model:
                    state_i_grad_rs = state_i_grad[l+1][t].reshape(self.num_models, int(self.batch_size / self.num_models), -1)
                    x_grad_per_beta_i = torch.bmm(state_i_grad_rs, self.weight_list[l+1]).reshape(self.batch_size, -1)
                else:
                    x_grad_per_beta_i = torch.mm(state_i_grad[l+1][t], self.layers[l+1].weight)
                state_x_grad[l][t] = self.beta_i * x_grad_per_beta_i
            elif self.fmap_type_list[l+1] == "conv":
                assert not self.multi_model
                padding = self.layers[l+1].padding
                x_grad_per_beta_i = torch.nn.grad.conv2d_input(state_x_grad[l][t].shape,
                                                               self.layers[l+1].weight,
                                                               state_i_grad[l+1][t],
                                                               padding=padding)
                state_x_grad[l][t] = self.beta_i * x_grad_per_beta_i

            # t_grad of spiked timestep should only be nonzero value.
            if X == 'T':
                state_x_grad[l][t] *= self.state_s[l][t]

        return

    def tprop_dLdV_to_dLdT(self, time, layer):
        t = time; l = layer
        with torch.no_grad():
            if time != self.term_length-1:
                self.state_v_grad_epr_ef1[l+1][t] =  - (- self.state_v_grad[l+1][t+1] / 2) * (1 - self.state_s[l+1][t]) # for dLdV[x+t] * -eps[t-1]
                self.state_v_grad_epr_ef1[l+1][t] += self.alpha_v * (1 - self.state_s[l+1][t]) * self.state_v_grad_epr_ef1[l+1][t+1]
                self.state_v_grad_epr_ef2[l+1][t] = self.alpha_i * (1 - self.state_s[l+1][t]) * self.state_v_grad_epr_ef2[l+1][t+1]
            else:
                self.state_v_grad_epr_ef1[l+1][t] = torch.zeros(self.batch_size, *self.fmap_shape_list[l+1])
                self.state_v_grad_epr_ef2[l+1][t] = torch.zeros(self.batch_size, *self.fmap_shape_list[l+1])

            # Changed the sign.
            self.state_v_grad_epr_ef1[l+1][t] += self.alpha_v * (- self.state_v_grad[l+1][t] / 2)                       # for dLdV[x+t] * eps[t+1]
            self.state_v_grad_epr_ef2[l+1][t] += self.beta_v * self.alpha_i * (- self.state_v_grad[l+1][t] / 2)         # for dLdV[x+t] * eps[t+1]
            self.state_v_grad_epr_ef2[l+1][t] += self.beta_v * self.state_v_grad_epr_ef1[l+1][t]

            self.prop_dLdI_to_dLdX(t, l, X='T')

        return

    def prop_dLdX_to_dLdX(self, time, layer, X):
        t = time; l = layer

        if X == 'S':
            state_x_grad = self.state_s_grad
        elif X == 'T':
            state_x_grad = self.state_t_grad
        else:
            raise ValueError("Invalid value for X.")

        with torch.no_grad():
            if "pool" in self.fmap_type_list[l+1]:
                kernel_size = self.layers[l+1].kernel_size
                if self.fmap_type_list[l+1] == "apool":
                    x_grad = F.interpolate(state_x_grad[l+1][t], scale_factor=kernel_size)
                    x_grad /= kernel_size * kernel_size
                elif self.fmap_type_list[l+1] == "mpool":
                    x_grad = F.max_unpool2d(state_x_grad[l+1][t],
                                            self.layers[l+1].max_index_list[t],
                                            kernel_size=kernel_size,
                                            output_size=self.fmap_shape_list[l][1:])
                if list(x_grad.shape[1:]) != self.fmap_shape_list[l]:
                    to_pad = self.fmap_shape_list[l][-1] - x_grad.shape[-1]
                    x_grad = F.pad(x_grad, (0, to_pad, 0, to_pad), "constant", 0)
                    assert list(x_grad.shape[1:]) == self.fmap_shape_list[l]
                state_x_grad[l][t] = x_grad
            elif self.fmap_type_list[l+1] == "flatten":
                state_x_grad[l][t] = state_x_grad[l+1][t].view(state_x_grad[l][t].shape)

        return

    def prop_dLdV_to_dLdI(self, style, time, layer):
        with torch.no_grad():
            t = time
            l = layer

            if style == 'RNN':
                ###### dL/dV[t] from next time step (dL/dV[t+1])
                if t != self.term_length-1:
                    self.state_v_grad[l][t] += self.alpha_v * (1 - self.state_s[l][t]) * self.state_v_grad[l][t+1]

                ###### dL/dI[t] from dL/dV[t]
                self.state_i_grad[l][t] = self.beta_v * self.state_v_grad[l][t]

                ###### dL/dI[t] from next time step (dL/dI[t+1])
                if t != self.term_length-1:
                    self.state_i_grad[l][t] += self.alpha_i * (1 - self.state_s[l][t]) * self.state_i_grad[l][t+1]

            elif style == 'SRM':
                ###### dL/dV[t] from next time step (dL/dV[t+1])
                self.state_v_dep_grad[l][t] = self.state_v_grad[l][t]
                if t != self.term_length-1:
                    self.state_v_dep_grad[l][t] += self.alpha_v * (1 - self.state_s[l][t]) * self.state_v_dep_grad[l][t+1]

                ###### dL/dI[t] from dL/dV[t]
                self.state_i_grad[l][t] = self.beta_v * self.state_v_dep_grad[l][t]

                ###### dL/dI[t] from next time step (dL/dI[t+1])
                if t != self.term_length-1:
                    self.state_i_grad[l][t] += self.alpha_i * (1 - self.state_s[l][t]) * self.state_i_grad[l][t+1]

            elif style == 'SLAYER':
                ###### dL/dV[t] from next time step (dL/dV[t+1])
                self.state_v_dep_grad[l][t] = self.state_v_grad[l][t]
                if t != self.term_length-1:
                    self.state_v_dep_grad[l][t] += self.alpha_v * self.state_v_dep_grad[l][t+1]

                ###### dL/dI[t] from dL/dV[t]
                self.state_i_grad[l][t] = self.beta_v * self.state_v_dep_grad[l][t]

                ###### dL/dI[t] from next time step (dL/dI[t+1])
                if t != self.term_length-1:
                    self.state_i_grad[l][t] += self.alpha_i * self.state_i_grad[l][t+1]

            else:
                raise ValueError("Invalid style name.")

    def prop_dLdI_to_dLdW(self, style, is_timing=False):
        with torch.no_grad():
            for l in range(self.num_layer):
                if self.fmap_type_list[l] in ["fc", "conv"]:
                    # time_length x batch x f_shape
                    if l == 0:
                        hidden_s_all = self.input.transpose(0,1)[:self.term_length]
                    else:
                        hidden_s_all = torch.stack(self.state_s[l-1])

                    if style == 'RNN':
                        v_dep_grad = self.state_v_grad[l]
                    elif style == 'SRM' or style == 'SLAYER':
                        v_dep_grad = self.state_v_dep_grad[l]
                    else:
                        raise ValueError("Something's wrong.")

                    ### Calc weight grad for fc layer.
                    if self.fmap_type_list[l] == "fc":
                        if self.multi_model:
                            hidden_t_m_b_n = hidden_s_all.reshape(self.term_length, self.num_models, int(self.batch_size / self.num_models), -1)
                            hidden_m_t_b_n = hidden_t_m_b_n.permute(1, 0, 2, 3)
                            hidden_m_tb_n = hidden_m_t_b_n.reshape(self.num_models, self.term_length * int(self.batch_size / self.num_models), -1)

                            igrad_t_m_b_n = self.state_i_grad[l].reshape(self.term_length, self.num_models, int(self.batch_size / self.num_models), -1)
                            igrad_m_t_b_n = igrad_t_m_b_n.permute(1, 0, 2, 3)
                            igrad_m_tb_n = igrad_m_t_b_n.reshape(self.num_models, self.term_length * int(self.batch_size / self.num_models), -1)
                            igrad_m_n_tb = igrad_m_tb_n.permute(0, 2, 1)

                            self.weight_grad[l] = torch.bmm(igrad_m_n_tb, hidden_m_tb_n) * self.beta_i
                        else:
                            self.weight_grad[l] = torch.mm(self.state_i_grad[l].reshape(-1, self.state_i_grad[l].shape[-1]).t(),
                                                            hidden_s_all.reshape(-1, hidden_s_all.shape[-1])) * self.beta_i

                        if is_timing:
                            if self.multi_model:
                                fan_in = self.weight_grad[l].shape[2]
                                timing_penalty_coeff = self.timing_penalty / fan_in
                                no_spike = 1 - (torch.stack(self.state_s[l]).sum(dim=0) > 0).float()
                                # model_batch x fan_out
                                no_spike = no_spike.reshape(self.num_models, int(self.batch_size / self.num_models), -1)
                                # model x batch x fan_out
                                no_spike_dw = timing_penalty_coeff * no_spike.mean(dim=1).reshape(self.num_models, -1, 1)
                                # model x fan_out x 1
                                self.weight_grad[l] -= no_spike_dw
                            else:
                                fan_in = self.weight_grad[l].shape[1]
                                timing_penalty_coeff = self.timing_penalty / fan_in
                                no_spike = 1 - (torch.stack(self.state_s[l]).sum(dim=0) > 0).float()
                                # batch x fan_out
                                no_spike_dw = timing_penalty_coeff * no_spike.mean(dim=0).reshape(-1, 1)
                                # fan_out x 1
                                self.weight_grad[l] -= no_spike_dw
                        if torch.isnan(self.weight_grad[l]).any():
                            self.weight_grad[l][torch.isnan(self.weight_grad[l])] = 0
                            print(f'nan appeared and replaced by 0')

                        if self.multi_model:
                            vgrad_t_m_b_n = v_dep_grad.reshape(self.term_length, self.num_models, int(self.batch_size / self.num_models), -1)
                            vgrad_m_n = vgrad_t_m_b_n.sum(0).sum(1) * self.beta_bias
                            self.bias_grad[l] = vgrad_m_n
                        else:
                            self.bias_grad[l] = v_dep_grad.sum(0).sum(0) * self.beta_bias


                    ### Calc weight grad for conv layer.
                    elif self.fmap_type_list[l] == "conv":
                        assert not self.multi_model
                        weight_shape = self.layers[l].weight.shape
                        hidden_s_all_rs = hidden_s_all.reshape(-1,*list(hidden_s_all.shape[2:]))
                        i_grad_rs = self.state_i_grad[l].reshape(-1,*list(self.state_i_grad[l].shape[2:]))
                        padding = self.layers[l].padding
                        self.weight_grad[l] = torch.nn.grad.conv2d_weight(
                            hidden_s_all_rs, self.layers[l].weight.shape, i_grad_rs,
                            padding=padding)
                        if torch.isnan(self.weight_grad[l]).any():
                            import pdb; pdb.set_trace()
                            self.weight_grad[l][torch.isnan(self.weight_grad[l])] = 0
                            print(f'nan appeared and replaced by 0')

                        self.bias_grad[l] = v_dep_grad.sum(dim=[0,1,3,4]) * self.beta_bias

    def act(self, input):
        return (input >= 1).float()

    def surr_deriv(self, input_v):
        with torch.no_grad():
            output = (-self.surr_beta * (input_v-1.0).abs()).exp() * self.surr_alpha
        return output
