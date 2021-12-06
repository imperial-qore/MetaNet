import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math, random
import numpy as np

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        
class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

class NPNLinear(nn.Module):
    def positive_s(self, x, use_sigmoid = 0):
        if use_sigmoid == 0:
            y = torch.log(torch.exp(x) + 1)
        else:
            y = F.sigmoid(x)
        return y

    def positive_s_inv(self, x, use_sigmoid = 0):
        if use_sigmoid == 0:
            y = torch.log(torch.exp(x) - 1)
        else:
            y = - torch.log(1 / x - 1)
        return y

    def __init__(self, in_channels, out_channels, dual_input = True, init_type = 0):
        # init_type 0: normal, 1: mixture of delta distr'
        super(NPNLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dual_input = dual_input

        self.W_m = nn.Parameter(2 * math.sqrt(6) / math.sqrt(in_channels + out_channels) * (torch.rand(in_channels, out_channels) - 0.5))
        if init_type == 0:
            W_s_init = 0.01 * math.sqrt(6) / math.sqrt(in_channels + out_channels) * torch.rand(in_channels, out_channels)
        else:
            bern = torch.bernoulli(torch.ones(in_channels, out_channels) * 0.5)
            W_s_init = bern * math.exp(-2) + (1 - bern) * math.exp(-14)
            print(W_s_init[:4,:4])
        self.W_s_ = nn.Parameter(self.positive_s_inv(W_s_init, 0))

        self.bias_m = nn.Parameter(torch.zeros(out_channels))
        if init_type == 0:
            self.bias_s_ = nn.Parameter(torch.ones(out_channels) * (-10))
        else:
            bern = torch.bernoulli(torch.ones(out_channels) * 0.5)
            bias_s_init = bern * math.exp(-2) + (1 - bern) * math.exp(-14)
            self.bias_s_ = nn.Parameter(self.positive_s_inv(bias_s_init, 0))

    def forward(self, x):
        if self.dual_input:
            x_m, x_s = x
        else:
            x_m = x
            x_s = x.clone()
            x_s = 0 * x_s

        o_m = torch.mm(x_m, self.W_m)
        o_m = o_m + self.bias_m.expand_as(o_m)

        #W_s = torch.log(torch.exp(self.W_s_) + 1)
        #bias_s = torch.log(torch.exp(self.bias_s_) + 1)
        W_s = self.positive_s(self.W_s_, 0)
        bias_s = self.positive_s(self.bias_s_, 0)

        o_s = torch.mm(x_s, W_s) + torch.mm(x_s, self.W_m * self.W_m) + torch.mm(x_m * x_m, W_s)
        o_s = o_s + bias_s.expand_as(o_s)

        #print('bingo om os')
        #print(o_m.data)
        #print(o_s.data)

        return o_m, o_s

class NPNRelu(nn.Module):
    def __init__(self):
        super(NPNRelu, self).__init__()
        self.scale = math.sqrt(8/math.pi) 

    def forward(self, x):
        assert(len(x) == 2)
        o_m, o_s = x
        a_m = F.sigmoid(self.scale * o_m * (o_s ** (-0.5))) * o_m + torch.sqrt(o_s) / math.sqrt(2 * math.pi) * torch.exp(-o_m ** 2 / (2 * o_s))
        a_s = F.sigmoid(self.scale * o_m * (o_s ** (-0.5))) * (o_m ** 2 + o_s) + o_m * torch.sqrt(o_s) / math.sqrt(2 * math.pi) * torch.exp(-o_m ** 2 / (2 * o_s)) - a_m ** 2  # mbr
        return a_m, a_s

class NPNSigmoid(nn.Module):
    def __init__(self):
        super(NPNSigmoid, self).__init__()
        self.xi_sq = math.pi / 8
        self.alpha = 4 - 2 * math.sqrt(2)
        self.beta = - math.log(math.sqrt(2) + 1)

    def forward(self, x):
        assert(len(x) == 2)
        o_m, o_s = x
        a_m = F.sigmoid(o_m / (1 + self.xi_sq * o_s) ** 0.5)
        a_s = F.sigmoid(self.alpha * (o_m + self.beta) / (1 + self.xi_sq * self.alpha ** 2 * o_s) ** 0.5) - a_m ** 2
        return a_m, a_s

class NPNDropout(nn.Module):
    def __init__(self, rate):
        super(NPNDropout, self).__init__()
        self.dropout = nn.Dropout2d(p = rate)
    def forward(self, x):
        assert(len(x) == 2)
        if self.training:
            self.dropout.train()
        else:
            self.dropout.eval()
        x_m, x_s = x
        x_m = x_m.unsqueeze(2)
        x_s = x_s.unsqueeze(2)
        x_com = torch.cat((x_m, x_s), dim = 2)
        x_com = x_com.unsqueeze(3)
        x_com = self.dropout(x_com)
        y_m = x_com[:,:,0,0]
        y_s = x_com[:,:,1,0]
        return y_m, y_s

def NPNBCELoss(pred_m, pred_s, label):
    loss = -torch.sum((torch.log(pred_m + 1e-10) * label + torch.log(1 - pred_m + 1e-10) * (1 - label))/ (pred_s + 1e-10) - torch.log(pred_s+ 1e-10))
    return loss

def KL_BG(pred_m, pred_s, label):
    loss = 0.5 * torch.sum((1 - label) * (pred_m ** 2 / pred_s + torch.log(torch.clamp(math.pi * 2 * pred_s, min=1e-6))) + label * ((pred_m - 1) ** 2 / pred_s + torch.log(torch.clamp(math.pi * 2 * pred_s, min=1e-6)))) / pred_m.size()[0] # min = 1e-6
    return loss

def L2_loss(pred, label):
    loss = torch.sum((pred - label) ** 2)
    return loss

def KL_loss(pred, label):
    assert(len(pred) == 2)
    pred_m, pred_s = pred
    # print(((pred_m - label) ** 2) / (pred_s) , torch.log(pred_s))
    loss = 0.5 * torch.sum(10 * ((pred_m - label) ** 2) / (pred_s) + torch.log(pred_s)) # may need epsilon
    return loss

def multi_logistic_loss(pred, label):
    assert(len(label.size()) == 1)
    print('bingo type\n', label.data.type())
    print('bingo label\n', pred[:, label])
    log_prob = torch.sum(torch.log(1 - pred)) + torch.sum(log(pred[:, label.data]) - log(1 - pred[:, label.data]))
    return -log_prob

def RMSE(pred, label):
    loss = torch.mean(torch.sum((pred - label) ** 2, 1), 0) ** 0.5
    return loss
