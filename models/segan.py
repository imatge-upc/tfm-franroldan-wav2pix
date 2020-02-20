import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _single
import json
from torch.autograd import Variable
import numpy as np
from utils import Concat_embed
import pdb
from models.spectral_norm import SpectralNorm
import os


class Saver(object):
    def __init__(self, model, save_path, max_ckpts=5, optimizer=None):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, 'checkpoints')
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer

    def save(self, model_name, step, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as ckpt_f:
                # read latest checkpoints
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {'latest': [], 'current': []}

        model_path = '{}-{}.ckpt'.format(model_name, step)
        if best_val:
            model_path = 'best_' + model_path

        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    try:
                        print('Removing old ckpt {}'.format(os.path.join(save_path,
                                                                         'weights_' + todel)))
                        os.remove(os.path.join(save_path, 'weights_' + todel))
                        latest = latest[1:]
                    except FileNotFoundError:
                        print('ERROR: ckpt is not there?')

        latest += [model_path]

        ckpts['latest'] = latest
        ckpts['current'] = model_path

        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))

        st_dict = {'step': step,
                   'state_dict': self.model.state_dict()}

        if self.optimizer is not None:
            st_dict['optimizer'] = self.optimizer.state_dict()
        # now actually save the model and its weights
        # torch.save(self.model, os.path.join(save_path, model_path))
        torch.save(st_dict, os.path.join(save_path,
                                         'weights_' + \
                                         model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print('[!] No checkpoint found in {}'.format(self.save_path))
            return False
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current']
            return curr_ckpt

    # def load(self):
    #    save_path = self.save_path
    #    ckpt_path = self.ckpt_path
    #    print('Reading latest checkpoint from {}...'.format(ckpt_path))
    #    if not os.path.exists(ckpt_path):
    #        raise FileNotFoundError('[!] Could not load model. Ckpt '
    #                                '{} does not exist!'.format(ckpt_path))
    #    with open(ckpt_path, 'r') as ckpt_f:
    #        ckpts = json.load(ckpt_f)
    #    curr_ckpt = ckpts['curent']
    #    st_dict = torch.load(os.path.join(save_path, curr_ckpt))
    #    return

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is False:
            if not os.path.exists(ckpt_path):
                print('[!] No weights to be loaded')
                return False
        else:
            st_dict = torch.load(os.path.join(save_path,
                                              'weights_' + \
                                              curr_ckpt))
            #st_dict = torch.load(os.path.join(save_path,
            #                                  'weights_' + \
            #                                  curr_ckpt), map_location="cpu")
            if 'state_dict' in st_dict:
                # new saving mode
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                # legacy mode, only model was saved
                self.model.load_state_dict(st_dict)
            print('[*] Loaded weights')
            return True

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True):
        model_dict = self.model.state_dict()
        st_dict = torch.load(ckpt_file,
                             map_location=lambda storage, loc: storage)
        #st_dict = torch.load(ckpt_file,
        #                     map_location=lambda storage, loc: "cpu")
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            # legacy mode
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            # Get rid of last layer params (fc output in D)
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        # Filter unnecessary keys from loaded ones and those not existing
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and \
                   k in allowed_keys and v.size() == model_dict[k].size()}
        print('Current Model keys: ', len(list(model_dict.keys())))
        print('Loading Pt Model keys: ', len(list(pt_dict.keys())))
        print('Loading matching keys: ', list(pt_dict.keys()))
        if len(pt_dict.keys()) != len(model_dict.keys()):
            print('WARNING: LOADING DIFFERENT NUM OF KEYS')
        # overwrite entries in existing dict
        model_dict.update(pt_dict)
        # load the new state dict
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print('WARNING: {} weights not loaded from pt ckpt'.format(k))
        if self.optimizer is not None and 'optimizer' in st_dict and load_opt:
            self.optimizer.load_state_dict(st_dict['optimizer'])

class Conv1DResBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth=3,
                 dilations=[1, 2, 4, 8], stride=4, bias=True,
                 transpose=False, act='prelu'):
        super(Conv1DResBlock, self).__init__()
        self.ninputs = ninputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.dilations = dilations
        self.stride = stride
        self.bias = bias
        self.transpose = transpose
        assert dilations[0] == 1, dilations[0]
        assert len(dilations) > 1, len(dilations)
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        prev_in = ninputs
        for n, d in enumerate(dilations):
            if n == 0:
                curr_stride = stride
            else:
                curr_stride = 1
            if n == 0 or (n + 1) >= len(dilations):
                # in the interfaces in/out it is different
                curr_fmaps = fmaps
            else:
                curr_fmaps = fmaps // 4
                assert curr_fmaps > 0, curr_fmaps
            if n == 0 and transpose:
                p_ = (self.kwidth - 4)//2
                op_ = 0
                if p_ < 0:
                    op_ = p_ * -1
                    p_ = 0
                self.convs.append(nn.ConvTranspose1d(prev_in, curr_fmaps, kwidth,
                                                     stride=curr_stride,
                                                     dilation=d,
                                                     padding=p_,
                                                     output_padding=op_,
                                                     bias=bias))
            else:
                self.convs.append(nn.Conv1d(prev_in, curr_fmaps, kwidth,
                                            stride=curr_stride,
                                            dilation=d,
                                            padding=0,
                                            bias=bias))
            self.acts.append(nn.PReLU(curr_fmaps))
            prev_in = curr_fmaps

    def forward(self, x):
        h = x
        res_act = None
        for li, layer in enumerate(self.convs):
            if self.stride > 1 and li == 0:
                # add proper padding
                pad_tuple = ((self.kwidth//2)-1, self.kwidth//2)
            else:
                # symmetric padding
                p_ = ((self.kwidth - 1) * self.dilations[li]) // 2
                pad_tuple = (p_, p_)
            #print('Applying pad tupple: ', pad_tuple)
            if not (self.transpose and li == 0):
                h = F.pad(h, pad_tuple)
            #print('Layer {}'.format(li))
            #print('h padded: ', h.size())
            h = layer(h)
            h = self.acts[li](h)
            if li == 0:
                # keep the residual activation
                res_act = h
            #print('h min: ', h.min())
            #print('h max: ', h.max())
            #print('h conved size: ', h.size())
        # add the residual activation in the output of the module
        return h + res_act


class GBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth,
                 activation, padding=None,
                 lnorm=False, dropout=0.,
                 pooling=2, enc=True, bias=False,
                 aal_h=None, linterp=False, snorm=False,
                 convblock=False):
        # linterp: do linear interpolation instead of simple conv transpose
        # snorm: spectral norm
        super(GBlock, self).__init__()
        self.pooling = pooling
        self.linterp = linterp
        self.enc = enc
        self.kwidth = kwidth
        self.convblock= convblock
        if padding is None:
            padding = 0
        if enc:
            if aal_h is not None:
                self.aal_conv = nn.Conv1d(ninputs, ninputs,
                                          aal_h.shape[0],
                                          stride=1,
                                          padding=aal_h.shape[0] // 2 - 1,
                                          bias=False)
                if snorm:
                    self.aal_conv = SpectralNorm(self.aal_conv)
                # apply AAL weights, reshaping impulse response to match
                # in channels and out channels
                aal_t = torch.FloatTensor(aal_h).view(1, 1, -1)
                aal_t = aal_t.repeat(ninputs, ninputs, 1)
                self.aal_conv.weight.data = aal_t
            if convblock:
                self.conv = Conv1DResBlock(ninputs, fmaps, kwidth,
                                           stride=pooling, bias=bias)
            else:
                self.conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                      stride=pooling,
                                      padding=padding,
                                      bias=bias)
            if snorm:
                self.conv = SpectralNorm(self.conv)
            if activation == 'glu':
                # TODO: REVIEW
                raise NotImplementedError
                self.glu_conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                          stride=pooling,
                                          padding=padding,
                                          bias=bias)
                if snorm:
                    self.glu_conv = spectral_norm(self.glu_conv)
        else:
            if linterp:
                # pre-conv prior to upsampling
                self.pre_conv = nn.Conv1d(ninputs, ninputs // 8,
                                          kwidth, stride=1, padding=kwidth//2,
                                          bias=bias)
                self.conv = nn.Conv1d(ninputs // 8, fmaps, kwidth,
                                      stride=1, padding=kwidth//2,
                                      bias=bias)
                if snorm:
                    self.conv = SpectralNorm(self.conv)
                if activation == 'glu':
                    self.glu_conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                              stride=1, padding=kwidth//2,
                                              bias=bias)
                    if snorm:
                        self.glu_conv = SpectralNorm(self.glu_conv)
            else:
                if convblock:
                    self.conv = Conv1DResBlock(ninputs, fmaps, kwidth,
                                               stride=pooling, bias=bias,
                                               transpose=True)
                else:
                    # decoder like with transposed conv
                    # compute padding required based on pooling
                    pad = (2 * pooling - pooling - kwidth)//-2
                    self.conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                                                   stride=pooling,
                                                   padding=pad,
                                                   output_padding=0,
                                                   bias=bias)
                if snorm:
                    self.conv = SpectralNorm(self.conv)
                if activation == 'glu':
                    # TODO: REVIEW
                    raise NotImplementedError
                    self.glu_conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                                                       stride=pooling,
                                                       padding=padding,
                                                       output_padding=pooling-1,
                                                       bias=bias)
                    if snorm:
                        self.glu_conv = spectral_norm(self.glu_conv)
        if activation is not None:
            self.act = activation
        if lnorm:
            self.ln = LayerNorm()
        if dropout > 0:
            self.dout = nn.Dropout(dropout)

    def forward(self, x):

        if len(x.size()) == 4:
            print(type(x.data))
            # inverse case from 1D -> 2D, go 2D -> 1D
            # re-format input from [B, K, C, L] to [B, K * C, L]
            # where C: frequency, L: time
            x = x.squeeze(1)
        if hasattr(self, 'aal_conv'):
            x = self.aal_conv(x)
        if self.linterp:
            x = self.pre_conv(x)
            x = F.upsample(x, scale_factor=self.pooling,
                           mode='linear', align_corners=True)
        if self.enc:
            # apply proper padding
            x = F.pad(x, ((self.kwidth//2)-1, self.kwidth//2))
        #print(x.data.shape)
        h = self.conv(x)
        #print(type(h.data))
        if not self.enc and not self.linterp and not self.convblock:
            # trim last value of h perque el kernel es imparell
            # TODO: generalitzar a kernel parell/imparell
            #print('h size: ', h.size())
            h = h[:, :, :-1]
        linear_h = h
        #print(type(linear_h.data))
        #print(type(h.data))
        if hasattr(self, 'act'):
            if self.act == 'glu':
                hg = self.glu_conv(x)
                h = h * F.sigmoid(hg)
            else:
                h = self.act(h)
        if hasattr(self, 'ln'):
            h = self.ln(h)
        if hasattr(self, 'dout'):
            h = self.dout(h)
        #print(type(h.data))
        return h, linear_h


class DiscBlock(nn.Module):

    def __init__(self, ninputs, kwidth, nfmaps,
                 activation, bnorm=True, pooling=4, SND=False,
                 dropout=0):
        super(DiscBlock, self).__init__()
        self.kwidth = kwidth
        seq_dict = OrderedDict()
        self.conv = nn.Conv1d(ninputs, nfmaps, kwidth, stride=pooling, padding=0)
        seq_dict['conv'] = conv
        if isinstance(activation, str):
            self.act = getattr(nn, activation)()
        else:
            self.act = activation
        self.bnorm = bnorm
        if bnorm:
            self.bn = nn.BatchNorm1d(nfmaps)
        self.dropout = dropout
        if dropout > 0:
            self.dout = nn.Dropout(dropout)

    def forward(self, x):
        #print(x.data.shape)
        x = F.pad(x, ((self.kwidth//2)-1, self.kwidth//2))
        #print(x.data.shape)
        conv_h = self.conv(x)
        if self.bnorm:
            conv_h = self.bn(conv_h)
        conv_h = self.act(conv_h)
        if self.dropout:
            conv_h = self.dout(conv_h)
        return conv_h, x

class Model(nn.Module):

    def __init__(self, name='BaseModel'):
        super(Model, self).__init__()
        self.name = name
        self.optim = None

    def save(self, save_path, step, best_val=False):
        model_name = self.name

        if not hasattr(self, 'saver'):
            self.saver = Saver(self, save_path,
                               optimizer=self.optim)

        self.saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path,
                                   optimizer=self.optim)
            self.saver.load_weights()
        else:
            print('Loading ckpt from ckpt: ', save_path)
            # consider it as ckpt to load per-se
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path, load_last=False):
        # tmp saver
        saver = Saver(self, '.', optimizer=self.optim)
        saver.load_pretrained_ckpt(ckpt_path, load_last)


    def activation(self, name):
        return getattr(nn, name)()

class LayerNorm(nn.Module):

    def __init__(self, *args):
        super(LayerNorm, self).__init__()

    def forward(self, activation):
        if len(activation.size()) == 3:
            ori_size = activation.size()
            activation = activation.view(-1, activation.size(-1))
        else:
            ori_size = None
        means = torch.mean(activation, dim=1, keepdim=True)
        stds = torch.std(activation, dim=1, keepdim=True)
        activation = (activation - means) / stds
        if ori_size is not None:
            activation = activation.view(ori_size)
        return activation


class G2Block(nn.Module):
    """ Conv2D Generator Blocks """

    def __init__(self, ninputs, fmaps, kwidth,
                 activation, padding=None,
                 bnorm=False, dropout=0.,
                 pooling=2, enc=True, bias=False):
        super().__init__()
        if padding is None:
            padding = (kwidth // 2)
        if enc:
            self.conv = nn.Conv2d(ninputs, fmaps, kwidth,
                                  stride=pooling,
                                  padding=padding,
                                  bias=bias)
        else:
            # decoder like with transposed conv
            self.conv = nn.ConvTranspose2d(ninputs, fmaps, kwidth,
                                           stride=pooling,
                                           padding=padding)
        if bnorm:
            self.bn = nn.BatchNorm2d(fmaps)
        if activation is not None:
            self.act = activation
        if dropout > 0:
            self.dout = nn.Dropout2d(dropout)

    def forward(self, x):
        if len(x.size()) == 3:
            # re-format input from [B, C, L] to [B, 1, C, L]
            # where C: frequency, L: time
            x = x.unsqueeze(1)
        h = self.conv(x)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'act'):
            h = self.act(h)
        if hasattr(self, 'dout'):
            h = self.dout(h)
        return h

class Discriminator(Model):
    def __init__(self, ninputs, d_fmaps, kwidth, activation,
                 bnorm=True, pooling=4, SND=False, pool_type='none',
                 dropout=0, Genc=None, pool_size=8, num_spks=None):
        #super().__init__(name='Discriminator')
        super(Discriminator, self).__init__(name='Discriminator')
        if Genc is None:
            if not isinstance(activation, list):
                activation = [activation] * len(d_fmaps)
            self.disc = nn.ModuleList()
            for d_i, d_fmap in enumerate(d_fmaps):
                act = activation[d_i]
                if d_i == 0:
                    inp = ninputs
                else:
                    inp = d_fmaps[d_i - 1]
                self.disc.append(DiscBlock(inp, kwidth, d_fmap,
                                           act, pooling=4))
        else:
            print('Assigning Genc to D')
            # Genc and Denc MUST be same dimensions
            self.disc = Genc
        self.pool_type = pool_type
        if pool_type == 'none':
            # resize tensor to fit into FC directly
            pool_size *= d_fmaps[-1]
            if isinstance(act, nn.LeakyReLU):
                self.fc = nn.Sequential(
                    nn.Linear(16384, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 128)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(pool_size, 256),
                    nn.PReLU(256),
                    nn.Linear(256, 128),
                    nn.PReLU(128),
                    nn.Linear(128, 128)
                )
        elif pool_type == 'rnn':
            if bnorm:
                self.ln = LayerNorm()
            pool_size = 128
            self.rnn = nn.LSTM(d_fmaps[-1], pool_size, batch_first=True,
                               bidirectional=True)
            # bidirectional size
            pool_size *= 2
            self.fc = nn.Linear(pool_size, 1)
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(d_fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_size, 1)
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)
        outs = 1
        if num_spks is not None:
            outs += num_spks

        #self.load_pretrained('weights_EOE_G-Generator1D-61101.ckpt', load_last=True)

    def forward(self, x):
        #print("Input: {}".format(x.data.shape))
        h = x
        # store intermediate activations
        int_act = {}
        for ii, layer in enumerate(self.disc):
            #print(ii)
            h, _ = layer(h)
            #print("After layer: {}".format(h.data.shape))
            int_act['h_{}'.format(ii)] = h
        if self.pool_type == 'rnn':
            if hasattr(self, 'ln'):
                h = self.ln(h)
                int_act['ln_conv'] = h
            ht, state = self.rnn(h.transpose(1, 2))
            h = state[0]
            # concat both states (fwd, bwd)
            hfwd, hbwd = torch.chunk(h, 2, 0)
            h = torch.cat((hfwd, hbwd), dim=2)
            h = h.squeeze(0)
            int_act['rnn_h'] = h
        elif self.pool_type == 'conv':
            h = self.pool_conv(h)
            h = h.view(h.size(0), -1)
            int_act['avg_conv_h'] = h
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
        #print("Final h: {}".format(h.data.shape))
        #print(type(h.data))
        y = self.fc(h)
        #print(type(y.data))
        int_act['logit'] = y
        # return F.sigmoid(y), int_act
        return y, int_act


class Generator(Model):
    def __init__(self, ninputs, enc_fmaps, kwidth,
                 activations, bnorm=False, dropout=0.,
                 pooling=4, z_dim=1024, z_all=False,
                 skip=True, skip_blacklist=[],
                 dec_activations=None, cuda=False,
                 bias=False, aal=False, wd=0.,
                 core2d=False, core2d_kwidth=None,
                 core2d_felayers=1,
                 skip_mode='concat'):
        # aal: anti-aliasing filter prior to each striding conv in enc
        super(Generator, self).__init__()
        self.skip_mode = skip_mode
        self.skip = skip
        self.z_dim = z_dim
        self.z_all = z_all
        self.do_cuda = cuda
        self.core2d = core2d
        self.wd = wd
        self.skip_blacklist = skip_blacklist
        if core2d_kwidth is None:
            core2d_kwidth = kwidth
        self.gen_enc = nn.ModuleList()
        if aal:
            # Make cheby1 filter to include into pytorch conv blocks
            from scipy.signal import cheby1, dlti, dimpulse
            system = dlti(*cheby1(8, 0.05, 0.8 / 2))
            tout, yout = dimpulse(system)
            filter_h = yout[0]
            self.filter_h = filter_h
        else:
            self.filter_h = None

        if isinstance(activations, str):
            activations = getattr(nn, activations)()
        if not isinstance(activations, list):
            activations = [activations] * len(enc_fmaps)
        # always begin with 1D block
        for layer_idx, (fmaps, act) in enumerate(zip(enc_fmaps,
                                                     activations)):
            if layer_idx == 0:
                inp = ninputs
            else:
                inp = enc_fmaps[layer_idx - 1]
            if core2d:
                if layer_idx < core2d_felayers:
                    self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                               padding=None, bnorm=bnorm,
                                               dropout=dropout, pooling=pooling,
                                               enc=True, bias=bias,
                                               aal_h=self.filter_h))
                else:
                    if layer_idx == core2d_felayers:
                        # fmaps is 1 after conv1d blocks
                        inp = 1
                    self.gen_enc.append(G2Block(inp, fmaps, core2d_kwidth, act,
                                                padding=None, bnorm=bnorm,
                                                dropout=dropout, pooling=pooling,
                                                enc=True, bias=bias))
            else:
                self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                           padding=None, snorm=bnorm,
                                           dropout=dropout, pooling=pooling,
                                           enc=True, bias=bias,
                                           aal_h=self.filter_h))
        dec_inp = enc_fmaps[-1]
        if self.core2d:
            # dec_fmaps = enc_fmaps[::-1][1:-2]+ [1, 1]
            dec_fmaps = enc_fmaps[::-1][:-2] + [1, 1]
        else:
            dec_fmaps = enc_fmaps[::-1][1:] + [1]
            # print(dec_fmaps)
        # print(enc_fmaps)
        # print('dec_fmaps: ', dec_fmaps)
        self.gen_dec = nn.ModuleList()
        if dec_activations is None:
            dec_activations = activations

        dec_inp += z_dim

        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps,
                                                     dec_activations)):
            if skip and layer_idx > 0 and layer_idx not in skip_blacklist:
                # print('Adding skip conn input of idx: {} and size:'
                #      ' {}'.format(layer_idx, dec_inp))
                if self.skip_mode == 'concat':
                    dec_inp += enc_fmaps[-(layer_idx + 1)]

            if z_all and layer_idx > 0:
                dec_inp += z_dim

            if layer_idx >= len(dec_fmaps) - 1:
                # act = None #nn.Tanh()
                act = nn.Tanh()
                bnorm = False
                dropout = 0

            if layer_idx < len(dec_fmaps) - 1 and core2d:
                self.gen_dec.append(G2Block(dec_inp,
                                            fmaps, core2d_kwidth + 1, act,
                                            padding=core2d_kwidth // 2,
                                            bnorm=bnorm,
                                            dropout=dropout, pooling=pooling,
                                            enc=False,
                                            bias=bias))
            else:
                if layer_idx == len(dec_fmaps) - 1:
                    # after conv2d channel condensation, fmaps mirror the ones
                    # extracted in 1D encoder
                    dec_inp = enc_fmaps[0]
                    if skip and layer_idx not in skip_blacklist:
                        dec_inp += enc_fmaps[-(layer_idx + 1)]
                self.gen_dec.append(GBlock(dec_inp,
                                           fmaps, kwidth + 1, act,
                                           padding=kwidth // 2,
                                           snorm=bnorm,
                                           dropout=dropout, pooling=pooling,
                                           enc=False,
                                           bias=bias))
            dec_inp = fmaps
        self.load_pretrained('weights_EOE_G-Generator1D-61101.ckpt', load_last=True)
        self.fc = nn.Sequential(
            nn.Linear(16384, 256),
            nn.PReLU(256),
            nn.Linear(256, 128),
            nn.PReLU(128),
            nn.Linear(128, 128)
        )
    def forward(self, x, z=None):
        hi = x
        skips = []
        for l_i, enc_layer in enumerate(self.gen_enc):
            hi, _ = enc_layer(hi)
            # print('ENC {} hi size: {}'.format(l_i, hi.size()))
            if self.skip and l_i < (len(self.gen_enc) - 1):
                # print('Appending skip connection')
                skips.append(hi)
                # print('hi size: ', hi.size())
        # print('=' * 50)
        skips = skips[::-1]
        if z is None:
            # make z
            # z = Variable(torch.randn(x.size(0), self.z_dim, hi.size(2)))
            # z = Variable(torch.randn(*hi.size()))
            z = Variable(torch.randn(hi.size(0), self.z_dim,
                                     *hi.size()[2:]).float()).cuda()
        if len(z.size()) != len(hi.size()):
            raise ValueError('len(z.size) {} != len(hi.size) {}'
                             ''.format(len(z.size()), len(hi.size())))
        if self.do_cuda:
            z = z.cuda()
        if not hasattr(self, 'z'):
            self.z = z
        hi = hi.view(hi.size(0), -1)
        hi = self.fc(hi)
        """# print('z size: ', z.size())
        hi = torch.cat((hi, z), dim=1)
        # print('Input to dec after concating z and enc out: ', hi.size())
        # print('Enc out size: ', hi.size())
        z_up = z
        for l_i, dec_layer in enumerate(self.gen_dec):
            print(l_i)
            # print('dec layer: {} with input: {}'.format(l_i, hi.size()))
            # print('DEC in size: ', hi.size())
            if self.skip and l_i > 0 and l_i not in self.skip_blacklist:
                skip_conn = skips[l_i - 1]
                # print('concating skip {} to hi {}'.format(skip_conn.size(),
                #                                          hi.size()))
                hi = self.skip_merge(skip_conn, hi)
                # print('Merged hi: ', hi.size())
                # hi = torch.cat((hi, skip_conn), dim=1)
            if l_i > 0 and self.z_all:
                # concat z in every layer
                # print('z.size: ', z.size())
                z_up = torch.cat((z_up, z_up), dim=2)
                hi = torch.cat((hi, z_up), dim=1)
            hi = dec_layer(hi)
            # print('-' * 20)
            # print('hi size: ', hi.size())"""
        return hi

    def skip_merge(self, skip, hi):
        if self.skip_mode == 'concat':
            if len(hi.size()) == 4 and len(skip.size()) == 3:
                hi = hi.squeeze(1)
            # 1-D case
            hi_ = torch.cat((skip, hi), dim=1)
        elif self.skip_mode == 'sum':
            hi_ = skip + hi
        else:
            raise ValueError('Urecognized skip mode: ', self.skip_mode)
        return hi_

    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'aal_conv' not in k:
                params.append({'params': v, 'weight_decay': self.wd})
            else:
                print('Excluding param: {} from Genc block'.format(k))
        return params


class generator(nn.Module):

    def __init__(self, dataset='youtubers'):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 62
        self.projected_embed_dim = 128
        self.raw_wav_dim = 64000
        self.latent_dim = self.projected_embed_dim
        self.dataset_name = dataset
        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        #self.d_fmaps = [64, 128, 256, 512, 1024, 1024]
        self.d_fmaps = [16, 32, 128, 256, 512, 1024]
        self.act = [nn.PReLU(fmaps) for fmaps in [64, 128, 256, 512, 1024, 1024]]
        self.audio_embedding = Discriminator(1, self.d_fmaps, 15, nn.LeakyReLU(0.3))
        #self.audio_embedding = Generator(1, self.d_fmaps, 31, self.act)
        self.ngf = 64

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0, bias=False)),
            #nn.BatchNorm2d(self.ngf * 8),
            nn.Dropout(),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)),
            nn.Dropout(),
            #nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)),
            nn.Dropout(),
            #nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 2,self.ngf, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(self.ngf),
            nn.Dropout(),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            SpectralNorm(nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False)),
            #nn.Dropout2d(),
            nn.Tanh()
             # state size. (num_channels) x 64 x 64
            )




    def forward(self, raw_wav, embed_vector, z, aux_z, project=False, only_wav=True):
        #melspec = raw_wav.view(raw_wav.size(0), -1).unsqueeze(1)
        #y, wav_embedding = self.audio_embedding(melspec)
        y, wav_embedding = self.audio_embedding(raw_wav.unsqueeze(1))
        #y = self.audio_embedding(raw_wav.unsqueeze(1))
        if self.dataset_name == 'youtubers' and not project:
            if not only_wav:
                padding = Variable(torch.cuda.FloatTensor(embed_vector.data.shape[0], self.projected_embed_dim - 62).fill_(
                    0).float()).cuda()
                #padding = Variable(torch.FloatTensor(embed_vector.data.shape[0], self.projected_embed_dim - 62).fill_(
                #    0).float())
                projected_embed = torch.cat([embed_vector, padding, y], 1).unsqueeze(2).unsqueeze(3)
                #print(projected_embed.data.shape)
            else:
                projected_embed = y.unsqueeze(2).unsqueeze(3)
        else:
            projected_embed = self.projection(embed_vector)
            projected_embed = torch.cat([projected_embed, y], 1).unsqueeze(2).unsqueeze(3)
        #latent_vector = torch.cat([projected_embed, z], 1)
        latent_vector = projected_embed
        output = self.netG(latent_vector)
        if not project and not only_wav:
            projected_embed = embed_vector
        return output, projected_embed

