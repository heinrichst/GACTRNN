# -*- coding: utf-8 -*-
"""
Definition of various CTRNN models with multiple modules
Heinrich 2020
"""

import numpy as np
from tensorflow import keras
import models.keras_extend.ctrnn as keras_layers_ctrnn
import models.keras_extend.xctrnn as keras_layers_xctrnn


def ctrnn_name(name, size, connectivity='dense'):
    name = name.lower()
    if (isinstance(size, (list, tuple, np.ndarray))):
        for k in range(len(size)):
            name += "_" + str(size[k])
    name += "_" + connectivity[0:5]
    return name


def CTRNN_model(size,  #[[3,2],[4,3,2]]
                inp_size,
                out_size=None,
                out_act='linear',
                tau=None,
                sigma=None,
                ctrnn_connectivity='adjacent',
                timesteps_max='100',
                rec_init='orthogonal',
                return_sequences=False,
                type='CTRNN'
                ):
    """
    out_act: 'elu', 'relu', 'selu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'exponential'
    ctrnn_connectivity: 'partitioned', 'clocked', 'adjacent', 'dense'
    """

    inputs = []
    if (isinstance(inp_size, (list, tuple, np.ndarray))):
        for k in inp_size:
            inputs += [keras.layers.Input(shape=(timesteps_max, k), name="input_" + str(len(inputs)+1))]
        if len(inputs) > 1:
            input = keras.layers.concatenate(inputs)
        else:
            input = inputs[-1]
    else:
        inputs += [keras.layers.Input(shape=(timesteps_max, inp_size), name="input_1")]
        input = inputs[-1]

    def CTRNN_wrapper(_size, _tau, _sigma=None, _rs=return_sequences, _name_pf=""):
        if _tau == None:
            _tau = [1. for k in _size] if (isinstance(_size, (list, tuple, np.ndarray))) else 0.
        if _tau is not None:
            print("tau:", _tau)
        if _sigma == None:
            _sigma = [max(1.,float(k)/2.) for k in _tau] if (isinstance(_tau, (list, tuple, np.ndarray))) else 0.
        if type == 'GCTRNN':
            return keras_layers_xctrnn.GCTRNN(
                _size, tau_vec=_tau, connectivity=ctrnn_connectivity,
                recurrent_initializer=rec_init,
                return_sequences=_rs, return_state=False,
                name=ctrnn_name(type + _name_pf, _size, ctrnn_connectivity))
        elif type == 'GACTRNN':
            return keras_layers_xctrnn.GACTRNN(
                _size, tau_vec=_tau, connectivity=ctrnn_connectivity,
                recurrent_initializer=rec_init,
                return_sequences=_rs, return_state=False,
                name=ctrnn_name(type + _name_pf, _size,
                                ctrnn_connectivity))
        elif type == 'ACTRNN':
            return keras_layers_xctrnn.ACTRNN(
                _size, tau_vec=_tau, connectivity=ctrnn_connectivity,
                recurrent_initializer=rec_init,
                return_sequences=_rs, return_state=False,
                name=ctrnn_name(type + _name_pf, _size, ctrnn_connectivity))
        else:
            return keras_layers_ctrnn.CTRNN(
                _size, tau_vec=_tau, connectivity=ctrnn_connectivity,
                recurrent_initializer=rec_init,
                return_sequences=_rs, return_state=False,
                name=ctrnn_name(type + _name_pf, _size, ctrnn_connectivity))

    if (isinstance(size, (list, tuple, np.ndarray))):
        if (isinstance(size[0], (list, tuple, np.ndarray))):
            ctrnns = []
            hiddens = []
            for h in range(len(size)):
                t = tau[h] if (isinstance(tau, (list, tuple, np.ndarray))) else tau
                s = sigma[h] if (isinstance(sigma, (list, tuple, np.ndarray))) else sigma
                rs = True if len(size) > (h+1) else return_sequences
                ctrnns += [CTRNN_wrapper(size[h], _tau=t, _sigma=s, _rs=rs, _name_pf="_h"+str(h))]
                if h == 0:
                    hiddens += [ctrnns[-1](input)]
                else:
                    hiddens += [ctrnns[-1](hiddens[-1][0])]
            #hidden = hiddens[-1]
            hidden = hiddens[-1][0]
        else:
            ctrnn = CTRNN_wrapper(size, _tau=tau, _sigma=sigma)
            #hidden = ctrnn(input)
            hidden = ctrnn(input)[0]
    else:
        ctrnn = CTRNN_wrapper(size, _tau=tau, _sigma=sigma)
        #hidden = ctrnn(input)
        hidden = ctrnn(input)[0]

    if out_size == None:
        out_size = inp_size

    outputs = []
    if (isinstance(out_size, (list, tuple, np.ndarray))):
        for k in out_size:
            outputs += [keras.layers.Dense(k, activation=out_act, name="output_"+out_act+"_"+str(len(outputs)+1))(hidden)]
    else:
        outputs += [keras.layers.Dense(out_size, activation=out_act, name="output_"+out_act+"_1")(hidden)]

    model = keras.Model(inputs, outputs)

    return model


def CTRNN_model_act(model, type, size, ctrnn_connectivity):
    # helper method to read out model activation
    full_ctrnn_name = ctrnn_name(type, size, ctrnn_connectivity)
    in_act_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(full_ctrnn_name).output[0])
    return in_act_model


def CTRNN_model_tsc(model, type, size, ctrnn_connectivity):
    # helper method to read out model timescales
    full_ctrnn_name = ctrnn_name(type, size, ctrnn_connectivity)
    in_tsc_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(full_ctrnn_name).output[1])
    return in_tsc_model


def LSTM_baseline(size,
                  inp_size,
                  out_size=None,
                  out_act='linear',
                  timesteps_max='100',
                  rec_init='orthogonal',
                  return_sequences=False
                  ):

    inputs = []
    if (isinstance(inp_size, (list, tuple, np.ndarray))):
        for k in inp_size:
            inputs += [keras.layers.Input(shape=(timesteps_max, k), name="input_" + str(len(inputs)+1))]
        if len(inputs) > 1:
            input = keras.layers.concatenate(inputs)
        else:
            input = inputs[-1]
    else:
        inputs += [keras.layers.Input(shape=(timesteps_max, inp_size), name="input_1")]
        input = inputs[-1]

    if (isinstance(size, (list, tuple, np.ndarray))):
        if (isinstance(size[0], (list, tuple, np.ndarray))):
            srns = []
            hiddens = []
            for h in range(len(size)):
                rs = True if len(size) > (h+1) else return_sequences
                srns += [keras.layers.LSTM(
                    sum(size[h]), name=ctrnn_name("srn_", size[h]),
                    recurrent_initializer=rec_init,
                    return_sequences=rs)]
                if h == 0:
                    hiddens += [srns[-1](input)]
                else:
                    hiddens += [srns[-1](hiddens[-1])]
            hidden = hiddens[-1]
        else:
            srn = keras.layers.LSTM(
                sum(size), name=ctrnn_name("srn_", size),
                recurrent_initializer=rec_init,
                return_sequences=return_sequences)
            hidden = srn(input)
    else:
        srn = keras.layers.LSTM(size, name=ctrnn_name("srn_", size),
                                     recurrent_initializer=rec_init,
                                     return_sequences=return_sequences)
        hidden = srn(input)

    if out_size == None:
        out_size = inp_size

    outputs = []
    if (isinstance(out_size, (list, tuple, np.ndarray))):
        for k in out_size:
            outputs += [keras.layers.Dense(k, activation=out_act, name="output_"+out_act+"_"+str(len(outputs)+1))(hidden)]
    else:
        outputs += [keras.layers.Dense(out_size, activation=out_act, name="output_"+out_act+"_1")(hidden)]

    model = keras.Model(inputs, outputs)

    return model


def GRU_baseline(size,
                  inp_size,
                  out_size=None,
                  out_act='linear',
                  timesteps_max='100',
                  rec_init='orthogonal',
                  return_sequences=False
                  ):

    inputs = []
    if (isinstance(inp_size, (list, tuple, np.ndarray))):
        for k in inp_size:
            inputs += [keras.layers.Input(shape=(timesteps_max, k), name="input_" + str(len(inputs)+1))]
        if len(inputs) > 1:
            input = keras.layers.concatenate(inputs)
        else:
            input = inputs[-1]
    else:
        inputs += [keras.layers.Input(shape=(timesteps_max, inp_size), name="input_1")]
        input = inputs[-1]

    if (isinstance(size, (list, tuple, np.ndarray))):
        if (isinstance(size[0], (list, tuple, np.ndarray))):
            srns = []
            hiddens = []
            for h in range(len(size)):
                rs = True if len(size) > (h+1) else return_sequences
                srns += [keras.layers.GRU(
                    sum(size[h]), name=ctrnn_name("srn_", size[h]),
                    recurrent_initializer=rec_init,
                    return_sequences=rs)]
                if h == 0:
                    hiddens += [srns[-1](input)]
                else:
                    hiddens += [srns[-1](hiddens[-1])]
            hidden = hiddens[-1]
        else:
            srn = keras.layers.GRU(
                sum(size), name=ctrnn_name("srn_", size),
                recurrent_initializer=rec_init,
                return_sequences=return_sequences)
            hidden = srn(input)
    else:
        srn = keras.layers.GRU(size, name=ctrnn_name("srn_", size),
                                     recurrent_initializer=rec_init,
                                     return_sequences=return_sequences)
        hidden = srn(input)

    if out_size == None:
        out_size = inp_size

    outputs = []
    if (isinstance(out_size, (list, tuple, np.ndarray))):
        for k in out_size:
            outputs += [keras.layers.Dense(k, activation=out_act, name="output_"+out_act+"_"+str(len(outputs)+1))(hidden)]
    else:
        outputs += [keras.layers.Dense(out_size, activation=out_act, name="output_"+out_act+"_1")(hidden)]

    model = keras.Model(inputs, outputs)

    return model


def SimpleRNN_baseline(size,  #[[3,2],[4,3,2]]
                       inp_size,
                       out_size=None,
                       out_act='linear',
                       timesteps_max='100',
                       rec_init='orthogonal',
                       return_sequences=False
                       ):

    inputs = []
    if (isinstance(inp_size, (list, tuple, np.ndarray))):
        for k in inp_size:
            inputs += [keras.layers.Input(shape=(timesteps_max, k), name="input_" + str(len(inputs)+1))]
        if len(inputs) > 1:
            input = keras.layers.concatenate(inputs)
        else:
            input = inputs[-1]
    else:
        inputs += [keras.layers.Input(shape=(timesteps_max, inp_size), name="input_1")]
        input = inputs[-1]

    if (isinstance(size, (list, tuple, np.ndarray))):
        if (isinstance(size[0], (list, tuple, np.ndarray))):
            srns = []
            hiddens = []
            for h in range(len(size)):
                rs = True if len(size) > (h+1) else return_sequences
                srns += [keras.layers.SimpleRNN(
                    sum(size[h]), name=ctrnn_name("srn_h"+str(h), size[h]),
                    recurrent_initializer=rec_init,
                    return_sequences=rs)]
                if h == 0:
                    hiddens += [srns[-1](input)]
                else:
                    hiddens += [srns[-1](hiddens[-1])]
            hidden = hiddens[-1]
        else:
            srn = keras.layers.SimpleRNN(
                sum(size), name=ctrnn_name("srn", size),
                recurrent_initializer=rec_init,
                return_sequences=return_sequences)
            hidden = srn(input)
    else:
        srn = keras.layers.SimpleRNN(size, name=ctrnn_name("srn", size),
                                     recurrent_initializer=rec_init,
                                     return_sequences=return_sequences)
        hidden = srn(input)

    if out_size == None:
        out_size = inp_size

    outputs = []
    if (isinstance(out_size, (list, tuple, np.ndarray))):
        for k in out_size:
            outputs += [keras.layers.Dense(k, activation=out_act, name="output_"+out_act+"_"+str(len(outputs)+1))(hidden)]
    else:
        outputs += [keras.layers.Dense(out_size, activation=out_act, name="output_"+out_act+"_1")(hidden)]

    model = keras.Model(inputs, outputs)

    return model


def SRN_baseline(size,
                 inp_size,
                 out_size=None,
                 out_act='linear',
                 timesteps_max='100',
                 rec_init='orthogonal',
                 return_sequences=False
                 ):
    return SimpleRNN_baseline(
        size, inp_size, out_size=out_size, out_act=out_act,
        timesteps_max=timesteps_max, rec_init=rec_init,
        return_sequences=return_sequences)


# Main, needed only for debug:
if __name__ == '__main__':

    #model = SimpleRNN_baseline(size=[[5,4,2],[4,4,2],[3,4,2]], inp_size=[2], return_sequences=True)
    model = CTRNN_model([5, 3], inp_size=[1,2], return_sequences=True, type='CTRNN')

    model.summary()
    keras.utils.plot_model(model)
