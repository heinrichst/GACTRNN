# -*- coding: utf-8 -*-
"""
This tensorflow extension implements ACTRNN, GCTRNN, GACTRNN.
Heinrich 2020
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers.recurrent \
    import _generate_dropout_mask, _generate_zero_filled_state_for_cell, RNN
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, gen_array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


_MAX_TIMESCALE = 999999
_MAX_SIGMA = 500000
_ALMOST_ONE = 0.999999
_ALMOST_ZERO = 0.000001
_DEBUGMODE = False


@tf_export('keras.layers.ACTRNNCell')
class ACTRNNCell(Layer):
    """Cell class for ACTRNNCell.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ACTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules
                                  for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., float(tau_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.modules > 1:
                self.tau_vec = [max(1., float(tau_vec)) for k in range(self.modules)]
            else:
                self.tau_vec = [max(1., float(tau_vec))]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=self.dtype, shape=[self.units],
                name="taus")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.w_tau_initializer = initializers.get(w_tau_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.w_tau_regularizer = regularizers.get(w_tau_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.w_tau_constraint = constraints.get(w_tau_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
        else:  # == 'dense'
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.w_tau = self.add_weight(
            shape=(self.units,),
            name='wtimescales',
            initializer=self.w_tau_initializer,
            regularizer=self.w_tau_regularizer,
            constraint=self.w_tau_constraint)

        """
        self.log_taus = K.log(self.taus)
        """
        self.log_taus = K.log(self.taus - _ALMOST_ONE)
        # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs          # for better readability
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        h = K.dot(x, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'partitioned':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
        else:  # == 'dense'
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])

        """
        #taus_act = K.exp(K.relu(self.w_tau + self.log_taus))
        #taus_act = K.exp(K.pow(self.w_tau + self.log_taus, 2))
        #taus_act = K.exp(K.pow(self.w_tau, 2) + self.log_taus)
        #taus_act = K.exp(self.w_tau + self.log_taus)
        taus_act = K.exp(K.relu(self.w_tau) + self.log_taus)
        """
        taus_act = K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        t = prev_z * 0 + taus_act

        if _DEBUGMODE:
            x = gen_array_ops.check_numerics(x, 'AVCTRNNCell: Numeric error for x')
            prev_y = gen_array_ops.check_numerics(prev_y, 'AVCTRNNCell: Numeric error for prev_y')
            prev_z = gen_array_ops.check_numerics(prev_z, 'AVCTRNNCell: Numeric error for prev_z')
            h = gen_array_ops.check_numerics(h, 'AVCTRNNCell: Numeric error for h')
            r = gen_array_ops.check_numerics(r, 'AVCTRNNCell: Numeric error for r')
            y = gen_array_ops.check_numerics(y, 'AVCTRNNCell: Numeric error for y')
            z = gen_array_ops.check_numerics(z, 'AVCTRNNCell: Numeric error for z')
            t = gen_array_ops.check_numerics(t, 'AVCTRNNCell: Numeric error for t')
            print("shapes taus_act", array_ops.shape(taus_act))
            print("shapes y", array_ops.shape(y))
            print("shapes z", array_ops.shape(z))
            print("shapes t", array_ops.shape(t))

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True
        #return y, [y, z]
        """
        StH 02.06.2020:
        I added t to the output to return the effective timescale.
        This is not necessary for the model and does not change any computation.
        The purpose is to read out the effective timescales during activation.
        """
        return [y, t], [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        """
        return {"tau_bias": self.get_tau()}
        """
        return {"tau_bias": self.get_tau(), "tau_biasraw": self.get_tau_raw()}

    def get_tau(self):
        """
        return K.exp(self.w_tau + self.log_taus) if self.built else None
        """
        return K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE if self.built else None

    def get_tau_raw(self):
        """
        return K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE if self.built else None
        """
        return self.w_tau if self.built else None


    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(ACTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.GCTRNNCell')
class GCTRNNCell(Layer):

    """Cell class for GCTRNNCell.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        tau_kernel_initializer: Initializer for the tau_kernel vector.
        tau_recurrent_initializer: Initializer for the tau_recurrent vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        tau_kernel_regularizer: Regularizer function applied to the tau_kernel vector.
        tau_recurrent_regularizer: Regularizer function applied to the tau_recurrent vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        tau_kernel_constraint: Constraint function applied to the tau_kernel vector.
        tau_recurrent_constraint: Constraint function applied to the tau_recurrent vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 tau_kernel_initializer='zeros',
                 tau_recurrent_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 tau_kernel_regularizer=None,
                 tau_recurrent_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 tau_kernel_constraint=None,
                 tau_recurrent_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GCTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules
                                  for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., float(tau_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.modules > 1:
                self.tau_vec = [max(1., float(tau_vec)) for k in range(self.modules)]
            else:
                self.tau_vec = [max(1., float(tau_vec))]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=self.dtype, shape=[self.units],
                name="taus")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.tau_kernel_initializer = initializers.get(tau_kernel_initializer)
        self.tau_recurrent_initializer = initializers.get(tau_recurrent_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.tau_kernel_regularizer = regularizers.get(tau_kernel_regularizer)
        self.tau_recurrent_regularizer = regularizers.get(tau_recurrent_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.tau_kernel_constraint = constraints.get(tau_kernel_constraint)
        self.tau_recurrent_constraint = constraints.get(tau_recurrent_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = (self.units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[
                               max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        else:  # == 'dense'
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
            self.tau_recurrent = [self.add_weight(
                    shape=(self.units, self.units),
                    name='tau_recurrent',
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.tau_kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='tau_kernel',
            initializer=self.tau_kernel_initializer,
            regularizer=self.tau_kernel_regularizer,
            constraint=self.tau_kernel_constraint)

        self.log_taus = K.log(self.taus - _ALMOST_ONE)
        # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs          # for better readability
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        h = K.dot(x, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'partitioned':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.tau_recurrent[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.tau_recurrent[k])
                    for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.tau_recurrent[k])
                    for k in range(self.modules)], 1)
        else:  # == 'dense'
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])
            tau_r = K.dot(prev_y, self.tau_recurrent[0])

        taus_act = K.exp(K.dot(x, self.tau_kernel) + tau_r + self.log_taus) + _ALMOST_ONE

        #taus_act_prev = K.exp(self.tau_kernel + self.log_taus) + _ALMOST_ONE

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        t = prev_z * 0. + taus_act

        if _DEBUGMODE:
            x = gen_array_ops.check_numerics(x, 'AVCTRNNCell: Numeric error for x')
            prev_y = gen_array_ops.check_numerics(prev_y, 'AVCTRNNCell: Numeric error for prev_y')
            prev_z = gen_array_ops.check_numerics(prev_z, 'AVCTRNNCell: Numeric error for prev_z')
            h = gen_array_ops.check_numerics(h, 'AVCTRNNCell: Numeric error for h')
            r = gen_array_ops.check_numerics(r, 'AVCTRNNCell: Numeric error for r')
            y = gen_array_ops.check_numerics(y, 'AVCTRNNCell: Numeric error for y')
            z = gen_array_ops.check_numerics(z, 'AVCTRNNCell: Numeric error for z')
            t = gen_array_ops.check_numerics(t, 'AVCTRNNCell: Numeric error for t')
            print("shapes y", array_ops.shape(y))
            print("shapes z", array_ops.shape(z))
            print("shapes t", array_ops.shape(t))

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True
        #return y, [y, z]
        """
        StH 02.06.2020:
        I added t to the output to return the effective timescale.
        This is not necessary for the model and does not change any computation.
        The purpose is to read out the effective timescales during activation.
        """
        return [y, t], [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return {"tau_recurrent": self.get_tau_recurrent(),
                "tau_kernel": self.get_tau_kernel()}

    def get_tau_bias(self):
        return None

    def get_tau_kernel(self):
        #return K.exp(self.tau_kernel) if self.built else None
        return self.tau_kernel if self.built else None

    def get_tau_recurrent(self):
        tau_recurrent_flat = array_ops.constant(
            0., dtype=self.dtype, shape=[self.units, self.units], name="taus")
        if self.connectivity == 'partitioned':
            #TODO
            pass
        elif self.connectivity == 'clocked':
            #TODO
            pass
        if self.connectivity == 'adjacent':
            #TODO
            pass
        else:  # == 'dense'
            tau_recurrent_flat = K.exp(
                self.tau_recurrent[0]) if self.built else array_ops.constant(
                0., dtype=self.dtype, shape=[self.units, self.units],
                name="taus")

        return tau_recurrent_flat

    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'tau_kernel_initializer':
                initializers.serialize(self.tau_kernel_initializer),
            'tau_recurrent_initializer':
                initializers.serialize(self.tau_recurrent_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'tau_kernel_regularizer':
                regularizers.serialize(self.tau_kernel_regularizer),
            'tau_recurrent_regularizer':
                regularizers.serialize(self.tau_recurrent_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'tau_kernel_constraint':
                constraints.serialize(self.tau_kernel_constraint),
            'tau_recurrent_constraint':
                constraints.serialize(self.tau_recurrent_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(GCTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.GACTRNNCell')
class GACTRNNCell(Layer):
#class GACTRNNCell_v2rec(Layer):

    """Cell class for GACTRNNCell.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        tau_recurrent_initializer: Initializer for the tau_recurrent vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        tau_recurrent_regularizer: Regularizer function applied to the tau_recurrent vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        tau_recurrent_constraint: Constraint function applied to the tau_recurrent vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 tau_bias_initializer='zeros',
                 tau_kernel_initializer='zeros',
                 tau_recurrent_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 tau_bias_regularizer=None,
                 tau_kernel_regularizer=None,
                 tau_recurrent_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 tau_bias_constraint=None,
                 tau_kernel_constraint=None,
                 tau_recurrent_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GACTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules
                                  for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., float(tau_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.modules > 1:
                self.tau_vec = [max(1., float(tau_vec)) for k in range(self.modules)]
            else:
                self.tau_vec = [max(1., float(tau_vec))]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=self.dtype, shape=[self.units],
                name="taus")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.tau_recurrent_initializer = initializers.get(tau_recurrent_initializer)
        self.tau_bias_initializer = initializers.get(tau_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.tau_recurrent_regularizer = regularizers.get(tau_recurrent_regularizer)
        self.tau_bias_regularizer = regularizers.get(tau_bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.tau_recurrent_constraint = constraints.get(tau_recurrent_constraint)
        self.tau_bias_constraint = constraints.get(tau_bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[
                               max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        else:  # == 'dense'
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
            self.tau_recurrent = [self.add_weight(
                    shape=(self.units, self.units),
                    name='tau_recurrent',
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.tau_bias = self.add_weight(
            shape=(self.units,),
            name='tau_bias',
            initializer=self.tau_bias_initializer,
            regularizer=self.tau_bias_regularizer,
            constraint=self.tau_bias_constraint)

        self.log_taus = K.log(self.taus - _ALMOST_ONE)
        # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs          # for better readability
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        h = K.dot(x, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'partitioned':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.tau_recurrent[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.tau_recurrent[k])
                    for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.tau_recurrent[k])
                    for k in range(self.modules)], 1)
        else:  # == 'dense'
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])
            tau_r = K.dot(prev_y, self.tau_recurrent[0])

        taus_act = K.exp(self.tau_bias + tau_r + self.log_taus) + _ALMOST_ONE

        #taus_act_prev = K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        t = prev_z * 0. + taus_act

        if _DEBUGMODE:
            x = gen_array_ops.check_numerics(x, 'AVCTRNNCell: Numeric error for x')
            prev_y = gen_array_ops.check_numerics(prev_y, 'AVCTRNNCell: Numeric error for prev_y')
            prev_z = gen_array_ops.check_numerics(prev_z, 'AVCTRNNCell: Numeric error for prev_z')
            h = gen_array_ops.check_numerics(h, 'AVCTRNNCell: Numeric error for h')
            r = gen_array_ops.check_numerics(r, 'AVCTRNNCell: Numeric error for r')
            y = gen_array_ops.check_numerics(y, 'AVCTRNNCell: Numeric error for y')
            z = gen_array_ops.check_numerics(z, 'AVCTRNNCell: Numeric error for z')
            t = gen_array_ops.check_numerics(t, 'AVCTRNNCell: Numeric error for t')
            print("shapes y", array_ops.shape(y))
            print("shapes z", array_ops.shape(z))
            print("shapes t", array_ops.shape(t))

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True
        #return y, [y, z]
        """
        StH 02.06.2020:
        I added t to the output to return the effective timescale.
        This is not necessary for the model and does not change any computation.
        The purpose is to read out the effective timescales during activation.
        """
        return [y, t], [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return {"tau_bias": self.get_tau_bias(),
                "tau_recurrent": self.get_tau_recurrent()}

    def get_tau_bias(self):
        return K.exp(self.tau_bias + self.log_taus) + _ALMOST_ONE if self.built else None

    def get_tau_kernel(self):
        return None

    def get_tau_recurrent(self):
        tau_recurrent_flat = array_ops.constant(
            0., dtype=self.dtype, shape=[self.units, self.units], name="taus")
        if self.connectivity == 'partitioned':
            #TODO
            pass
        elif self.connectivity == 'clocked':
            #TODO
            pass
        if self.connectivity == 'adjacent':
            #TODO
            pass
        else:  # == 'dense'
            tau_recurrent_flat = K.exp(
                self.tau_recurrent[0]) if self.built else array_ops.constant(
                0., dtype=self.dtype, shape=[self.units, self.units],
                name="taus")

        return tau_recurrent_flat

    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'tau_recurrent_initializer':
                initializers.serialize(self.tau_recurrent_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'tau_recurrent_regularizer':
                regularizers.serialize(self.tau_recurrent_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'tau_recurrent_constraint':
                constraints.serialize(self.tau_recurrent_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(GACTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.GACTRNNCell')
class GACTRNNCell(Layer):

    """Cell class for GACTRNNCell.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        tau_kernel_initializer: Initializer for the tau_kernel vector.
        tau_recurrent_initializer: Initializer for the tau_recurrent vector.
        tau_bias_initializer: Initializer for the tau_bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        tau_kernel_regularizer: Regularizer function applied to the tau_kernel vector.
        tau_recurrent_regularizer: Regularizer function applied to the tau_recurrent vector.
        tau_bias_regularizer: Regularizer function applied to the tau_bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        tau_kernel_constraint: Constraint function applied to the tau_kernel vector.
        tau_recurrent_constraint: Constraint function applied to the tau_recurrent vector.
        tau_bias_constraint: Constraint function applied to the tau_bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 tau_kernel_initializer='zeros',
                 tau_recurrent_initializer='zeros',
                 tau_bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 tau_kernel_regularizer=None,
                 tau_recurrent_regularizer=None,
                 tau_bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 tau_kernel_constraint=None,
                 tau_recurrent_constraint=None,
                 tau_bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(GACTRNNCell, self).__init__(**kwargs)
        self.connectivity = connectivity

        if isinstance(units_vec, list):
            self.units_vec = units_vec[:]
            self.modules = len(units_vec)
            self.units = 0
            for k in range(self.modules):
                self.units += units_vec[k]
        else:
            self.units = units_vec
            if modules is not None and modules > 1:
                self.modules = int(modules)
                self.units_vec = [units_vec//modules
                                  for k in range(self.modules)]
            else:
                self.modules = 1
                self.units_vec = [units_vec]
                self.connectivity = 'dense'

        # smallest timescale should be 1.0
        if isinstance(tau_vec, list):
            if len(tau_vec) != self.modules:
                raise ValueError("vector of tau must be of same size as "
                                 "num_modules or size of vector of num_units")
            for k in  range(self.modules):
                if tau_vec[k] < 1:
                    raise ValueError("time scales must be equal or larger 1")
            self.tau_vec = tau_vec[:]
            self.taus = array_ops.constant(
                [[max(1., float(tau_vec[k]))] for k in range(self.modules)
                 for n in range(self.units_vec[k])],
                dtype=self.dtype, shape=[self.units],
                name="taus")
        else:
            if tau_vec < 1:
                raise ValueError("time scales must be equal or larger 1")
            if self.modules > 1:
                self.tau_vec = [max(1., float(tau_vec)) for k in range(self.modules)]
            else:
                self.tau_vec = [max(1., float(tau_vec))]
            self.taus = array_ops.constant(
                max(1., tau_vec), dtype=self.dtype, shape=[self.units],
                name="taus")

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.tau_kernel_initializer = initializers.get(tau_kernel_initializer)
        self.tau_recurrent_initializer = initializers.get(tau_recurrent_initializer)
        self.tau_bias_initializer = initializers.get(tau_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.tau_kernel_regularizer = regularizers.get(tau_kernel_regularizer)
        self.tau_recurrent_regularizer = regularizers.get(tau_recurrent_regularizer)
        self.tau_bias_regularizer = regularizers.get(tau_bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.tau_kernel_constraint = constraints.get(tau_kernel_constraint)
        self.tau_recurrent_constraint = constraints.get(tau_recurrent_constraint)
        self.tau_bias_constraint = constraints.get(tau_bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.units, self.units)
        self.output_size = (self.units, self.units)
        #self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.connectivity == 'partitioned':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(self.units_vec[k], self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        elif self.connectivity == 'clocked':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[k:self.modules]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        elif self.connectivity == 'adjacent':
            self.recurrent_kernel_vec = []
            self.tau_recurrent = []
            for k in range(self.modules):
                self.recurrent_kernel_vec += [self.add_weight(
                    shape=(sum(self.units_vec[
                           max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='recurrent_kernel' + str(k),
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
                self.tau_recurrent += [self.add_weight(
                    shape=(sum(self.units_vec[
                               max(0, k - 1):min(self.modules, k + 1 + 1)]),
                           self.units_vec[k]),
                    name='tau_recurrent' + str(k),
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]
        else:  # == 'dense'
            self.recurrent_kernel_vec = [self.add_weight(
                    shape=(self.units, self.units),
                    name='recurrent_kernel',
                    initializer=self.recurrent_initializer,
                    regularizer=self.recurrent_regularizer,
                    constraint=self.recurrent_constraint)]
            self.tau_recurrent = [self.add_weight(
                    shape=(self.units, self.units),
                    name='tau_recurrent',
                    initializer=self.tau_recurrent_initializer,
                    regularizer=self.tau_recurrent_regularizer,
                    constraint=self.tau_recurrent_constraint)]

        if self.use_bias:
            self.bias = self.add_weight(
                    shape=(self.units,),
                    name='bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint)
        else:
            self.bias = None

        self.tau_kernel = self.add_weight(
            shape=(input_dim, self.units),
            name='tau_kernel',
            initializer=self.tau_kernel_initializer,
            regularizer=self.tau_kernel_regularizer,
            constraint=self.tau_kernel_constraint)

        self.tau_bias = self.add_weight(
            shape=(self.units,),
            name='tau_bias',
            initializer=self.tau_bias_initializer,
            regularizer=self.tau_bias_regularizer,
            constraint=self.tau_bias_constraint)

        self.log_taus = K.log(self.taus - _ALMOST_ONE)
        # we do this here in order to space one recurring computation

        self.built = True

    def call(self, inputs, states, training=None):
        x = inputs          # for better readability
        prev_y = states[0]  # previous output state
        prev_z = states[1]  # previous internal state
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(x),
                    self.dropout,
                    training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                    array_ops.ones_like(prev_y),
                    self.recurrent_dropout,
                    training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask

        if dp_mask is not None:
            x *= dp_mask
        h = K.dot(x, self.kernel)

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_y *= rec_dp_mask
            #prev_z *= rec_dp_mask #TODO: test whether this should masked as well

        if self.connectivity == 'partitioned':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(prev_y_vec[k], self.tau_recurrent[k])
                 for k in range(self.modules)], 1)
        elif self.connectivity == 'clocked':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[k:self.modules], 1),
                    self.tau_recurrent[k])
                    for k in range(self.modules)], 1)
        elif self.connectivity == 'adjacent':
            prev_y_vec = array_ops.split(prev_y, self.units_vec, axis=1)
            r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.recurrent_kernel_vec[k])
                 for k in range(self.modules)], 1)
            tau_r = array_ops.concat(
                [K.dot(array_ops.concat(
                    prev_y_vec[max(0, k - 1):min(self.modules, k + 1 + 1)], 1),
                    self.tau_recurrent[k])
                    for k in range(self.modules)], 1)
        else:  # == 'dense'
            r = K.dot(prev_y, self.recurrent_kernel_vec[0])
            tau_r = K.dot(prev_y, self.tau_recurrent[0])

        taus_act = K.exp(self.tau_bias + K.dot(x, self.tau_kernel) + tau_r + self.log_taus) + _ALMOST_ONE

        #taus_act_prev = K.exp(self.w_tau + self.log_taus) + _ALMOST_ONE

        z = (1. - 1. / taus_act) * prev_z + (1. / taus_act) * (h + r)

        if self.activation is not None:
            y = self.activation(z)
        else:
            y = z

        t = prev_z  * 0. + taus_act

        if _DEBUGMODE:
            x = gen_array_ops.check_numerics(x, 'AVCTRNNCell: Numeric error for x')
            prev_y = gen_array_ops.check_numerics(prev_y, 'AVCTRNNCell: Numeric error for prev_y')
            prev_z = gen_array_ops.check_numerics(prev_z, 'AVCTRNNCell: Numeric error for prev_z')
            h = gen_array_ops.check_numerics(h, 'AVCTRNNCell: Numeric error for h')
            r = gen_array_ops.check_numerics(r, 'AVCTRNNCell: Numeric error for r')
            y = gen_array_ops.check_numerics(y, 'AVCTRNNCell: Numeric error for y')
            z = gen_array_ops.check_numerics(z, 'AVCTRNNCell: Numeric error for z')
            t = gen_array_ops.check_numerics(t, 'AVCTRNNCell: Numeric error for t')
            print("shapes y", array_ops.shape(y))
            print("shapes z", array_ops.shape(z))
            print("shapes t", array_ops.shape(t))

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None and not context.executing_eagerly():
                # This would be harmless to set in eager mode, but eager tensors
                # disallow setting arbitrary attributes.
                y._uses_learning_phase = True
        #return y, [y, z]
        """
        StH 02.06.2020:
        I added t to the output to return the effective timescale.
        This is not necessary for the model and does not change any computation.
        The purpose is to read out the effective timescales during activation.
        """
        return [y, t], [y, z]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_taus(self):
        return {"tau_bias": self.get_tau_bias(),
                "tau_kernel": self.get_tau_kernel(),
                "tau_recurrent": self.get_tau_recurrent()}

    def get_tau_bias(self):
        return K.exp(self.tau_bias + self.log_taus) + _ALMOST_ONE if self.built else None

    def get_tau_kernel(self):
        return K.exp(self.tau_kernel) if self.built else None

    def get_tau_recurrent(self):
        tau_recurrent_flat = array_ops.constant(
            0., dtype=self.dtype, shape=[self.units, self.units], name="taus")
        if self.connectivity == 'partitioned':
            #TODO
            pass
        elif self.connectivity == 'clocked':
            #TODO
            pass
        if self.connectivity == 'adjacent':
            #TODO
            pass
        else:  # == 'dense'
            tau_recurrent_flat = K.exp(
                self.tau_recurrent[0]) if self.built else array_ops.constant(
                0., dtype=self.dtype, shape=[self.units, self.units],
                name="taus")

        return tau_recurrent_flat

    def get_config(self):
        config = {
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'tau_kernel_initializer':
                initializers.serialize(self.tau_kernel_initializer),
            'tau_recurrent_initializer':
                initializers.serialize(self.tau_recurrent_initializer),
            'tau_bias_initializer':
                initializers.serialize(self.tau_bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'tau_kernel_regularizer':
                regularizers.serialize(self.tau_kernel_regularizer),
            'tau_recurrent_regularizer':
                regularizers.serialize(self.tau_recurrent_regularizer),
            'tau_bias_regularizer':
                regularizers.serialize(self.tau_bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'tau_kernel_constraint':
                constraints.serialize(self.tau_kernel_constraint),
            'tau_recurrent_constraint':
                constraints.serialize(self.tau_recurrent_constraint),
            'tau_bias_constraint':
                constraints.serialize(self.tau_bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(GACTRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.ACTRNN')
class ACTRNN(RNN):
    """Adaptive Continuous Time RNN that can have several modules
       where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        w_tau_initializer: Initializer for the w_tau vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        w_tau_regularizer: Regularizer function applied to the w_tau vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        w_tau_constraint: Constraint function applied to the w_tau vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 w_tau_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 w_tau_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 w_tau_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `ACTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = ACTRNNCell(
            units_vec,
            modules=modules,
            tau_vec=tau_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            w_tau_initializer=w_tau_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            w_tau_regularizer=w_tau_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            w_tau_constraint=w_tau_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(ACTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(ACTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def w_tau_initializer(self):
        return self.cell.w_tau_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def w_tau_regularizer(self):
        return self.cell.w_tau_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def w_tau_constraint(self):
        return self.cell.w_tau_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_taus(self):
        return self.cell.get_taus()

    def get_config(self):
        config = {
            'units':
                self.units,
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'w_tau_initializer':
                initializers.serialize(self.w_tau_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'w_tau_regularizer':
                regularizers.serialize(self.w_tau_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'w_tau_constraint':
                constraints.serialize(self.w_tau_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(ACTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


@tf_export('keras.layers.GCTRNN')
class GCTRNN(RNN):
    """Timescale Gated Continuous Time RNN that can have several modules
       where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        tau_kernel_initializer: Initializer for the tau_kernel vector.
        tau_recurrent_initializer: Initializer for the tau_recurrent vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        tau_kernel_regularizer: Regularizer function applied to the tau_kernel vector.
        tau_recurrent_regularizer: Regularizer function applied to the tau_recurrent vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        tau_kernel_constraint: Constraint function applied to the tau_kernel vector.
        tau_recurrent_constraint: Constraint function applied to the tau_recurrent vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 tau_kernel_initializer='zeros',
                 tau_recurrent_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 tau_kernel_regularizer=None,
                 tau_recurrent_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 tau_kernel_constraint=None,
                 tau_recurrent_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `GCTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = GCTRNNCell(
            units_vec,
            modules=modules,
            tau_vec=tau_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            tau_kernel_initializer=tau_kernel_initializer,
            tau_recurrent_initializer=tau_recurrent_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            tau_kernel_regularizer=tau_kernel_regularizer,
            tau_recurrent_regularizer=tau_recurrent_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            tau_kernel_constraint=tau_kernel_constraint,
            tau_recurrent_constraint=tau_recurrent_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(GCTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(GCTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def tau_kernel_initializer(self):
        return self.cell.tau_kernel_initializer

    @property
    def tau_recurrent_initializer(self):
        return self.cell.tau_recurrent_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def tau_kernel_regularizer(self):
        return self.cell.tau_kernel_regularizer

    @property
    def tau_recurrent_regularizer(self):
        return self.cell.tau_recurrent_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def tau_kernel_constraint(self):
        return self.cell.tau_kernel_constraint

    @property
    def tau_recurrent_constraint(self):
        return self.cell.tau_recurrent_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_taus(self):
        return self.cell.get_taus()

    def get_config(self):
        config = {
            'units':
                self.units,
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'tau_kernel_initializer':
                initializers.serialize(self.tau_kernel_initializer),
            'tau_recurrent_initializer':
                initializers.serialize(self.tau_recurrent_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'tau_kernels_regularizer':
                regularizers.serialize(self.tau_kernel_regularizer),
            'tau_recurrent_regularizer':
                regularizers.serialize(self.tau_recurrent_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'tau_kernel_constraint':
                constraints.serialize(self.tau_kernel_constraint),
            'tau_recurrents_constraint':
                constraints.serialize(self.tau_recurrent_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(GCTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


@tf_export('keras.layers.GACTRNN')
class GACTRNN(RNN):
    """Timescale Gated Continuous Time RNN that can have several modules
       where the output is to be fed back to input.

    Arguments:
        units_vec: Positive integer or vector of positive integer,
            dimensionality of the output space.
        modules: Positive integer, number of modules.
            The dimensionality of the outputspace is a concatenation of
            all modules k with the respective units_vec[k] size.
            Default: depends on size of units_vec or 1 in case of units_vec
            being a scalar.
        tau_vec: Positive float or vector of positive float >= 1, timescale.
            Unit-dependent time constant of leakage.
            Default: 1.0
        connectivity: Connection scheme in case of more than one modules
            Default: `dense`
            Other options are `partitioned`, `clocked`, and `adjacent`
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
        bias_initializer: Initializer for the bias vector.
        tau_bias_initializer: Initializer for the tau_bias vector.
        tau_kernel_initializer: Initializer for the tau_kernel vector.
        tau_recurrent_initializer: Initializer for the tau_recurrent vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        tau_bias_regularizer: Regularizer function applied to the tau_bias vector.
        tau_kernel_regularizer: Regularizer function applied to the tau_kernel vector.
        tau_recurrent_regularizer: Regularizer function applied to the tau_recurrent vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        tau_bias_constraint: Constraint function applied to the tau_bias vector.
        tau_kernel_constraint: Constraint function applied to the tau_kernel vector.
        tau_recurrent_constraint: Constraint function applied to the tau_recurrent vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 units_vec,
                 modules=None,
                 tau_vec=1.,
                 connectivity='dense',
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 tau_bias_initializer='zeros',
                 tau_kernel_initializer='zeros',
                 tau_recurrent_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 tau_bias_regularizer=None,
                 tau_kernel_regularizer=None,
                 tau_recurrent_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 tau_bias_constraint=None,
                 tau_kernel_constraint=None,
                 tau_recurrent_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            logging.warning('The `implementation` argument '
                            'in `GACTRNN` has been deprecated. '
                            'Please remove it from your layer call.')
        cell = GACTRNNCell(
            units_vec,
            modules=modules,
            tau_vec=tau_vec,
            connectivity=connectivity,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            tau_bias_initializer=tau_bias_initializer,
            tau_kernel_initializer=tau_kernel_initializer,
            tau_recurrent_initializer=tau_recurrent_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            tau_bias_regularizer=tau_bias_regularizer,
            tau_kernel_regularizer=tau_kernel_regularizer,
            tau_recurrent_regularizer=tau_recurrent_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            tau_bias_constraint=tau_bias_constraint,
            tau_kernel_constraint=tau_kernel_constraint,
            tau_recurrent_constraint=tau_recurrent_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout)
        super(GACTRNN, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None

        return super(GACTRNN, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def units_vec(self):
        return self.cell.units_vec

    @property
    def modules(self):
        return self.cell.modules

    @property
    def tau_vec(self):
        return self.cell.tau_vec

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def tau_bias_initializer(self):
        return self.cell.tau_bias_initializer

    @property
    def tau_kernel_initializer(self):
        return self.cell.tau_kernel_initializer

    @property
    def tau_recurrent_initializer(self):
        return self.cell.tau_recurrent_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def tau_bias_regularizer(self):
        return self.cell.tau_bias_regularizer

    @property
    def tau_kernel_regularizer(self):
        return self.cell.tau_kernel_regularizer

    @property
    def tau_recurrent_regularizer(self):
        return self.cell.tau_recurrent_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def tau_bias_constraint(self):
        return self.cell.tau_bias_constraint

    @property
    def tau_kernel_constraint(self):
        return self.cell.tau_kernel_constraint

    @property
    def tau_recurrent_constraint(self):
        return self.cell.tau_recurrent_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_taus(self):
        return self.cell.get_taus()

    def get_config(self):
        config = {
            'units':
                self.units,
            'units_vec':
                self.units_vec,
            'modules':
                self.modules,
            'tau_vec':
                self.tau_vec,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'tau_bias_initializer':
                initializers.serialize(self.tau_bias_initializer),
            'tau_kernel_initializer':
                initializers.serialize(self.tau_kernel_initializer),
            'tau_recurrent_initializer':
                initializers.serialize(self.tau_recurrent_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'tau_bias_regularizer':
                regularizers.serialize(self.tau_bias_regularizer),
            'tau_kernels_regularizer':
                regularizers.serialize(self.tau_kernel_regularizer),
            'tau_recurrent_regularizer':
                regularizers.serialize(self.tau_recurrent_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'tau_bias_constraint':
                constraints.serialize(self.tau_bias_constraint),
            'tau_kernel_constraint':
                constraints.serialize(self.tau_kernel_constraint),
            'tau_recurrent_constraint':
                constraints.serialize(self.tau_recurrent_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(GACTRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)
