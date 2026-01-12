import numpy as np
import torch
import sys
import os
import time

def create_temporaly_convolutional_model(max_input_window_size, num_segments, num_syn_types, num_DVT_outputs,
                                                                                             filter_sizes_per_layer,
                                                                                             num_filters_per_layer,
                                                                                             activation_function_per_layer,
                                                                                             l2_regularization_per_layer,
                                                                                             strides_per_layer,
                                                                                             dilation_rates_per_layer,
                                                                                             initializer_per_layer):
    
    # define input and flatten it
    binary_input_mat = Input(shape=(max_input_window_size, num_segments * num_syn_types), name='input_layer')

    for k in range(len(filter_sizes_per_layer)):
        num_filters   = num_filters_per_layer[k]
        filter_size   = filter_sizes_per_layer[k]
        activation    = activation_function_per_layer[k]
        l2_reg        = l2_regularization_per_layer[k]
        stride        = strides_per_layer[k]
        dilation_rate = dilation_rates_per_layer[k]
        initializer   = initializer_per_layer[k]
        
        if activation == 'lrelu':
            leaky_relu_slope = 0.25
            activation = lambda x: LeakyReLU(alpha=leaky_relu_slope)(x)
            print('leaky relu slope = %.4f' %(leaky_relu_slope))
            
        if not isinstance(initializer, basestring):
            initializer = initializers.TruncatedNormal(stddev=initializer)
        
        if k == 0:
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(binary_input_mat)
        else:
            x = Conv1D(num_filters, filter_size, activation=activation, kernel_initializer=initializer, kernel_regularizer=l2(l2_reg),
                       strides=stride, dilation_rate=dilation_rate, padding='causal', name='layer_%d' %(k + 1))(x)
        x = BatchNormalization(name='layer_%d_BN' %(k + 1))(x)
        
    output_spike_init_weights = initializers.TruncatedNormal(stddev=0.001)
    output_spike_init_bias    = initializers.Constant(value=-2.0)
    output_soma_init  = initializers.TruncatedNormal(stddev=0.03)
    output_dend_init  = initializers.TruncatedNormal(stddev=0.05)

    output_spike_predictions = Conv1D(1, 1, activation='sigmoid', kernel_initializer=output_spike_init_weights, bias_initializer=output_spike_init_bias,
                                                                  kernel_regularizer=l2(1e-8), padding='causal', name='spikes')(x)
    output_soma_voltage_pred = Conv1D(1, 1, activation='linear' , kernel_initializer=output_soma_init, kernel_regularizer=l2(1e-8), padding='causal', name='somatic')(x)
    output_dend_voltage_pred = Conv1D(num_DVT_outputs, 1, activation='linear' , kernel_initializer=output_dend_init, kernel_regularizer=l2(1e-8), padding='causal', name='dendritic')(x)

    temporaly_convolutional_network_model = Model(inputs=binary_input_mat, outputs=
                                                  [output_spike_predictions, output_soma_voltage_pred, output_dend_voltage_pred])

    optimizer_to_use = Nadam(lr=0.0001)
    temporaly_convolutional_network_model.compile(optimizer=optimizer_to_use,
                                                  loss=['binary_crossentropy','mse','mse'],
                                                  loss_weights=[1.0, 0.006, 0.002])
    temporaly_convolutional_network_model.summary()
    
    return temporaly_convolutional_network_model



