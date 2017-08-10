import numpy as np
import tensorflow as tf

def rnn_model(data, args):
    '''
    The RNN model in TensorFlow.
    '''
    # Using He intializer to generate tensors with unit variance for weights & biases.
    # Users can write their own functions to create weight and biase tensors, obviously,
    # but they can also use one of the several initializers available in TensorFlow. He
    # and Xavier are two of the most renowned ones.
    with tf.variable_scope("rnn",
                           initializer=\
                           tf.contrib.layers.variance_scaling_initializer()):
        if args.rnn_type == 'basiclstm':
            # Use list comprehension to build RNN layers.
            basic_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=args.hidden_dim)
                           for layer in range(args.n_layers)]
            # To apply dropout, we need to use a wrapper layer.
            drop_cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=args.in_keep_prob)
                          for cell in basic_cells]
            # Then we need to stack all the layers into a multilayer cell: MultiRNNCell
            multi_cells = tf.contrib.rnn.MultiRNNCell(drop_cells)
            # Apply dynamic unrolling method on the multilayer cell
            _, states = tf.nn.dynamic_rnn(multi_cells, data, dtype=tf.float32)
            # Get the final state: Since the output of a recurrent neuron at any
            # time step is a function of all the inputs/states from previous time
            #  steps, the final or last state carries all the memory.
            final_state = states[-1][1]
        elif args.rnn_type == 'lstm':
            lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=args.hidden_dim)
                          for layer in range(args.n_layers)]
            drop_cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=args.in_keep_prob)
                          for cell in lstm_cells]
            multi_cells = tf.contrib.rnn.MultiRNNCell(drop_cells)
            _, states = tf.nn.dynamic_rnn(multi_cells, data, dtype=tf.float32)

            final_state = states[-1][1]
        elif args.rnn_type == 'gru':
            gru_cells = [tf.contrib.rnn.GRUCell(num_units=args.hidden_dim)
                         for layer in range(args.n_layers)]
            drop_cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=args.in_keep_prob)
                          for cell in gru_cells]
            multi_cells = tf.contrib.rnn.MultiRNNCell(drop_cells)
            _, states = tf.nn.dynamic_rnn(multi_cells, data, dtype=tf.float32)

            final_state = states[-1]
        else:
            raise Exception("\ns{} is not a relevant RNN type!".format(args.rnn_type))

    return final_state
