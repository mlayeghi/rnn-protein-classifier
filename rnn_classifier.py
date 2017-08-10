# Imports
import argparse
import numpy as np
import os
import tensorflow as tf
from utils import load_data, next_batch
from model import rnn_model

def main():
    '''
    Main function: set the parameters & call training.
    Training parameters can be adjusted here.
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-input_file', type=str, default='data/proteins40bp.csv',
                        help='The path to the input fasta file.')
    parser.add_argument('-rnn_type', type=str, default='gru',
                        help='The type of RNN network: basiclstm, lstm, gru.')
    parser.add_argument('-batch_size', type=int, default=50,
                        help='Minibatch size.')
    parser.add_argument('-n_layers', type=int, default=3,
                        help='Number of layers in the RNN.')
    parser.add_argument('-hidden_dim', type=int, default=1000,
                        help='Size of RNN hidden state; number of neuron in each layer.')
    parser.add_argument('-n_inputs', type=int, default=40,
                        help='Protein sequence length.')
    parser.add_argument('-n_classes', type=int, default=2,
                        help='Number of classes; protein families/clusters.')
    parser.add_argument('-in_keep_prob', type=float, default=1,
                        help='Dropout probability.')
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('-n_epochs', type=int, default=20,
                        help='Number of training epochs.')

    args = parser.parse_args()

    # Load train & test data & their labels
    x_train, x_test, y_train, y_test = load_data(args.input_file)

    # Spit out details about data
    classes = np.sort(np.unique(y_train))
    print("\n=================================\nData details:")
    print("- Training-set:\t\t{}".format(len(y_train)))
    print("- Test-set:\t\t{}".format(len(y_test)))
    print("- Features:\t\t{}".format(args.n_inputs))
    print("- Classes:\t\t{}".format(classes))
    print("=================================\n\n")

    # Call the train function
    train(x_train, x_test, y_train, y_test, args)

def train(x_train, x_test, y_train, y_test, args):
    '''
    Train the RNN network.
    '''
    # We need to reshape input training data into correct shape:
    # number of examples, number of input, dimension of each input.
    # It will be done, below in the graph. But, we also need a TF placeholder.
    # In our case, number of examples would be the batch_size, which
    # is an adjustable parameter, so we assign it to None to signify that.
    # Our data is 1D, so 1 was assigned to the third shape dimension.
    # The 2nd dimension is the length of protein sequences.
    x_input = tf.placeholder(tf.float32, [None, args.n_inputs, 1])
    # For target, the size relies again on the batch_size.
    target = tf.placeholder(tf.int32, [None])
    # Similarly, the test data must be resaped too.
    n_test_examples = len(y_test)
    x_test = x_test.reshape((n_test_examples, args.n_inputs, 1))

    # We call the model and capture the last state in the RNN network.
    final_state = rnn_model(x_input, args)

    # Calculate probabilities
    with tf.name_scope('Logits'):
        logits = tf.layers.dense(final_state, args.n_classes)

    # Loss function
    with tf.name_scope('Loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy)

    # Optimization
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        training_op = optimizer.minimize(loss)

    # Accuracy
    with tf.name_scope('Accuracy'):
        # Whether the targets are in the top K predictions
        correct = tf.nn.in_top_k(logits, target, k=1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # To initialize all TF variables
    init = tf.global_variables_initializer()

    # Early stopping: if no improvement in test accuracy seen within 10 epochs
    # We store the best accuracy and update it if new accuracy after each epoch
    # was higher. We'll stop optimization if no improvement seen in 10 epochs.
    best_accuracy = 0.0
    last_improvement = 0
    check_improvement = 10
    # Number of examples in the training set
    n_train_examples = len(y_train)

    # The graph!
    with tf.Session() as sess:
        init.run()
        for epoch in range(args.n_epochs):
            for i in range(n_train_examples // args.batch_size):
                # Get the trainig batch
                X_batch, y_batch = next_batch(x_train, y_train, args.batch_size)
                # Reshape training batch, as explained above, in the shape of the placeholder
                X_batch = X_batch.reshape((args.batch_size, args.n_inputs, 1))
                sess.run(training_op, feed_dict={x_input: X_batch, target: y_batch})

            # Obtain accuracy on training & test data
            acc_train = accuracy.eval(feed_dict={x_input: X_batch, target: y_batch})
            acc_test = accuracy.eval(feed_dict={x_input: x_test, target: y_test})

            # Update the best accuracy & output accuracy results
            if acc_test > best_accuracy:
                best_accuracy = acc_test
                last_improvement = epoch
                print("Epoch", epoch, ":\tTrain accuracy =", acc_train, ";\tTest accuracy =", acc_test)

            # Early stopping in case of no improvement.
            if epoch - last_improvement > check_improvement:
                print("\n#### No improvement seen in ", check_improvement, " epochs ==> early stopping.\n")
                break

if __name__ == '__main__':
    main()
