"""
Aaron Tuor: Oct. 2016
DNN autoencoder implementation for unsupervised training of
one pass streaming data with meta data included in stream.

"""
import tensorflow as tf
import numpy as np
import argparse
import sys
import math

OPTIMIZERS = {'grad': tf.train.GradientDescentOptimizer, 'adam': tf.train.AdamOptimizer}

def fan_scale(initrange, activation, tensor_in):
    if activation == tf.nn.relu:
        initrange *= np.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
    else:
        initrange *= (1.0/np.sqrt(float(tensor_in.get_shape().as_list()[1])))
    return initrange

def weights(distribution, shape, dtype=tf.float32, initrange=1e-5,
            seed=None, l2=0.0, name='weights'):
    """
    Wrapper parameterizing common constructions of tf.Variables.

    :param distribution: A string identifying distribution 'tnorm' for truncated normal, 'rnorm' for random normal, 'constant' for constant, 'uniform' for uniform.
    :param shape: Shape of weight tensor.
    :param dtype: dtype for weights
    :param initrange: Scales standard normal and trunctated normal, value of constant dist., and range of uniform dist. [-initrange, initrange].
    :param seed: For reproducible results.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param name: For variable scope.
    :return: A tf.Variable.
    """

    if distribution == 'norm':
        wghts = tf.Variable(initrange*tf.random_normal(shape, 0, 1, dtype, seed))
    elif distribution == 'tnorm':
        wghts = tf.Variable(initrange*tf.truncated_normal(shape, 0, 1, dtype, seed))
    elif distribution == 'uniform':
        wghts = tf.Variable(tf.random_uniform(shape, -initrange, initrange, dtype, seed))
    elif distribution == 'constant':
        wghts = tf.Variable(tf.constant(initrange, dtype=dtype, shape=shape))
    else:
        raise ValueError("Argument 'distribution takes values 'norm', 'tnorm', 'uniform', 'constant', "
                          "Received %s" % distribution)
    if l2 != 0.0:
        tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(wghts), l2, name=name + 'weight_loss'))
    return wghts


def batch_normalize(tensor_in, epsilon=1e-5, decay=0.999, name="batch_norm"):
    """
    Batch Normalization:
    `Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift`_

    An exponential moving average of means and variances in calculated to estimate sample mean
    and sample variance for evaluations. For testing pair placeholder is_training
    with [0] in feed_dict. For training pair placeholder is_training
    with [1] in feed_dict. Example:

    Let **train = 1** for training and **train = 0** for evaluation

    .. code-block:: python
        bn_deciders = {decider:[train] for decider in tf.get_collection('bn_deciders')}
        feed_dict.update(bn_deciders)

    :param tensor_in: input Tensor_
    :param epsilon: A float number to avoid being divided by 0.
    :param name: For variable_scope_
    :return: Tensor with variance bounded by a unit and mean of zero according to the batch.
    """

    is_training = tf.placeholder(tf.int32, shape=[None]) # [1] or [0], Using a placeholder to decide which
                                          # statistics to use for normalization allows
                                          # either the running stats or the batch stats to
                                          # be used without rebuilding the graph.
    tf.add_to_collection('bn_deciders', is_training)

    pop_mean = tf.Variable(tf.zeros([tensor_in.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([tensor_in.get_shape()[-1]]), trainable=False)

    # calculate batch mean/var and running mean/var
    batch_mean, batch_variance = tf.nn.moments(tensor_in, [0], name=name)

    # The running mean/variance is updated when is_training == 1.
    running_mean = tf.assign(pop_mean,
                             pop_mean * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                             batch_mean * (1.0 - decay) * tf.to_float(is_training))
    running_var = tf.assign(pop_var,
                            pop_var * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                            batch_variance * (1.0 - decay) * tf.to_float(is_training))

    # Choose statistic
    mean = tf.nn.embedding_lookup(tf.pack([running_mean, batch_mean]), is_training)
    variance = tf.nn.embedding_lookup(tf.pack([running_var, batch_variance]), is_training)

    shape = tensor_in.get_shape().as_list()
    gamma = weights('constant', [shape[1]], initrange=0.0, name=name + '_gamma')
    beta = weights('constant', [shape[1]], initrange=1.0, name=name + '_beta')

    # Batch Norm Transform
    inv = tf.rsqrt(epsilon + variance, name=name)
    tensor_in = beta * (tensor_in - mean) * inv + gamma

    return tensor_in

def dropout(tensor_in, prob, name='Dropout'):
    """
    Adds dropout node.
    :param tensor_in: Input tensor_.
    :param prob: The percent of units to keep.
    :param name: A name for the tensor.
    :return: Tensor_ of the same shape of *tensor_in*.
    """
    if isinstance(prob, float):
        keep_prob = tf.placeholder(tf.float32)
        tf.add_to_collection('dropout_prob', (keep_prob, prob))
    return tf.nn.dropout(tensor_in, keep_prob)

def dnn(x, layers=[100, 408], act=tf.nn.relu, scale_range=1.0, bn=False, keep_prob=None, name='nnet'):
    """
    An arbitrarily deep neural network.

    :param x: Input to the network.
    :param layers: List of sizes of network layers.
    :param act: Activation function to produce hidden layers of neural network.
    :param scale_range: Scaling factor for initial range of weights (Set to 1/sqrt(fan_in).
    :param bn: Whether to use batch normalization.
    :param keep_prob: The percent of nodes to keep in dropout layers.
    :param name: For naming and variable scope.

    :return: (tf.Tensor) Output of neural net. This will be just following a linear transform,
             so that final activation has not been applied.
    """


    """

    :param name: An identifier for retrieving tensors made by dnn.
    """

    for ind, hidden_size in enumerate(layers):
        with tf.variable_scope('layer_%s' % ind):

            fan_in = x.get_shape().as_list()[1]
            W = tf.Variable(fan_scale(scale_range, act, x)*tf.truncated_normal([fan_in, hidden_size],
                                                     mean=0.0, stddev=1.0,
                                                     dtype=tf.float32, seed=None, name='W'))
            tf.add_to_collection(name + '_weights', W)
            b = tf.Variable(tf.zeros([hidden_size]))
            tf.add_to_collection(name + '_bias', b)
            x = tf.matmul(x,W) + b
            if bn:
                x = batch_normalize(x, name=name + '_bn')
            if ind != len(layers) - 1:
                x = act(x, name='h' + str(ind)) # The hidden layer
                tf.add_to_collection(name + '_activation', x)
                if keep_prob:
                    x = dropout(x, keep_prob, name=name + '_dropouts')
    return x

class Loop():

        def __init__(self, badlimit=20):
            """
            :param badlimit: limit of badcount for early stopping
            """
            self.badlimit = badlimit
            self.badcount = 0
            self.data = data
            self.current_loss = sys.float_info.max

        def __call__(self, mat, loss):
            """
            Returns a boolean for customizable stopping criterion. For first loop set loss to sys.float_info.max.
            :param mat: Current batch of features for training.
            :param loss: Current loss during training.
            :return: boolean, True when mat is not None and self.badcount < self.badlimit and loss != inf, nan.
            """
            if mat is None:
                sys.stderr.write('Done Training. End of data stream.')
                cond = False
            elif math.isnan(loss) or math.isinf(loss):
                sys.stderr.write('Exiting due divergence: %s\n\n' % loss)
                cond = False
            elif loss > self.current_loss:
                self.badcount += 1
                if self.badcount >= self.badlimit:
                    sys.stderr.write('Exiting. Exceeded max bad count.')
                    cond = False
                else:
                    cond = True
            else:
                self.badcount = 0
                cond = True
            self.current_loss = loss
            return cond

class SimpleModel():
    """
    A class for gradient descent training arbitrary models.

    :param loss: Loss Tensor for gradient descent optimization (should evaluate to real number).
    :param pointloss: A tensor of individual losses for each datapoint in minibatch.
    :param ph_dict: A dictionary with string keys and tensorflow placeholder values.
    :param learnrate: step_size for gradient descent.
    :param resultfile: Where to print loss during training.
    :param debug: Whether to print debugging info.
    :param badlimit: Number of times to not improve during training before quitting.
    """

    def __init__(self, loss, pointloss, contrib, ph_dict,
                 learnrate=0.01, resultfile=None,
                 opt='adam', debug=False,
                 verbose=0):
        self.loss = loss
        self.pointloss = pointloss
        self.contrib = contrib
        self.ph_dict = ph_dict
        self.out = open(resultfile, 'w')
        self.debug = debug
        self.verbose = verbose
        self.train_step = OPTIMIZERS[opt](learnrate).minimize(loss)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)


    def train(self, train_data, loop):
        """
        :param train_data: A Batcher object that delivers batches of train data.
        :param loop: A function or callable object that returns a boolean depending on current data and current loss.
        """
        self.out.write('day user red loss\n')
        mat = train_data.next_batch()
        loss = sys.float_info.max
        while loop(mat, loss): #mat is not None and self.badcount < self.badlimit and loss != inf, nan:
            datadict = {'features': mat[:, 3:], 'red': mat[:,2], 'user': mat[:,1], 'day': mat[:,0]}
            _, loss, pointloss, contrib = self.sess.run((self.train_step, self.loss, self.pointloss, self.contrib),
                                    feed_dict=self.get_feed_dict(datadict, self.ph_dict))
            if self.verbose == 1:
                self.print_all_contrib(datadict, loss, pointloss, contrib, train_data.index)
            elif self.verbose == 0:
                self.print_results(datadict, loss, pointloss, train_data.index)

            mat = train_data.next_batch()
        self.out.close()

    def print_results(self, datadict, loss, pointloss, index):
        for d, u, t, l, in zip(datadict['day'].tolist(), datadict['user'].tolist(),
                               datadict['red'].tolist(), pointloss.flatten().tolist()):
            self.out.write('%s %s %s %s\n' % (d, u, t, l))
        print('index: %s loss: %.4f' % (index, loss))

    def print_all_contrib(self, datadict, loss, pointloss, contrib, index):
        for time, user, red, loss, contributor in zip(datadict['day'].tolist(),
                                                      datadict['user'].tolist(),
                                                      datadict['red'].tolist(),
                                                      pointloss.flatten().tolist(),
                                                      contrib.tolist()):
            self.out.write('%s %s %s %s ' % (time, user, red, loss))
            self.out.write(str(contributor).strip('[').strip(']').replace(',', ''))
            self.out.write('\n')
        print('index: %s loss: %.4f' % (index, loss))


    def get_feed_dict(self, datadict, ph_dict, train=1):

        """
        :param datadict: A dictionary with keys matching keys in ph_dict, and values are numpy matrices.
        :param ph_dict: A dictionary where the keys match keys in datadict and values are placeholder tensors.
        :return: A feed dictionary with keys of placeholder tensors and values of numpy matrices.
        """
        fd = {ph_dict[key]:datadict[key] for key in ph_dict}
        dropouts = tf.get_collection('dropout_prob')
        bn_deciders = tf.get_collection('bn_deciders')
        if dropouts:
            for prob in dropouts:
                if train == 1:
                    fd[prob[0]] = prob[1]
                else:
                    fd[prob[0]] = 1.0
        if bn_deciders:
            fd.update({decider:[train] for decider in bn_deciders})
        if self.debug:
            for desc in ph_dict:
                print('%s\n\tph: %s\t%s\tdt: %s\t%s' % (desc,
                                                        ph_dict[desc].get_shape().as_list(),
                                                        ph_dict[desc].dtype,
                                                        datadict[desc].shape,
                                                        datadict[desc].dtype))
                print(fd.keys())
        return fd

def print_datadict(datadict):
    for k, v in datadict.iteritems():
        print(k + str(v.shape))

class OnlineBatcher():
    """
    For batching data too large to fit into memory. Written for one pass on data!!!
    Option to normalize input using running mean and variance.
    """

    def __init__(self, datafile, batch_size, normalize=False, alpha=0.9, varinit=None):
        """
        :param datafile: File to read lines from.
        :param batch_size: Mini-batch size.
        :param normalize: Whether to normalize the batch by centering on
                          estimated mean and scaling by estimated variance.
        :param alpha: Parameter for exponential moving average.
        """
        self.f = open(datafile, 'r')
        self.batch_size = batch_size
        self.mu = 0.0
        self.variance = 0.0
        self.norm = normalize
        self.index = 0
        self.varinit = varinit


    def next_batch(self):
        """
        :return: until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        matlist = []
        l = self.f.readline()
        if l == '':
            return None
        rowtext = np.array([float(k) for k in l.strip().split(',')])
        matlist.append(rowtext)
        for i in range(self.batch_size - 1):
            l = self.f.readline()
            if l == '':
                break
            rowtext = np.array([float(k) for k in l.strip().split(',')])
            matlist.append(rowtext)
        data = np.array(matlist)
        self.index += self.batch_size
        return data



def mvn(truth, h, scale_range=1.0, variance_floor=0.1):
    """
    Brian Hutchinson's calculations for a diagonal covariance loss.

    :param truth: (tf.Tensor) The truth for this minibatch.
    :param h:(tf.Tensor) The output of dnn.
             (Here the output of dnn , h, is assumed to be the same dimension as truth)
    :param variance_floor: (float, positive) To ensure model doesn't find trivial optimization.
    :return: (tf.Tensor) A vector of losses for each pair of vectors in truth, pair.
    """
    fan_in = h.get_shape().as_list()[1]
    U = tf.Variable(fan_scale(scale_range, tf.tanh, h)*tf.truncated_normal([fan_in, 2*fan_in],
                                                     mean=0.0, stddev=1.0,
                                                     dtype=tf.float32, seed=None, name='W'))
    b = tf.Variable(tf.zeros([2*fan_in]))
    y = tf.matmul(h, U) + b
    mu, var = tf.split(1, 2, y) # split y into two even sized matrices, each with half the columns
    var = tf.maximum(tf.exp(var), #  make the variance non-negative
                     tf.constant(variance_floor, shape=[fan_in], dtype=tf.float32))
    logdet = tf.reduce_sum(tf.log(var), 1, keep_dims=True) # MB x 1
    loss_columns = tf.concat(1, [tf.square(truth-mu)/var, logdet]) # is MB x D + 1
    return tf.reduce_sum(loss_columns, reduction_indices=[1]), loss_columns


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Dnn auto-encoder for online unsupervised training.")
    parser.add_argument('datafile',
                        type=str,
                        help='The csv data file for our unsupervised training.'+\
                             'fields: day, user, redcount, [count1, count2, ...., count408]')
    parser.add_argument('results', type=str, help='The folder to print results to.')
    parser.add_argument('-learnrate', type=float, default=0.001,
                        help='Step size for gradient descent.')
    parser.add_argument("-layers", nargs='+',
                        type=int, default=[100, 100, 408], help="A list of hidden layer sizes.")
    parser.add_argument('-mb', type=int, default=256, help='The mini batch size for stochastic gradient descent.')
    parser.add_argument('-act', type=str, default='tanh', help='May be "tanh" or "relu"')
    parser.add_argument('-bn', action='store_true', help='Use this flag if using batch normalization.')
    parser.add_argument('-keep_prob', type=float, default=None,
                        help='Percent of nodes to keep for dropout layers.')
    parser.add_argument('-debug', action='store_true',
                        help='Use this flag to print feed dictionary contents and dimensions.')
    parser.add_argument('-dist', type=str, default='diag',
                        help='"diag" or "ident". Describes whether to model multivariate guassian with identity, '
                             'or abitrary diagonal covariance matrix.')
    parser.add_argument('-variance_floor', type=float, default=0.1,
                        help='For diagonal covariance matrix loss calculation.')
    parser.add_argument('-scalerange', type=float, default=1.0, help='Extra scaling on top of fan_in scaling.')
    parser.add_argument('-opt', type=str, default='adam', help='Optimization strategy. {"grad", "adam"}')
    parser.add_argument('-maxbadcount', type=str, default=20, help='Threshold for early stopping.')
    parser.add_argument('-norm', action='store_true', 
                        help='Whether to normalized data to have zero mean and unit variance with respect to an exponential moving average') 
    parser.add_argument('-alpha', type=float, default='0.9', 
                        help='Decay rate for exponential moving average.')
    parser.add_argument('-verbose', type=int, default=0, help='1 to print full loss contributors')
    args = parser.parse_args()

    if args.act == 'tanh':
        activation = tf.tanh
    elif args.act == 'relu':
        activation = tf.nn.relu
    else:
        raise ValueError('Activation must be "relu", or "tanh"')

    data = OnlineBatcher(args.datafile, args.mb, normalize=args.norm, alpha=args.alpha)

    x = tf.placeholder(tf.float32, shape=[None, 408])
    h = dnn(x, layers=args.layers, act=activation, keep_prob=args.keep_prob,
            scale_range=args.scalerange, bn=args.bn)
    if args.dist == 'ident':
        contrib = tf.square(x-h)
        pointloss = tf.reduce_sum(contrib, reduction_indices=[1])
    elif args.dist == 'diag':
        pointloss, contrib = mvn(x, h, scale_range=args.scalerange, variance_floor=args.variance_floor)
    else:
        raise ValueError('Argument dist must be "ident" or "diag".')
    loss = tf.reduce_mean(pointloss)

    placeholderdict = {'features': x}
    model = SimpleModel(loss, pointloss, contrib, placeholderdict, learnrate=args.learnrate,
                        resultfile=args.results, opt=args.opt, debug=args.debug, verbose=args.verbose)
    loop = Loop(badlimit=args.maxbadcount)
    model.train(data, loop)
