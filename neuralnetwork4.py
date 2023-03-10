"""neuralnetwork4.py
~~~~~~~~~~~~~~

An improved version of network2.py

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np

# activation functions
class SigmoidActivation:
    @staticmethod
    def fn(z):
        #sigmoid
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def derivative(z):
        #derivative
        return SigmoidActivation.fn(z) * (1 - SigmoidActivation.fn(z))

class TanhActivation:
    @staticmethod
    def fn(z):
        #sigmoid
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

    @staticmethod
    def derivative(z):
        #derivative
        return 1-TanhActivation.fn(z)**2

class LinearActivation:
    @staticmethod
    def fn(z):
        #sigmoid
        return z

    @staticmethod
    def derivative(z):
        #derivative
        return np.full(z.shape, 1.0)

class RelUActivation:
    @staticmethod
    def fn(z):
        # sigmoid
        return np.max(z, 0.0)

    @staticmethod
    def derivative(z):
        # derivative
        return np.full(z.shape, 1.0)
        # incorrect


#### cost functions

class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta_z(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * SigmoidActivation.derivative(z)

    @staticmethod
    def delta_a(z, a, y):
        """array of partial derivatives cost function by
         last layer activations array"""
        return a - y


class QuatroCost(object):
    # fourth degree
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return (np.linalg.norm(a-y)**4)/24

    @staticmethod
    def delta_z(z, a, y):
        """Return the error delta from the output layer."""
        return ((a-y)**3) * SigmoidActivation.derivative(z)

    @staticmethod
    def delta_z(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y)**3


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta_z(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

    @staticmethod
    def delta_a(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return np.nan_to_num(-y/a + (1-y)/(1-a))


class Network(object):

    def __init__(self, layers, cost=CrossEntropyCost):
        self.num_layers = len(layers) + 1
        self.layers = layers
        self.cost = cost

    def get_input(self, expected_output):
        y = expected_output
        random_input = (np.random.randn(self.sizes[0], 1)) % 1
        x = random_input
        for not_matter_counter in range (1000):
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = SigmoidActivation.fn(z)
                activations.append(activation)
            # backward pass
            delta = (self.cost).delta(zs[-1], activations[-1], y)
            #print(delta.shape)
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = SigmoidActivation.derivative(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                #print(delta.shape)

            delta = np.dot(self.weights[0].transpose(), delta)
            x -= delta
            for x1_index in range (len(x)):
                x1 = x[x1_index]
                for x2_index in range (len(x1)):
                    # ?????????????????? ???? 0.998(?????????? ?????????????????????? ?????????????? 1,???????????????????????? ??????????????,
                    # ?????????? ???? ???????????????????????? ????????????????) ?????????????? ???????????????? ???? ????????
                    x2 = min(max((x1[x2_index])*0.998, 0), 1)
                    x[x1_index][x2_index] = x2
        print(self.feedforward(x))
        return x

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for layer in self.layers:
            a = layer.feedforward(a)[1]
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            early_stopping_n=0):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """

        # early stopping functionality:
        best_accuracy = 1

        #training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            #evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy=0
        no_accuracy_change=0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(self.accuracy(evaluation_data), n_data))

            # Early stopping:
            if early_stopping_n > 0:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_accuracy_change = 0
                    #print("Early-stopping: Best so far {}".format(best_accuracy))
                else:
                    no_accuracy_change += 1

                if (no_accuracy_change == early_stopping_n):
                    #print("Early-stopping: No accuracy change in last epochs: {}".format(early_stopping_n))
                    return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]
        i = 0
        print("new mini batch")
        for x, y in mini_batch:
            i+=1
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        for nabla_layer_w, nabla_layer_b, layer in zip(nabla_w, nabla_b, self.layers):
            print(np.sum(nabla_layer_w) / np.prod(nabla_layer_w.shape))
            print(np.sum(nabla_layer_b) / np.prod(nabla_layer_b.shape))
            layer.w = (1-eta*(lmbda/n))*layer.w-(eta/len(mini_batch))*nabla_layer_w
            layer.b = layer.b - (eta / len(mini_batch)) * nabla_layer_b

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_w = [[] for _ in range (len(self.layers))]
        nabla_b = [[] for _ in range (len(self.layers))]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for layer in self.layers:
            #print("feedforward next layer")
            z, activation = layer.feedforward(activation)
            zs.append(z)
            activations.append(activation)
        # backward pass
        delta_a = (self.cost).delta_a(zs[-1], activations[-1], y)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        #print("feedforward end")
        # that Python can use negative indices in lists.
        for l in range(1, self.num_layers):
            #print(l)
            layer = self.layers[-l]
            #print("b")
            #print(layer.n_in)
            z = zs[-l]
            delta_a, this_nabla_b, this_nabla_w = layer.get_delta(delta_a, z, activations[-l-1])
            nabla_b[-l] = this_nabla_b
            nabla_w[-l] = this_nabla_w
            #print(this_nabla_w.shape)
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y, 10)
            cost += self.cost.fn(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) # '**' - to the power of.
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


#### Define layer types

# later
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation=SigmoidActivation):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.result_shape = (self.filter_shape[0],
                             (self.image_shape[1]-self.filter_shape[2]+1)//self.poolsize[0],
                             (self.image_shape[2]-self.filter_shape[3]+1)//self.poolsize[1])
        self.activation = activation
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape)
        self.b = np.random.normal(loc=0, scale=1.0, size=(filter_shape[0]))
        self.params = [self.w, self.b]

    def feedforward(self, inpt):
        inpt = inpt.reshape(self.image_shape)
        conv_out = convolute_2d(inpt, self.filter_shape, self.w)
        pooled_out = pool_2d(conv_out, self.poolsize, ignore_border=True)
        #conv_out = conv.conv2d(
        #    input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
        #    image_shape=self.image_shape)
        #pooled_out = pool_2d(
        #    input=conv_out, ws=self.poolsize, ignore_border=True)
        #print(self.result_shape)
        #print(pooled_out.shape)
        #print(dimshuffle(self.b, pooled_out.shape[1:]).shape)
        output_z = self.activation.fn(
            pooled_out + dimshuffle(self.b, pooled_out.shape[1:]))
        output_a = self.activation.fn(output_z)
        return output_z.reshape(int(np.prod(self.result_shape)),1),\
               output_a.reshape(int(np.prod(self.result_shape)),1)

    def get_delta(self, a_vector_delta, z_vector, previous_a_vector):
        # reshaping from 1dim vectors
        z_vector = z_vector.reshape(self.result_shape) #3dimension: output_layers_count, x, y
        a_vector_delta = a_vector_delta.reshape(self.result_shape) #3dimension: output_layers_count, x, y
        previous_a_vector = previous_a_vector.reshape(self.image_shape)  #3dimension: input_layers_count, x, y

        z_pool_vector_prime = self.activation.derivative(z_vector)
        # ?????????????????????? ???????????????? ?? ???????????? ?????????????? z
        z_pool_vector_delta = a_vector_delta * z_pool_vector_prime
        # ???????????????????????? ??????????????????
        nabla_b_before = z_pool_vector_delta
        b_delta = [np.sum(layer_nabla_b)/np.prod(layer_nabla_b.shape)
                   for layer_nabla_b in nabla_b_before]
        z_conv_vector_delta = anti_pool_2d(z_pool_vector_delta, self.poolsize)
        w_delta = get_weights_convlayer_derivative(previous_a_vector, z_conv_vector_delta, self.filter_shape)
        #print(self.w.shape)
        #print(w_delta.shape)
        previous_a_vector_delta = backpropagate_derivatives_conv_layer(self.w, z_conv_vector_delta, self.filter_shape)
        #print(z_conv_vector_delta.shape)
        #print(previous_a_vector_delta.shape)
        #print(previous_a_vector.shape)
        return [previous_a_vector_delta, np.array(b_delta), np.array(w_delta)]



class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation=SigmoidActivation):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        # self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_out, n_in))

        self.b = np.random.normal(loc=0.0, scale=1.0, size=(n_out, 1))

        self.params = [self.w, self.b]

    def feedforward(self, inpt):
        z_vector = np.dot(self.w, inpt)
        z_vector = z_vector + self.b
        a_vector = self.activation.fn(z_vector)
        return [z_vector, a_vector]

    def get_delta(self, a_vector_delta, z_vector, previous_a_vector):
        #print(a_vector_delta.shape)
        #print(z_vector.shape)
        #print(previous_a_vector.shape)
        z_vector_prime = self.activation.derivative(z_vector)
        # ?????????????????????? ???????????????? ?? ???????????? ?????????????? z
        z_vector_delta = a_vector_delta * z_vector_prime
        # ???????????????????????? ??????????????????
        previous_a_vector_delta = np.dot(self.w.transpose(), z_vector_delta)
        w_delta = np.dot(z_vector_delta, previous_a_vector.transpose())
        b_delta = z_vector_delta
        return [previous_a_vector_delta, b_delta, w_delta]

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return np.mean(np.eq(y, self.y_out))



#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j, result_length):
    e = np.zeros((result_length, 1))
    e[j] = 1.0
    return e


class WeightInitializer:
    @staticmethod
    def default_weight_initializer(array_shape, scale):
        weights = np.random.normal(loc=0.0, scale=scale, size=array_shape)
        return weights

    @staticmethod
    def large_weight_initializer(array_shape):
        weights = np.random.normal(loc=0.0, scale=1.0, size=array_shape)
        return weights

def dimshuffle(biases, add_shape):
    result = np.zeros((biases.shape[0], add_shape[0], add_shape[1]))
    for i in range (len(biases)):
        b = biases[i]
        #print(b)
        result[i] = np.full(add_shape,b)
    return result


def pool_2d(image, pool_size, ignore_border=True):
    # image is a 3 dimensional: image_layers_count, x, y
    image_shape = image.shape
    pooled_image = np.zeros((image_shape[0],
                            image_shape[1]//pool_size[0],
                            image_shape[2]//pool_size[1]))
    for image_layer_index in range (image_shape[0]):
        for i in range (image_shape[1]//pool_size[0]):
            for j in range(image_shape[2]//pool_size[1]):
                result = np.zeros(pool_size)
                for ii in range (pool_size[0]):
                    for jj in range (pool_size[1]):
                        result[ii][jj] = image[image_layer_index][i*pool_size[0]+ii][j*pool_size[1]+jj]
                pooled_image[image_layer_index,i,j] = np.max(result)
    return pooled_image

def anti_pool_2d(image, pool_size, ignore_border=True):
    # image is a 3 dimensional: image_layers_count, x, y
    image_shape = image.shape
    anti_pooled_image = np.zeros((image_shape[0],
                            image_shape[1]*pool_size[0],
                            image_shape[2]*pool_size[1]))
    for image_layer_index in range (image_shape[0]):
        for i in range (image_shape[1]):
            for j in range(image_shape[2]):
                for ii in range (pool_size[0]):
                    for jj in range (pool_size[1]):
                        anti_pooled_image[image_layer_index][i*pool_size[0]+ii][j*pool_size[1]+jj] = image[image_layer_index][i][j]
    return anti_pooled_image
#print(pool_2d(np.array([[[9,0,0,0],[1,1,1,1],[2,2,3,3],[2,2,3,7]]]), (2,2)))
#print(anti_pool_2d(pool_2d(np.array([[[9,0,0,0],[1,1,1,1],[2,2,3,3],[2,2,3,7]]]), (2,2)), (2,2)))


def convolute_2d(image, filter_shape, weights):
    # image is a 3 dimensional: image_layers_count, x, y
    # convoluted_image is a 3 dimensional: image_out_layers_count, x, y
    # weights.shape === filter_shape
    image_shape = image.shape
    convoluted_image = np.zeros((filter_shape[0],
                            image_shape[1]-filter_shape[2]+1,
                            image_shape[2]-filter_shape[3]+1))

    for i in range (convoluted_image.shape[1]):
        for j in range (convoluted_image.shape[2]):
            #result = image[:][i:i+filter_shape[2]][j:j+filter_shape[3]]
            conv_for = np.array([[[image[kk][i+ii][j+jj] for jj in range (filter_shape[3])]
                                for ii in range(filter_shape[2])]
                               for kk in range (image.shape[0])])
            for convoluted_image_layer_index in range(filter_shape[0]):
                convoluted_image[convoluted_image_layer_index][i][j]\
                    = np.sum(conv_for*weights[convoluted_image_layer_index])
    return convoluted_image

def wide_convolute_2d(image, filter_shape, weights):
    # ?????????????? ?????????????? (???????????? ???????????? ???????????? ????-???? ???????????? ?????????????????? ?????????? ?????????????? ????????)
    # (?????????? ?????????????? ?????????????????????? ??????,
    # ?????????? ???????????????? ???????? ???? 1 ?????????????????? ???? ?????????????????????? ????????, ?? ???? ?????????????????????? ?????????? ??????)
    # image is a 3 dimensional: image_layers_count, x, y
    # convoluted_image is a 3 dimensional: image_out_layers_count, x, y
    # weights.shape === filter_shape
    image_shape = image.shape
    convoluted_image = np.zeros((filter_shape[0],
                            image_shape[1]+filter_shape[2]-1,
                            image_shape[2]+filter_shape[3]-1))
    #print(convoluted_image.shape)

    for i in range (convoluted_image.shape[1]):
        for j in range (convoluted_image.shape[2]):
            #result = image[:][i:i+filter_shape[2]][j:j+filter_shape[3]]
            #print(i-filter_shape[2]+1)
            #print(j-filter_shape[3]+1)
            conv_for = \
                np.array([[[(0 if not (image_shape[1]>i-filter_shape[2]+1+ii >= 0 and image_shape[2]>j-filter_shape[3]+1+jj >= 0)
                else image[kk][i-filter_shape[2]+1+ii][j-filter_shape[3]+1+jj])
                                 for jj in range (filter_shape[3])]
                                for ii in range (filter_shape[2])]
                               for kk in range (image.shape[0])])

            for convoluted_image_layer_index in range(filter_shape[0]):
                convoluted_image[convoluted_image_layer_index][i][j]\
                    = np.sum(conv_for*weights[convoluted_image_layer_index])
    return convoluted_image


def get_weights_convlayer_derivative(previous_a_vector, z_conv_vector_delta, weights_shape):
    previous_a_vector_shape = previous_a_vector.shape
    z_conv_vector_delta_shape = z_conv_vector_delta.shape
    weights = np.zeros(weights_shape)
    for output_layer_index in range (weights_shape[0]):
        for input_layer_index in range (weights_shape[1]):
            for y in range (weights_shape[2]):
                for x in range(weights_shape[3]):
                    #weights[output_layer_index][input_layer_index][y][x]\
                    #    = np.sum(previous_a_vector[input_layer_index][y:y+z_conv_vector_delta_shape[1]][x:x+z_conv_vector_delta_shape[2]]\
                    #    * z_conv_vector_delta[output_layer_index])
                    weights[output_layer_index][input_layer_index][y][x]\
                        = np.sum([[previous_a_vector[input_layer_index][y+yy][x+xx] for xx in range (z_conv_vector_delta_shape[2])] for yy in range (z_conv_vector_delta_shape[1])]\
                        * z_conv_vector_delta[output_layer_index])
    return weights

def backpropagate_derivatives_conv_layer(weights, z_conv_vector_delta, filter_shape):
    # z_conv_vector_delta is a 3 dimensional: image_layers_count, x, y
    # convoluted_image is a 3 dimensional: image_out_layers_count, x, y
    # weights.shape === filter_shape
    output_image_shape = z_conv_vector_delta.shape
    #unconvoluted_image_delta = np.zeros((filter_shape[1],
    #                        output_image_shape[1]+filter_shape[2]-1,
    #                        output_image_shape[2]+filter_shape[3]-1))
    transposed_weights = np.zeros((filter_shape[1],filter_shape[0],
                                   filter_shape[3],filter_shape[2]))
    for input_image_layer_index in range (filter_shape[1]):
        for output_image_layer_index in range (filter_shape[0]):
            #transposed_weights[input_image_layer_index][output_image_layer_index]\
            #    = weights[output_image_layer_index][input_image_layer_index][::-1][::-1]
            transposed_weights[input_image_layer_index][output_image_layer_index] \
                = weights[output_image_layer_index][input_image_layer_index]\
            [::-1].transpose()[::-1].transpose()
    #print(transposed_weights.shape)
    #print(wide_convolute_2d(z_conv_vector_delta, transposed_weights.shape, transposed_weights).shape)
    return wide_convolute_2d(z_conv_vector_delta, transposed_weights.shape, transposed_weights)
    #    for i in range (convoluted_image.shape[1]):
    #        for j in range (convoluted_image.shape[2]):
    #            result = image[:][i:i+filter_shape[2]][j:j+filter_shape[3]]
    #            print(result.shape)
    #            convoluted_image[convoluted_image_layer_index][i][j]\
    #                = np.sum(result*weights[convoluted_image_layer_index])
    #return convoluted_image

