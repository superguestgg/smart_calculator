import numpy as np
import random


def sigmoid(z):
    return (1/(1+np.exp(-z)))
def my_sigmoid(z):
    return (1/(1+np.exp(-z)))*2-1
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    #производная сигмоиды
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def save(self, file_name):
        file_to_save = open(file_name, "w")
        file_to_save.write(str(self.biases))
        file_to_save.write("\n\n")
        file_to_save.write(str(self.weights))


    def feedforward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        #данные для обучения, количество эпох, размер мини-пакета
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If "test_data" is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        """Обучите нейронную сеть с помощью мини-пакетного стохастика
        градиентный спуск. «training_data» — это список кортежей
        «(x, y)», представляющие входные данные для обучения и желаемый результат.
        выходы. Другие необязательные параметры
        самоочевидно. Если предоставлено "test_data", то
        сеть будет оцениваться по тестовым данным после каждого
        эпоха, и частичный прогресс распечатан. Это полезно для
        отслеживание прогресса, но существенно замедляет работу. Обучайте нейронную сеть с помощью мини-пакетного стохастика.
        градиентный спуск. «training_data» — это список кортежей
        «(x, y)», представляющие входные данные для обучения и желаемый результат.
        выходы. Другие необязательные параметры
        самоочевидно. Если предоставлено "test_data", то
        сеть будет оцениваться по тестовым данным после каждого
        эпоха, и частичный прогресс распечатан. Это полезно для
        отслеживание прогресса, но существенно замедляет работу."""
        if test_data:
            #test_data = list(test_data)
            n_test = len(test_data)
        #training_data = list(training_data)
        n = len(training_data)
        for j in range (epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                result_test = self.evaluate(test_data)
                print("Epoch {0}: {1} / {2}".format(
                    j, result_test, n_test))
                if result_test==n_test:
                    break
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""

        """Обновите веса и смещения сети, применив
        градиентный спуск с использованием обратного распространения к одной мини-партии.
        «mini_batch» — это список кортежей «(x, y)» и «eta».
        скорость обучения."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #создают массивы такие же как с добавками и весами, но заполненные нулями
        #w.shape - размеры numpy array для каждого уровня
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # x and y - in and out
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            #print(nabla_w)
            ## просто добавляем изменение весов
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        """Возвращает кортеж ``(nabla_b, nabla_w)``, представляющий
         градиент для функции стоимости C_x. ``nabla_b`` и
         ``nabla_w`` - это послойные списки массивов numpy, похожие
         к ``self.biases`` и ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # x and y - in and out
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        # список для хранения всех активаций, слой за слоем
        zs = []  # list to store all the z vectors, layer by layer
        # список для хранения всех векторов z, слой за слоем
        for b, w in zip(self.biases, self.weights):
            ##проходимся по слоям, находим значения на нейронах
            z = np.dot(w, activation) + b
            #print(z)
            zs.append(z)
            activation = sigmoid(z)
            #print(sigmoid(z))
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        # считает ошибку
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        # Обратите внимание, что переменная l в приведенном ниже цикле используется немного
        # иначе, чем обозначения в главе 2 книги. Здесь,
        # l = 1 означает последний слой нейронов, l = 2 —
        # предпоследний слой и так далее. это перенумерация
        # схема в книге, используемая здесь, чтобы воспользоваться этим фактом
        # что Python может использовать отрицательные индексы в списках.
        # последний комментарий устарел
        for l in range(self.num_layers-1, 1, -1):
            # перебираем l от предпоследнего слоя до 2го
            z = zs[l-2]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[l+1-2].transpose(), delta) * sp
            nabla_b[l-2] = delta
            nabla_w[l-2] = np.dot(delta, activations[l-2].transpose())
            #sp = sigmoid_prime(z)
            #delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            #nabla_b[-l] = delta
            #nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        #оценивать
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        """Вернуть вектор частных производных \partial C_x/
                 \partial a для выходных активаций."""
        return -y/output_activations+(1-y)/(1-output_activations)
        #return (output_activations - y)**3
        #return (output_activations - y)



class Network_linear(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def save(self, file_name):
        file_to_save = open(file_name, "w")
        file_to_save.write(str(self.biases))
        file_to_save.write("\n\n")
        file_to_save.write(str(self.weights))


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = (np.dot(w, a)+b)
        # a = sigmoid(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range (epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # x and y - in and out
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = z
            activations.append(activation)
            #print(z)
        #print("kkk")
        #activations[-1] = sigmoid(activations[-1])
        delta = self.cost_derivative(activations[-1], y)
#                sigmoid_prime(zs[-1])
        delta/=max(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(self.num_layers-1, 1, -1):
            z = zs[l-2]
            delta = np.dot(self.weights[l+1-2].transpose(), delta)
            delta/=max(delta)
            #print(delta)
            nabla_b[l-2] = delta
            nabla_w[l-2] = np.dot(delta, activations[l-2].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        #print(test_data[0])
        #print(self.feedforward(test_data[0][0]))
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
"""net = Network([2, 3, 1])
print(net.biases)
for b in net.biases:
    print(b.shape)
aa=[[1,2,3,4,5],[5,6,7,8,9]]
bb=[[9,8,7,6,5],[5,4,3,2,1]]
for x1 in zip(aa,bb):
    print(x1)"""
#nabla_b = [np.zeros(b.shape) for b in self.biases]
"""print(sigmoid([[0.27285351]
 [11.23286048]
 [-9.45780437]
 [19.82168624]
 [4.22436225]
 [18.17788671]
 [-3.54499085]
 [-2.9077173 ]
 [18.25164755]
 [20.13187667]]))"""