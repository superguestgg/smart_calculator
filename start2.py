import time
import mnist_loader
import get_data
import numpy as np
import neuralnetwork2 as network


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

net = network.Network([784, 30, 10])
net.SGD(training_data[:1000], 30, 10, 0.5, evaluation_data=test_data[0:1000], lmbda=5,
        monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)