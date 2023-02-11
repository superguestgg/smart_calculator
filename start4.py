import draw
import shape_open_json
import time
import mnist_loader
import get_data_2
import get_data_2_2
import numpy as np
import neuralnetwork4 as network

print("starting mnist_digits_recognition")
net = network.Network(
    [network.FullyConnectedLayer(784, 30),
     network.FullyConnectedLayer(30, 10)]
)
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net.SGD(training_data, 10, 10, 0.5,
        evaluation_data=test_data[0:10000], lmbda=1,
        monitor_evaluation_accuracy=True, monitor_evaluation_cost=False,
        monitor_training_accuracy=True, monitor_training_cost=False)

'''net  = network.Network(
    [network.ConvPoolLayer(image_shape=(1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2)),
     network.ConvPoolLayer(image_shape=(20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2)),
     network.FullyConnectedLayer(n_in=(40*4*4), n_out=100),
     network.FullyConnectedLayer(100, 10)]
)'''
