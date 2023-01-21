import time
import mnist_loader
import get_data
import numpy as np
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#training_data = list(training_data)
user_net = False
operation = get_data.operation_surprise
training_data, test_data = get_data.get_data(operation)

import neuralnetwork as network

if user_net:
    net = network.Network([20, 20, 20, 20, 20])
    net.SGD(training_data, 1000, 20, 8.0, test_data=test_data)
elif operation is get_data.operation_string_a_b:
    net = network.Network([20, 110])
    net.SGD(training_data, 1000, 20, 8.0, test_data=test_data)
elif operation is get_data.operation_surprise:
    net = network.Network([20, 20, 20, 20, 120])
    net.SGD(training_data, 1000, 20, 5.0, test_data=test_data)
elif operation is get_data.operation_plus:
    # for plus
    net = network.Network([20, 20, 20, 20, 20])
    net.SGD(training_data, 500, 20, 5.0, test_data=test_data)
elif operation is get_data.operation_multiply:
    # for multiply
    net = network.Network([20, 20, 20, 100])
    net.SGD(training_data, 500, 20, 10.0, test_data=test_data)
elif operation is get_data.operation_square_length:
    net = network.Network([20, 20, 20, 60, 200])
    net.SGD(training_data, 500, 20, 10.0, test_data=test_data)
else:
    net = network.Network([20, 20, 20, 60, operation(get_data.get_ranges())])
    net.SGD(training_data, 500, 20, 5.0, test_data=test_data)


print(time.process_time())
while True:
    i, j = map(int, input().split(" "))
    print(np.argmax(net.feedforward(get_data.get_arrays_by_numbers(i, j))))
