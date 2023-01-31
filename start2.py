import time
import mnist_loader
import get_data_2
import numpy as np
import neuralnetwork2 as network

if False:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)

    net = network.Network([784, 30, 10])
    net.SGD(training_data[:5000], 30, 10, 0.5, evaluation_data=test_data[0:1000], lmbda=100,
        monitor_evaluation_accuracy=False, monitor_evaluation_cost=False,
        monitor_training_cost=True, monitor_training_accuracy=True)
else:
    operation = get_data_2.operation_plus
    ranges = get_data_2.Ranges(2, 2, operation)
    training_data, test_data = get_data_2.get_data(operation, ranges)


    net = network.Network([40, 30, 30])
    net.SGD(training_data, 10, 20, 5.0, evaluation_data=test_data, lmbda=0.0,
        monitor_evaluation_accuracy=False, monitor_evaluation_cost=False,
        monitor_training_cost=True, monitor_training_accuracy=True)
    while True:
        a, b = map(int, input().split())
        result = net.feedforward(get_data_2.get_arrays_by_numbers(a, b, ranges))
        result_str = ""
        for i in range (ranges.get_ranges_count()[2]):
            if max(result[i*10:(i+1)*10]) > 0.5:
                result_str = str(np.argmax(result[i*10:(i+1)*10])) + result_str
            else:
                result_str = "0" + result_str
        print(result_str)