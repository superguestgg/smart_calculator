import time
import mnist_loader
import get_data
import numpy as np
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#print(len(list(training_data)))
#print((list(training_data))[5][0][0])
#training_data = list(training_data)
operation = get_data.operation_stepen
training_data, test_data = get_data.get_data(operation)
#print(list(test_data)[0])
import neuralnetwork as network

#net = network.Network([20, 20, 20, 20, 20])
net = network.Network([20, 20, 20, 100])
net.SGD(training_data, 500, 20, 5.0, test_data=test_data)
print(time.process_time())
while True:
    i, j = map(int, input().split(" "))
    print(np.argmax(net.feedforward(get_data.get_arrays_by_numbers(i, j))))
