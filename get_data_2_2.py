import numpy as np
import random
import get_data_2


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def anti_sigmoid(a):
    return -np.log(1.0/a-1.0)



def bin_operation_square(x):
    return x**2


def get_data(operation=bin_operation_square):
    training_data = []
    test_data = []
    for i in range (1000):
        x = random.random() * 36
        y = bin_operation_square(x)
        training_data.append([np.array([sigmoid(x)]), np.array([sigmoid(y)])])
    return [training_data, test_data]
