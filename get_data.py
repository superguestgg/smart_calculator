import numpy as np


def get_ranges():
    return [10, 10]


def operation_plus(a, b):
    return a+b
# 98 / 100 correct max if the cost function is (output_activations-y)**2
# 100 / 100 correct max if the cost function is (output_activations-y)**4
# and so the cost_derivative function is (output_activations - y)**3
# 400 / 400
# 1513 / 1600 after 500 epochs and 296.96875 seconds
# 1600 / 1600 after 155 epochs if the cost function is
# -y*np.log(output_activations) - (1 - y) * np.log(1 - output_activations)
# and so the cost_derivative function is -y/output_activations + (1 - y)/(1 - output_activations)
# cross-entropy function above

def operation_multiply(a, b):
    return a*b
# 90 percent max
# 100 / 100 correct max if the cost function is (output_activations-y)**4
# and so cost_derivative function is (output_activations - y)**3


def operation_square_length(a, b):
    return a*a+b*b
# 60 percent max
# 98 / 100 correct max if the cost function is (output_activations-y)**4
# and so cost_derivative function is (output_activations - y)**3


def operation_surprise(a, b):
    return a+b+a*b
# 91 percent max
# 100 / 100 correct max if the cost function is (output_activations-y)**4
# and so cost_derivative function is (output_activations - y)**3


def operation_string_a_b(a, b):
    return a*10+b
# max percent 97
# 100 / 100 correct max if the cost function is (output_activations-y)**4
# and so cost_derivative function is (output_activations - y)**3


def get_data(operation=operation_plus):
    training_data = []
    test_data = []
    range_i, range_j = get_ranges()
    for i in range (range_i):
        for j in range (range_j):
            one_part_data_input = [[0] for _ in range (range_i + range_j)]
            one_part_data_input[i] = [1]
            one_part_data_input[j+range_i] = [1]
            one_part_data_output = [[0] for _ in range (operation(range_i, range_j))]
            one_part_data_output[operation(i, j)] = [1]
            training_data.append([np.array(one_part_data_input), np.array(one_part_data_output)])
            one_part_data_output_int = operation(i, j)
            test_data.append([np.array(one_part_data_input), one_part_data_output_int])
    return [training_data, test_data]


def get_arrays_by_numbers(i, j):
    range_i, range_j = get_ranges()
    one_part_data_input = [[0] for _ in range(range_i + range_j)]
    one_part_data_input[i] = [1]
    one_part_data_input[j + range_i] = [1]
    return one_part_data_input