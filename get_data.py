import numpy as np


def get_ranges():
    return [10, 10]


def operation_plus(a, b):
    return a+b
# 98 percent max


def operation_multiply(a, b):
    return a*b
# 90 percent max


def operation_square_length(a, b):
    return a*a+b*b
# 60 percent max


def operation_surprise(a, b):
    return a+b+a*b
# 91 percent max

def operation_string_a_b(a, b):
    return a*10+b
# max percent 97


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