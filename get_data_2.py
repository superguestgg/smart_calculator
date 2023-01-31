import numpy as np


def get_digits(value, result_length=False):
    list_of_digits = []
    while value > 0:
        list_of_digits.append(value % 10)
        value //= 10
    if result_length:
        if result_length == len(list_of_digits):
            return list_of_digits
        elif result_length < len(list_of_digits):
            return list_of_digits[:result_length-1]
        elif result_length > len(list_of_digits):
            for i in range (result_length - len(list_of_digits)):
                list_of_digits.append(0)
            return list_of_digits
    return list_of_digits


def get_count_digits(value):
    return len(get_digits(value))


def digits_to_vector(digits_list, result_length, digits_list_2=False):
    if digits_list_2:
        list_vector = [[0] for _ in range(result_length)]
        for digit_index in range(len(digits_list)):
            digit = digits_list[digit_index]
            list_vector[10 * digit_index + digit] = [1]
        for digit_2_index in range(len(digits_list_2)):
            digit_2 = digits_list_2[digit_2_index]
            list_vector[len(digits_list) * 10 + 10 * digit_2_index + digit_2] = [1]

        return list_vector
    else:
        list_vector = [[0] for _ in range(result_length)]
        for i_digit_index in range(len(digits_list)):
            i_digit = digits_list[i_digit_index]
            list_vector[10 * i_digit_index + i_digit] = [1]
        return list_vector


class Ranges:
    def __init__(self, a_digits_count, b_digits_count, operation):
        # здесь числа - порядки перебора для 1го и 2го чисел
        self.a_digits_count = a_digits_count
        self.b_digits_count = b_digits_count
        self.result_digits_count = get_count_digits(operation(10 ** a_digits_count, 10 ** a_digits_count))

    def get_ranges_count(self):
        return [self.a_digits_count, self.b_digits_count,
                self.result_digits_count]

    def get_ranges_value(self):
        return [10 ** self.a_digits_count, 10 ** self.b_digits_count,
                10 ** self.result_digits_count]


def operation_plus(a, b):
    return a + b


def operation_multiply(a, b):
    return a * b


def operation_square_length(a, b):
    return a * a + b * b


def operation_surprise(a, b):
    return a + b + a * b


def operation_string_a_b(a, b):
    return a * 10 + b


def get_data(operation, ranges):
    training_data = []
    test_data = []
    range_i, range_j, range_result = ranges.get_ranges_count()
    range_i_big, range_j_big, range_result_big = ranges.get_ranges_value()

    for i in range(range_i_big):
        i_list = get_digits(i, result_length=range_i)
        for j in range(range_j_big):
            j_list = get_digits(j, result_length=range_j)

            one_part_data_input = digits_to_vector(i_list, range_i*10+range_j*10, j_list)

            this_result = operation(i, j)
            this_result_copy = this_result
            this_result_list_of_digits = get_digits(this_result_copy)
            one_part_data_output = [[0] for _ in range(10 * range_result)]
            for r_digit_index in range(len(this_result_list_of_digits)):
                r_digit = this_result_list_of_digits[r_digit_index]
                one_part_data_output[10 * r_digit_index + r_digit] = [1]
            training_data.append([np.array(one_part_data_input), np.array(one_part_data_output)])
            one_part_data_output_int = this_result
            test_data.append([np.array(one_part_data_input), one_part_data_output_int])
    return [training_data, test_data]


def get_arrays_by_numbers(i, j, ranges):
    range_i, range_j, notmatter = ranges.get_ranges_count()
    list_i = get_digits(i, range_i)
    list_j = get_digits(j, range_j)
    result = digits_to_vector(list_i, 10*range_i + 10*range_j, list_j)
    return result
