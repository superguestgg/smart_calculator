import draw
import time
import mnist_loader
import get_data_2
import numpy as np
import neuralnetwork2 as network

if True:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)

    net = network.Network([784, 30, 10])
    net.SGD(training_data[:5000], 2, 10, 0.5, evaluation_data=test_data[0:1000], lmbda=10,
        monitor_evaluation_accuracy=False, monitor_evaluation_cost=False,
        monitor_training_cost=True, monitor_training_accuracy=True)
    #net.draw_input(np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]))
    draw.draw_by_pixels(net.draw_input(np.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]])))
else:
    # не знаю зачем, но такой калькулятор складывает числа до 100 без ошибок
    # после 10 эпохи( в хорошем случае), в плохом после 20
    # а потом я понял что это х...,(генерация рандомов не работала на бошльших числах (до 1000)
    # потому что люди складывают, умножают числа больше 1000 по разрядам, а не сразу
    # кстати запомнить таблицу умножения до 100 1 скрытый слой не смог
    # а потом я вспомнил, что для умножения я использовал 3 скрытых слоя
    # (просто вспомнил как компьютер побитово умножает)
    # ну и веса в функцию стоимости добавлять не надо, потому что каждая активация нейрона
    # имеет большой смысл и независима от соседних нейронов (я про входной слой)
    operation = get_data_2.operation_multiply
    ranges = get_data_2.Ranges(2, 2, operation)
    training_data, test_data = get_data_2.get_data(operation, ranges)
    #training_data, test_data = get_data_2.get_data_econom(operation, ranges, 100)


    net = network.Network([40, 40, 40, 40])
    net.SGD(training_data, 500, 20, 5.0, evaluation_data=test_data, lmbda=0.00,
        monitor_evaluation_accuracy=False, monitor_evaluation_cost=False,
        monitor_training_cost=False, monitor_training_accuracy=False)
    print(time.process_time())
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