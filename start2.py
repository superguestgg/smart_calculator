import draw
import shape_open_json
import time
import mnist_loader
import get_data_2
import get_data_2_2
import numpy as np
import neuralnetwork2 as network

if False:
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)

    net = network.Network([784, 30, 10])
    net.SGD(training_data[:5000], 2, 10, 0.5, evaluation_data=test_data[0:1000], lmbda=10,
        monitor_evaluation_accuracy=False, monitor_evaluation_cost=False,
        monitor_training_cost=False, monitor_training_accuracy=False)
    #net.draw_input(np.array([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]))
    a_s = 0 # activation standard (for not activated neurons)
    a_a = 1 # activation for activated neurons
    draw.draw_by_pixels(net.get_input(np.array([[a_s], [a_s], [a_s], [a_s], [a_s], [a_s], [a_s], [a_a], [a_s], [a_s]])))
elif False:
    # не знаю зачем, но такой калькулятор складывает числа до 100 без ошибок
    # после 10 эпохи( в хорошем случае), в плохом после 20
    # а потом я понял что это х...,(генерация рандомов не работала на бошльших числах (до 1000)
    # потому что люди складывают, умножают числа больше 1000 по разрядам, а не сразу
    # кстати запомнить таблицу умножения до 100 1 скрытый слой не смог
    # а потом я вспомнил, что для умножения я использовал 3 скрытых слоя
    # (просто вспомнил как компьютер побитово умножает)
    # ну и веса в функцию стоимости добавлять не надо, потому что каждая активация нейрона
    # имеет большой смысл и независима от соседних нейронов (я про входной слой)
    operation = get_data_2.operation_plus
    ranges = get_data_2.Ranges(1, 1, operation)
    training_data, test_data = get_data_2.get_data(operation, ranges)
    #training_data, test_data = get_data_2.get_data_econom(operation, ranges, 100)


    net = network.Network([20, 20, 20, 20])
    net.SGD(training_data, 500, 20, 5.0, evaluation_data=test_data, lmbda=0.00,
        monitor_evaluation_accuracy=False, monitor_evaluation_cost=False,
        monitor_training_cost=True, monitor_training_accuracy=True)
    print(net.get_input(np.array([[0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])))
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
elif False:
    # my example for 4 chapter of book
    # compute the uno_fuction, forexample x**2,
    # using anti-sigmoid(y) (y = result) as answer
    # к слову эта часть доказывает, что нейросеть
    # может подогнать любую функцию
    # (типо переобучения, условно запомнить все входы
    # и выходы для них и выдавать на них максимально правильный
    # результат), но при этом не факт что она будет давать
    # правильные ответы на запросы, которых не было раньше,
    # ну и еще доказывает что обучать нейросеть с 1м выходом и 1м входом можно,
    # но как будто менее эффективно
    # такая нейросеть не может выводить слищком большие значения
    # (вероятнее всего изза того, что для большого значения z в последнем слое нужны большие веса,
    # а они не получаются изза малого количества эпох, большого количества данных для обучения на маленьктх числах
    # и перенасыщения нейронов

    training_data, test_data = get_data_2_2.get_data()

    net = network.Network([1, 80, 40, 1])
    net.SGD(training_data, 500, 20, 1.0, evaluation_data=test_data, lmbda=0.00,
        monitor_evaluation_accuracy=False, monitor_evaluation_cost=False,
        monitor_training_cost=True, monitor_training_accuracy=False)
    print(time.process_time())
    while True:
        z1 = float(input())
        if z1 > 6:
            print("value is too big ((z1**2)>36)")
            print("значение слишком большое ((z1**2)>36)")
            print("при вычислении анти-сигмоиды возникает бесконечность")
            continue
        result = get_data_2_2.anti_sigmoid(net.feedforward(get_data_2_2.sigmoid(z1)))
        result_str = str(result)
        print(get_data_2_2.anti_sigmoid(get_data_2_2.sigmoid(z1**2)))
        print(result_str)
else:
    # а теперь по этому примеру:
    # есть 3 файла: [2(72 обычных примера),
    # 3 (94 расширенных примера( попадается больше различных вариантов),
    # 4 (50 обычных примеров)]
    # обучение 3->2, 3->5 : 100% correct,
    # 2->3 60% correct max(58/94) (lambda=4.25, eta=0.5),
    # 2->5 100% correct(lmbda=0.25, eta=0.5, after 5 epochs)
    # 5->2 100%, 5->3 60% correct max (lambda=4.125, eta=0.5),
    # 2->2, 5->5 100% after 9 examples, low lambda,
    # потому что здесь примеры с одинаковыми ответами очень похожи
    # 3->3 50-54/94 after 9 examples,
    training_data = shape_open_json.open_json()
    test_data = shape_open_json.open_json_test_data()

    net = network.Network([784, 80, 40, 3])
    net.SGD(training_data[:10], 5000, 10, 0.05, evaluation_data=test_data, lmbda=0.125,
        monitor_evaluation_accuracy=True, monitor_evaluation_cost=True,
        monitor_training_cost=False, monitor_training_accuracy=False)
    print(time.process_time())
