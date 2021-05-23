from sklearn import datasets
from evolving_classifier.EvolvingClassifier import *
import os

random.seed(1001)
ec_pop_size = 10
ec_starting_pop_size = 10
ec_iterations = 5
nn_epochs = 50
power = 6
n_id = "N"
r_id = "R"

res_folder = "ec_test"

def run_tests(repetitions: int, path: str, nn_data: ([np.ndarray], [np.ndarray], [np.ndarray], [np.ndarray])):
    log = open(f"{res_folder}{os.path.sep}{path}{os.path.sep}result.txt", "a+")
    for i in range(repetitions):
        print(f"{i + 1}")
        ec = EvolvingClassifier(logs_path=f"{res_folder}{os.path.sep}{path}")
        ec_seed = random.randint(0, 10000)
        ec.prepare(popSize=ec_pop_size, startPopSize=ec_starting_pop_size, nn_data=nn_data, seed=ec_seed)
        point = ec.run(iterations=ec_iterations, power=power)
        nn_seed = random.randint(0, 10000)
        network = network_from_point(point=point, seed=nn_seed)
        network.train(nn_data[0], nn_data[1], epochs=nn_epochs)
        test_res = network.test(nn_data[2], nn_data[3])

        log.write(f"{n_id} {point.to_string_full()} {nn_seed}\n")
        log.write(f"{r_id} Acc: {test_res[0]} Pre: {test_res[1]} Rec: {test_res[2]}, Eff: {mean(test_res[0:3])}\n")
        log.flush()
    log.close()



def ec_test_irises(repetitions: int, path: str):
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    x = [x.reshape((4, 1)) for x in x]
    y = one_hot_endode(y)

    perm = list(range(0, len(y)))
    random.shuffle(perm)

    x = [x[i] for i in perm]
    y = [y[i] for i in perm]

    train_X = [x[i] for i in range(120)]
    train_y = [y[i] for i in range(120)]
    test_X = [x[i] for i in range(120, 150)]
    test_y = [y[i] for i in range(120, 150)]

    run_tests(repetitions=repetitions, path=path, nn_data=(train_X, train_y, test_X, test_y))

def ec_test_count(repetitions: int, path: str):
    count_tr = 300
    count_test = 300
    size = 5
    train_X, train_y = generate_counting_problem(count_tr, size)
    test_X, test_y = generate_counting_problem(ceil(count_test), size)

    run_tests(repetitions=repetitions, path=path, nn_data=(train_X, train_y, test_X, test_y))




if __name__ == '__main__':
    # ec_test_irises(repetitions=7, path="iris_test")
    ec_test_count(repetitions=5, path="count_test")

