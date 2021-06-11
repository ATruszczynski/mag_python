from sklearn import datasets
from evolving_classifier.EvolvingClassifier import *
import os

from utility.RunHistory import *

random.seed(1001)
ec_pop_size = 10
ec_starting_pop_size = 10
ec_iterations = 5
nn_epochs = 10
nn_reps = 5
power = 12
pm = 0.05
pc = 0.8
r_id = "R"
n_id = "N"
b_id = "B"
s_id = "S"
filler = "-"
filler_space = f"{filler},{filler},{filler},{filler},{filler},{filler},{filler},{filler}"
res_folder = "ec_test"

def run_tests(repetitions: int, res_subdir_path: str, nn_data: ([np.ndarray], [np.ndarray], [np.ndarray], [np.ndarray]),
              cus_hrange: HyperparameterRange = None, cus_cross_op: CrossoverOperator = None, cus_mut_op: MutationOperator = None,
              cus_sel_op: SelectionOperator = None, cus_ff: FitnessFunction = None, cus_fc: FitnessCalculator = None):
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)

    test_subdir = f"{res_folder}{os.path.sep}{res_subdir_path}"
    if not os.path.exists(test_subdir):
        os.mkdir(test_subdir)

    log = open(f"{test_subdir}{os.path.sep}result.csv", "a+")
    log.write("run,type,it,ff,acc,prec,rec,eff,f1,lc,size,bnet,bff,bacc,bprec,brec,beff,bf1,blc,bsize,epo\n")

    for i in range(repetitions):
        print(f"{i + 1}")
        ec = get_ec(test_subdir=test_subdir, cus_hrange=cus_hrange, cus_cross_op=cus_cross_op,
                    cus_mut_op=cus_mut_op, cus_sel_op=cus_sel_op, cus_ff=cus_ff, cus_fc=cus_fc)
        ec_seed = random.randint(0, 10000)
        ec.prepare(popSize=ec_pop_size, startPopSize=ec_starting_pop_size, nn_data=nn_data, seed=ec_seed)
        point = ec.run(iterations=ec_iterations, pm=pm, pc=pc, power=power)

        learningIts = ec.ff.learningIts
        write_down_run_hist(file=log, run_it=i, rh=ec.supervisor.rh, learningIts=learningIts)

        net_data_points = []
        for j in range(nn_reps):
            nn_seed = random.randint(0, 10000)
            network = network_from_point(point=point, seed=nn_seed)
            network.train(nn_data[0], nn_data[1], epochs=nn_epochs)
            test_res = network.test(nn_data[2], nn_data[3])
            ann_data_point = AnnDataPoint(point=point)
            ann_data_point.add_data(new_ff=0, new_conf_mat=test_res[3])
            net_data_points.append(ann_data_point)
            write_down_net_res(file=log, run_it=i, variant=n_id, n_it=j, data_point=ann_data_point, learningIts=learningIts)

        write_down_avg_net(file=log, run_it=i, data_points=net_data_points, learningIts=learningIts)
        ordered_pounts = sorted(net_data_points, key=lambda x: x.get_eff(), reverse=True)
        write_down_net_res(file=log, run_it=i, variant=b_id, n_it=-1, data_point=ordered_pounts[0], learningIts=learningIts)

        log.flush()
    log.close()

#TODO rounding too few digits
#TODO sprawdzić czy na pewno kolejności danych są dobre
def write_down_run_hist(file, run_it: int, rh: RunHistory, learningIts: int):
    for i in range(len(rh.it_hist)):
        summary = rh.summary_dict(i)
        best = rh.get_it_best(i)
        record = f"{run_it},{r_id},{i}," \
                 f"{summary[mean_ff_id]},{summary[mean_acc_id]},{summary[mean_prec_id]},{summary[mean_rec_id]}," \
                 f"{summary[mean_eff_id]}," \
                 f"{summary[mean_f1_id]},{summary[mean_lc_id]},{summary[mean_size_id]}," \
                 f"{best.point.to_string()},{best.ff},{best.acc},{best.prec},{best.rec},{best.get_eff()},{best.f1}," \
                 f"{len(best.point.neuronCounts)-2},{best.point.size()},{learningIts}\n"
        file.write(record)
#TODO czy eff ma jakąś oficjalną nazwę

def write_down_net_res(file, run_it: int, variant: str, n_it: int, data_point: AnnDataPoint, learningIts: int):
    record = ""
    if variant == n_id:
        record = f"{run_it},{variant},{n_it},"
    else:
        record = f"{run_it},{variant},{filler},"

    record += f"{filler_space}," \
              f"{data_point.point.to_string()},{filler}," \
              f"{data_point.acc},{data_point.prec},{data_point.rec},{data_point.get_eff()},{data_point.f1}," \
              f"{len(data_point.point.neuronCounts)-2},{data_point.point.size()},{learningIts}\n"

    file.write(record)

def write_down_avg_net(file, run_it: int, data_points: [AnnDataPoint], learningIts: int):
    rh = RunHistory()
    rh.add_it_hist(data_points)
    summary = rh.summary_dict(0)

    record = f"{run_it},{s_id}," \
             f"{filler},{filler}," \
             f"{summary[mean_acc_id]},{summary[mean_prec_id]},{summary[mean_rec_id]},{summary[mean_eff_id]}," \
             f"{summary[mean_f1_id]},{summary[mean_lc_id]},{summary[mean_size_id]}," \
             f"{data_points[0].point.to_string()}," \
             f"{filler_space},{learningIts}\n"

    file.write(record)

def get_ec(test_subdir,
           cus_hrange: HyperparameterRange = None, cus_cross_op: CrossoverOperator = None,
           cus_mut_op: MutationOperator = None, cus_sel_op: SelectionOperator = None,
           cus_ff: FitnessFunction = None, cus_fc: FitnessCalculator = None):
    ec = EvolvingClassifier(logs_path=f"{test_subdir}")
    if cus_hrange is not None:
        ec.hrange = cus_hrange
    if cus_cross_op is not None:
        ec.co = cus_cross_op
    if cus_mut_op is not None:
        cus_mut_op.hrange = ec.hrange
        ec.mo = cus_mut_op
    if cus_sel_op is not None:
        ec.so = cus_sel_op
    if cus_ff is not None:
        ec.ff = cus_ff
    if cus_fc is not None:
        ec.fc = cus_fc
    return ec








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

    run_tests(repetitions=repetitions, res_subdir_path=path, nn_data=(train_X, train_y, test_X, test_y), cus_fc=PlusSizeFitnessCalculator(def_frac, 0.5))

def ec_test_count(repetitions: int, path: str):
    count_tr = 300
    count_test = 300
    size = 5
    train_X, train_y = generate_counting_problem(count_tr, size)
    test_X, test_y = generate_counting_problem(ceil(count_test), size)

    run_tests(repetitions=repetitions, res_subdir_path=path, nn_data=(train_X, train_y, test_X, test_y))




if __name__ == '__main__':
    ec_test_irises(repetitions=7, path="iris_test")
    # ec_test_count(repetitions=5, path="count_test")

