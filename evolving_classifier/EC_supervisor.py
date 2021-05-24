import datetime
import os
import time
from statistics import mean
from ann_point.HyperparameterRange import *

from ann_point.AnnPoint import *

record_id = "R"
summary_id = "S"
details_id = "D"
iteration_id = "I"


class EC_supervisor():
    def __init__(self, path):
        self.start_point = 0
        self.iterations = 0
        self.desc_string = ""
        self.sps = 0
        self.ps = 0
        self.total_work = 0
        self.work_done = 0

        if not os.path.exists(path):
            os.makedirs(path)

        now = datetime.datetime.now()

        self.log_path = f"{path}{os.path.sep}log_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.txt"

    def get_algo_data(self, input_size: int, output_size: int, pm: float, pc: float, ts: int, sps: int, ps: int, fracs: [float], hrange: HyperparameterRange):
        self.desc_string += f"{details_id} ins {input_size}\n"
        self.desc_string += f"{details_id} outs {output_size}\n"
        self.desc_string += f"{details_id} pm {pm}\n"
        self.desc_string += f"{details_id} pc {pc}\n"
        self.desc_string += f"{details_id} ts {ts}\n"
        self.desc_string += f"{details_id} sps {sps}\n"
        self.desc_string += f"{details_id} ps {ps}\n"
        self.desc_string += f"{details_id} fracs {','.join([str(f) for f in fracs])}\n"
        # self.desc_string += f"{details_id} lc {hrange.layerCountMin} - {hrange.layerCountMax}\n"
        # self.desc_string += f"{details_id} nc {hrange.neuronCountMin} - {hrange.neuronCountMax}\n"
        self.desc_string += f"{details_id} acfs {','.join([f.to_string() for f in hrange.actFunSet])}\n"
        # self.desc_string += f"{details_id} agfs {','.join([f.to_string() for f in hrange.aggrFunSet])}\n"
        # self.desc_string += f"{details_id} lfs {','.join([f.to_string() for f in hrange.lossFunSet])}\n"
        # self.desc_string += f"{details_id} lr {hrange.learningRateMin} - {hrange.learningRateMax}\n"
        # self.desc_string += f"{details_id} mc {hrange.momentumCoeffMin} - {hrange.momentumCoeffMax}\n"
        # self.desc_string += f"{details_id} bs {hrange.batchSizeMin} - {hrange.batchSizeMax}\n"

        self.sps = sps
        self.ps = ps


    def start(self, iterations: int):
        self.start_point = time.time()
        self.iterations = iterations
        self.total_work = self.sps + iterations * self.ps



    def check_point(self, evals: [(AnnPoint, float)], iteration: int):
        # predict execution time
        elapsed_time = (time.time() - self.start_point)
        self.work_done += len(evals)
        frac_of_work_done = self.work_done / self.total_work
        frac_velocity = frac_of_work_done / elapsed_time

        pred_time = 1 / frac_velocity

        print(f"Remaining time: {round(pred_time - elapsed_time, 2)}")

        # evaluate statistics

        mean_eval = mean([eval[1] for eval in evals])

        best_eval = None
        for i in range(len(evals)):
            eval = evals[i]
            if best_eval is None or eval[1] > best_eval[1]:
                best_eval = eval

        # write down iteration results

        if not os.path.exists(self.log_path):
            log = open(self.log_path, "w+")
            log.write(f"{self.desc_string}")
        else:
            log = open(self.log_path, "a+")

        log.write(f"{iteration_id} Iteration {iteration + 1} \n")

        evals = sorted(evals, key=lambda x: x[1], reverse=True)

        for i in range(len(evals)):
            log.write(f"R {iteration} {i + 1} {evals[i][0].to_string()} - {round(evals[i][1], 2)}\n")

        log.write(f"S Mean eval: {round(mean_eval, 2)}\n")
        log.write(f"S Best eval: {best_eval[0].to_string()} - {round(best_eval[1], 2)}\n")

        log.close()



        print(f"desu - {iteration + 1} - {round(mean_eval, 2)} - {round(best_eval[1],  2)} - {best_eval[0].to_string()}")


