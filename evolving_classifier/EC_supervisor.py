# import datetime
# import os
# import time
# from statistics import mean
# from ann_point.HyperparameterRange import *
#
# # from ann_point.AnnPoint import *
# from utility.CNDataPoint import CNDataPoint, ChaosNet
# from utility.RunHistory import RunHistory
#
# record_id = "R"
# summary_id = "S"
# details_id = "D"
# iteration_id = "I"
#
# class EC_supervisor():
#     def __init__(self, path):
#         self.start_point = 0
#         self.iterations = 0
#         self.desc_string = ""
#         self.sps = 0
#         self.ps = 0
#         self.total_work = 0
#         self.work_done = 0
#         self.rh = RunHistory()
#
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         now = datetime.datetime.now()
#
#         self.log_path = f"{path}{os.path.sep}log_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.txt"
#
#     def get_algo_data(self, input_size: int, output_size: int, pmS: float, pmE: float, pcS: float, pcE: float, ts: int, sps: int, ps: int, fracs: [float], hrange: HyperparameterRange, learningIts: int):
#         self.desc_string += f"{details_id} ins {input_size}\n"
#         self.desc_string += f"{details_id} outs {output_size}\n"
#         self.desc_string += f"{details_id} pm {pmS}-{pmE}\n"
#         self.desc_string += f"{details_id} pc {pcS}-{pcE}\n"
#         self.desc_string += f"{details_id} ts {ts}\n"
#         self.desc_string += f"{details_id} sps {sps}\n"
#         self.desc_string += f"{details_id} ps {ps}\n"
#         self.desc_string += f"{details_id} fracs {','.join([str(f) for f in fracs])}\n"
#         # self.desc_string += f"{details_id} lc {hrange.hiddenLayerCountMin} - {hrange.hiddenLayerCountMax}\n"
#         # self.desc_string += f"{details_id} nc {hrange.neuronCountMin} - {hrange.neuronCountMax}\n"
#         self.desc_string += f"{details_id} acfs {','.join([f.to_string() for f in hrange.actFunSet])}\n"
#         # self.desc_string += f"{details_id} lfs {','.join([f.to_string() for f in hrange.lossFunSet])}\n"
#         self.desc_string += f"{details_id} lits {learningIts}\n"
#
#         self.sps = sps
#         self.ps = ps
#
#
#     def start(self, iterations: int):
#         self.start_point = time.time()
#         self.iterations = iterations
#         self.total_work = self.sps + iterations * self.ps
#
#     def check_point(self, evals: [(ChaosNet, CNDataPoint)], iteration: int):
#         # predict execution time
#         curr_time = time.time()
#         elapsed_time = (curr_time - self.start_point)
#         self.work_done += len(evals)
#         frac_of_work_done = self.work_done / self.total_work
#         frac_velocity = frac_of_work_done / elapsed_time
#
#         pred_time = 1 / frac_velocity
#
#         print(f"Remaining time: {round(pred_time - elapsed_time, 2)}")
#
#         # evaluate statistics
#
#         it_hist = [evals[i] for i in range(len(evals))]
#         self.rh.add_it_hist(it_hist)
#
#         # write down iteration results
#
#         if not os.path.exists(self.log_path):
#             log = open(self.log_path, "w+")
#             log.write(f"{self.desc_string}")
#         else:
#             log = open(self.log_path, "a+")
#
#         log.write(f"{iteration_id} Iteration {iteration + 1} \n")
#
#         evals = sorted(evals, key=lambda x: x.ff, reverse=True)
#
#         round_prec = 3
#
#         for i in range(len(evals)):
#             # log.write(f"R {iteration} {i + 1} {evals[i][0].to_string()} - {round(evals[i][1].ff, round_prec)}\n")
#             log.write(f"R {iteration} {i + 1} {evals[i].net.to_string()} - {round(evals[i].ff, round_prec)}\n")
#
#         best_eval = self.rh.get_it_best(iteration)
#
#         stat_string = self.rh.get_it_summary_string(iteration)
#
#         log.write(f"{summary_id} {stat_string}\n")
#         log.write(f"{summary_id} Best ff: {best_eval.net.to_string()} - {round(best_eval.ff, round_prec)}\n")
#
#         log.close()
#
#         # iteration prints
#
#         # print(f"--- Iteration - {iteration + 1} - {stat_string}")
#         # print(f"--- Best point: {best_eval.net.to_string()}")
#
#
