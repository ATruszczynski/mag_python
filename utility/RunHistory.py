from statistics import mean
import numpy as np

from utility.CNDataPoint import CNDataPoint

mean_ff_id = "m.f."
mean_acc_id = "m.a."
mean_prec_id = "m.p."
mean_rec_id = "m.r."
mean_eff_id = "m.e."
mean_size_id = "m.s."
mean_f1_id = "m.f1."
mean_lc_id = "m.lc."
round_prec_rh = 7
mean_used_id="m.u."

# TODO - B - remove useless code
class RunHistory:
    def __init__(self):
        self.it_hist = []

    def add_it_hist(self, data_points: [CNDataPoint]):
        it_rec = []
        for i in range(len(data_points)):
            dp = data_points[i].copy()
            dp.net.weights = np.zeros((0, 0))
            dp.net.links = np.zeros((0, 0))
            dp.net.biases = np.zeros((0, 0))
            it_rec.append(dp)
        self.it_hist.append(it_rec)

    # def get_it_best(self, iteration: int) -> CNDataPoint:
    #     it = self.it_hist[iteration]
    #
    #     best_eval = None
    #     for i in range(len(it)):
    #         eval = it[i]
    #         if best_eval is None or eval.ff > best_eval.ff:
    #             best_eval = eval
    #
    #     return best_eval

    # def get_it_summary_string(self, iteration: int):
    #     evals = self.it_hist[iteration]
    #
    #     mean_ff = mean([eval.ff for eval in evals])
    #     mean_acc = mean([eval.acc for eval in evals])
    #     mean_prec = mean([eval.prec for eval in evals])
    #     mean_rec = mean([eval.rec for eval in evals])
    #     mean_eff = mean([eval.get_eff() for eval in evals])
    #     mean_size = mean([eval.net.size() for eval in evals])
    #     mean_used = mean([eval.net.get_number_of_used_neurons() for eval in evals])
    #
    #     result = f"{iteration} |{mean_ff_id}:{round(mean_ff, round_prec_rh)}|" + \
    #              f"{mean_acc_id}:{round(mean_acc, round_prec_rh)}|" + \
    #              f"{mean_prec_id}:{round(mean_prec, round_prec_rh)}|{mean_rec_id}:{round(mean_rec, round_prec_rh)}|" + \
    #              f"{mean_size_id}:{round(mean_size,round_prec_rh)}|{mean_eff_id}:{round(mean_eff, round_prec_rh)}|" \
    #              f"{mean_used_id}:{round(mean_used, round_prec_rh)}"
    #
    #     return result


    def to_csv_file(self, fpath: str, reg: bool):
        file = open(fpath, "w")
        file.write("it,rk,is,os,nc,ec,af,ag,ni,mr,sqrp,linp,pmp,cp,dstp,afp,ff")
        if not reg:
            file.write(",eff,meff,acc,prc,rec,f1s")
        if self.it_hist[0][0].net.output_size == 2:
            file.write(",tp,fn,fp,tn")

        file.write("\n")

        for it in range(len(self.it_hist)):
            it_nets = self.it_hist[it]
            for rk in range(len(it_nets)):
                cndatapoint = it_nets[rk]
                net = cndatapoint.net
                file.write(f"{it + 1},{rk + 1},{net.input_size},{net.output_size},{net.neuron_count},"
                           f"{net.get_edge_count()},"
                           f"{net.get_act_fun_string()},{net.aggrFun.to_string()},{net.net_it},"
                           f"{net.mutation_radius},{net.sqr_mut_prob},{net.lin_mut_prob},"
                           f"{net.p_mutation_prob},{net.c_prob},{net.dstr_mut_prob},{net.act_mut_prob},{cndatapoint.ff}")
                if not reg:
                    file.write(f",{cndatapoint.get_eff()}")
                    file.write(f",{cndatapoint.get_meff()}")
                    file.write(f",{cndatapoint.get_acc()}")
                    file.write(f",{cndatapoint.get_avg_prec()}")
                    file.write(f",{cndatapoint.get_avg_rec()}")
                    file.write(f",{cndatapoint.get_avg_f1()}")
                if self.it_hist[0][0].net.output_size == 2:
                    file.write(f",{cndatapoint.get_TP_perc()}")
                    file.write(f",{cndatapoint.get_FN_perc()}")
                    file.write(f",{cndatapoint.get_FP_perc()}")
                    file.write(f",{cndatapoint.get_TN_perc()}")
                file.write("\n")
        file.close()





