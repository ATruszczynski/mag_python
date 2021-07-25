from statistics import mean

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

class RunHistory:
    def __init__(self):
        self.it_hist = []

    def add_it_hist(self, data_points: [CNDataPoint]): #TODO test
        self.it_hist.append(data_points)

    def get_it_best(self, iteration: int) -> CNDataPoint: #TODO test
        it = self.it_hist[iteration]

        best_eval = None
        for i in range(len(it)):
            eval = it[i]
            if best_eval is None or eval.ff > best_eval.ff:
                best_eval = eval

        return best_eval

    def get_it_summary_string(self, iteration: int): #TODO test
        evals = self.it_hist[iteration]

        mean_ff = mean([eval.ff for eval in evals])
        mean_acc = mean([eval.acc for eval in evals])
        mean_prec = mean([eval.prec for eval in evals])
        mean_rec = mean([eval.rec for eval in evals])
        mean_eff = mean([eval.get_eff() for eval in evals])
        mean_size = mean([eval.net.size() for eval in evals])
        mean_used = mean([eval.net.get_number_of_used_neurons() for eval in evals])

        result = f"{iteration} |{mean_ff_id}:{round(mean_ff, round_prec_rh)}|" + \
                 f"{mean_acc_id}:{round(mean_acc, round_prec_rh)}|" + \
                 f"{mean_prec_id}:{round(mean_prec, round_prec_rh)}|{mean_rec_id}:{round(mean_rec, round_prec_rh)}|" + \
                 f"{mean_size_id}:{round(mean_size,round_prec_rh)}|{mean_eff_id}:{round(mean_eff, round_prec_rh)}|" \
                 f"{mean_used_id}:{round(mean_used, round_prec_rh)}"

        return result

    def summary_dict(self, iteration: int): #TODO test
        evals = self.it_hist[iteration]

        mean_ff = mean([eval.ff for eval in evals])
        mean_acc = mean([eval.acc for eval in evals])
        mean_prec = mean([eval.prec for eval in evals])
        mean_rec = mean([eval.rec for eval in evals])
        mean_eff = mean([eval.get_eff() for eval in evals])
        mean_size = mean([eval.net.size() for eval in evals])
        mean_lc = mean([len(eval.net.neuronCounts) - 1 for eval in evals])

        res_dict = {}
        res_dict[mean_ff_id] = mean_ff
        res_dict[mean_acc_id] = mean_acc
        res_dict[mean_prec_id] = mean_prec
        res_dict[mean_rec_id] = mean_rec
        res_dict[mean_eff_id] = mean_eff
        res_dict[mean_f1_id] = -666
        res_dict[mean_size_id] = mean_size
        res_dict[mean_lc_id] = mean_lc

        return res_dict

