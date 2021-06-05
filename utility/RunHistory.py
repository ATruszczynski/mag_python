from statistics import mean

from utility.AnnDataPoint import AnnDataPoint


mean_ff_id = "m.f."
mean_acc_id = "m.a."
mean_prec_id = "m.p."
mean_rec_id = "m.r."
mean_eff_id = "m.e."
mean_size_id = "m.s."
round_prec_rh = 3

class RunHistory:
    def __init__(self):
        self.it_hist = []

    def add_it_hist(self, data_points: [AnnDataPoint]):
        self.it_hist.append(data_points)

    def get_it_best(self, iteration: int) -> AnnDataPoint:
        it = self.it_hist[iteration]

        best_eval = None
        for i in range(len(it)):
            eval = it[i]
            if best_eval is None or eval.ff > best_eval.ff:
                best_eval = eval

        return best_eval

    def get_it_summary_string(self, iteration: int):
        evals = self.it_hist[iteration]

        mean_ff = mean([eval.ff for eval in evals])
        mean_acc = mean([eval.acc for eval in evals])
        mean_prec = mean([eval.prec for eval in evals])
        mean_rec = mean([eval.rec for eval in evals])
        mean_eff = mean([eval.get_eff() for eval in evals])
        mean_size = mean([eval.point.size() for eval in evals])

        result = f"{iteration} |{mean_ff_id}:{round(mean_ff, round_prec_rh)}|" + \
                 f"{mean_acc_id}:{round(mean_acc, round_prec_rh)}|" + \
                 f"{mean_prec_id}:{round(mean_prec, round_prec_rh)}|{mean_rec_id}:{round(mean_rec, round_prec_rh)}|" + \
                 f"{mean_size_id}:{round(mean_size,round_prec_rh)}|{mean_eff_id}:{round(mean_eff, round_prec_rh)}"

        return result

