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


    def to_csv_file(self, fpath: str, reg: bool):
        file = open(fpath, "w")
        file.write("it,rk,is,os,nc,ec,af,ag,ni,mr,mult,ppm,prad,cp,swp,depr")
        for f in range(len(self.it_hist[0][0].ff)):
            file.write(f",ff{f + 1}")
        if not reg:
            file.write(",eff,meff,acc,prc,rec,f1s")
        if self.it_hist[0][0].net.output_size == 2:
            file.write(",tp,fn,fp,tn")

        file.write("\n")

        for it in range(len(self.it_hist)):
            it_nets = self.it_hist[it]

            for ffi in reversed(range(len(it_nets[0].ff))):
                it_nets = sorted(it_nets, key=lambda x: x.ff[ffi], reverse=True)

            for rk in range(len(it_nets)):
                cndatapoint = it_nets[rk]
                net = cndatapoint.net
                file.write(f"{it + 1},{rk + 1},{net.input_size},{net.output_size},{net.neuron_count},"
                           f"{net.edge_count},"
                           f"{net.get_act_fun_string()},{net.aggrFun.to_string()},{net.net_it},"
                           f"{net.mutation_radius},{net.multi},"
                           f"{net.p_prob},{net.p_rad},{net.c_prob},{net.swap_prob},{net.depr_2}")
                for f in range(len(cndatapoint.ff)):
                    file.write(f",{cndatapoint.ff[f]}")
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





