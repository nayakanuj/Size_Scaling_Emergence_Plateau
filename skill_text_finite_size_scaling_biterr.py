# main script
# single (lower) layer bipartite graph
# stores data for plotting later

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import *
import scipy as scp
from modules_core import *
from modules_dbg import *

FONT_SIZE = 12
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.size"] = str(FONT_SIZE)
plt.rcParams['figure.figsize'] = 8, 6
NUM_PTS = 40

if __name__ == "__main__":
    
    # dt, eps, eps_str = 4, 0.9, '0pt9'
    #dt, eps, eps_str = 2, 0.9, '0pt9'
    #dt, eps, eps_str = 3, 0.5, '0pt5'
    # dt, eps, eps_str = 4, 0.8, '0pt8'
    # dt, eps, eps_str = 4, 0.3, '0pt3'
    # dt, eps, eps_str = 6, 0.7, '0pt7'
    # dt, eps, eps_str = 4, 0.5, '0pt5'
    dt, eps, eps_str = 6, 0.5, '0pt5'
    # dt, eps, eps_str = 7, 0.5, '0pt5'
    # dt, eps, eps_str = 10, 0.5, '0pt5'
    # dt, eps, eps_str = 14, 0.5, '0pt5'
    # dt, eps, eps_str = 8, 0.5, '0pt5'
    # dt, eps, eps_str = 8, 0.5, '0pt5'
    degree_dist = 'binomial_binomial'
    
    ####### R sweep - range ########
    # min_scale = 0.2 # dt, eps, eps_str = 4, 0.5, '0pt5'
    # max_scale = 0.9

    min_scale = 0.2 # dt, eps, eps_str = 6, 0.5, '0pt5'
    max_scale = 0.6

    # min_scale = 0.1 # dt, eps, eps_str = 7, 0.5, '0pt5'
    # max_scale = 0.5

    # min_scale = 0.01 # dt, eps, eps_str = 10, 0.5, '0pt5'
    # max_scale = 0.2   

    # min_scale = 0.1
    # max_scale = 2

    ####### numFLOPs #########
    # min_FLOPS = 1e2
    # max_FLOPS = 1e26
    # num_FLOPS = 40

    # min_FLOPS = 1e2
    # max_FLOPS = 1e28
    # num_FLOPS = 40

    # min_FLOPS = 1e2
    # max_FLOPS = 1e12
    # num_FLOPS = 50

    # min_FLOPS = 1e2
    # max_FLOPS = 1e14
    # num_FLOPS = 10

    # min_FLOPS = 1e2
    # max_FLOPS = 1e14
    # num_FLOPS = 50

    min_FLOPS = 1e2
    max_FLOPS = 1e20
    num_FLOPS = 200

    # ## for Iso-FLOP curves
    # min_FLOPS = 1e4
    # max_FLOPS = 1e8
    # num_FLOPS = 10

    # min_FLOPS = 1e2
    # max_FLOPS = 1e16
    # num_FLOPS = 10

    # min_FLOPS = 1e4 # 1e6 is better
    # max_FLOPS = 1e14
    # num_FLOPS = 10

    # min_FLOPS = 1e2
    # max_FLOPS = 1e26
    # num_FLOPS = 20

    # min_FLOPS = 1e4
    # max_FLOPS = 1e26
    # num_FLOPS = 40

    # min_FLOPS = 1e5
    # max_FLOPS = 1e14
    # num_FLOPS = 20

    # min_FLOPS = 1e2
    # max_FLOPS = 1e12
    # num_FLOPS = 40

    # min_FLOPS = 1e2
    # max_FLOPS = 1e12
    # num_FLOPS = 8
    
    # min_FLOPS = 1e5
    # max_FLOPS = 1e6
    # num_FLOPS = 5
    # min_scale = 0.4
    # max_scale = 0.9

    FLOPS_vec = np.logspace(np.log10(min_FLOPS), np.log10(max_FLOPS), num_FLOPS)    

    s_opt_vec = np.zeros(FLOPS_vec.shape)
    eps_BP_vec = np.zeros(FLOPS_vec.shape)
    x_BP_vec = np.zeros(FLOPS_vec.shape)
    s_learnt_vec = np.zeros(FLOPS_vec.shape)
    alpha_vec = np.zeros(FLOPS_vec.shape)

    for ind_FLOPS, FLOPS in enumerate(FLOPS_vec):
        s_low = max(1.0, np.sqrt(FLOPS)*min_scale)
        s_high = np.sqrt(FLOPS)*max_scale

        # plot_slearnt_vs_s_biterr(s_low, s_high, FLOPS, eps, dt, alpha)        

        args = (dt, FLOPS, eps)
        s_opt = my_minimizer(num_skills_learnt_biterr, args, s_low, s_high, num_pts=NUM_PTS)

        s_learnt, eps_BP, x_BP, _, _, _, alpha_val = num_skills_learnt_biterr(s_opt, dt, FLOPS, eps, optimize_flag=False)
        # print(f"FLOPS={FLOPS}, sqrt(FLOPS)={np.sqrt(FLOPS)}, s_opt={s_opt}, s_learnt={s_learnt}, eps_BP = {eps_BP}, x_BP = {x_BP}")        
        print(f"{ind_FLOPS}/{num_FLOPS}, FLOPS={FLOPS}, sqrt(FLOPS)={np.sqrt(FLOPS)}, s_opt={s_opt}, s_learnt={s_learnt}, eps_BP = {eps_BP}, alpha = {alpha_val}")

        # # # >>>> for debug - begin
        # t = FLOPS/s_opt
        # sbar = s_opt*(1-eps)/eps
        # n = sbar+s_opt
        # nv, nc = t, n # these are not number of variable nodes and check nodes. These are n's of binomial distribution
        # ds = dt*t/s_opt
        # dtbar = ds*n/t  
        # pv, pc = ds/nv, dtbar/nc
        # plot_fx_vs_x_dbg(eps_BP, nv, nc, pv, pc, marker="-.")
        # brkpnt1 = 1
        # # # >>>> for debug - end

        s_opt_vec[ind_FLOPS] = s_opt
        s_learnt_vec[ind_FLOPS] = s_learnt
        eps_BP_vec[ind_FLOPS] = eps_BP
        x_BP_vec[ind_FLOPS] = x_BP
        alpha_vec[ind_FLOPS] = alpha_val

        # plot_slearnt_vs_s(s_low, s_high, FLOPS, eps, dt)
        # plt.subplot(1,2,1)
        # plt.plot(s_opt, s_learnt, "ko")
        # plt.subplot(1,2,2)
        # plt.plot(s_opt, eps_BP, "ko")
        # # plot_s_t_vs_flops(FLOPS_vec, s_opt_vec, s_learnt_vec)
        # plt.show(block=False)

        brkpnt1 = 1
        
    # plot_slearnt_vs_s_postproc(s_opt_vec, s_learnt_vec, eps_BP_vec)

    ## dump data
    flops_slearnt_dict = {} # saving for the first time
    for ind_FLOPS, FLOPS in enumerate(FLOPS_vec):
        # flops_slearnt_dict[str(FLOPS)] = {'s_opt':s_opt_vec[ind_FLOPS], 's_learnt':s_learnt_vec[ind_FLOPS], 'eps_BP':eps_BP_vec[ind_FLOPS]}    
        flops_slearnt_dict[str(FLOPS)] = {'s_opt':s_opt_vec[ind_FLOPS], 's_learnt':s_learnt_vec[ind_FLOPS], 'eps_BP':eps_BP_vec[ind_FLOPS], 'x_BP':x_BP_vec[ind_FLOPS], 'alpha':alpha_vec[ind_FLOPS]}
    
    np.save(degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts'+str(num_FLOPS)+'.npy', flops_slearnt_dict)

    # plot_s_t_vs_flops(FLOPS_vec, s_opt_vec, s_learnt_vec)

    # brkpnt1 = 1
    