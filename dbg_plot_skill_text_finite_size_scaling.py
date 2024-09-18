# debug script - plots s_learnt vs s
# lower layer bipartite graph only
# does not store data

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


if __name__ == "__main__":
    
    # dt, eps, eps_str = 4, 0.9, '0pt9'
    #dt, eps, eps_str = 2, 0.9, '0pt9'
    #dt, eps, eps_str = 3, 0.5, '0pt5'
    # dt, eps, eps_str = 4, 0.8, '0pt8'
    # dt, eps, eps_str = 4, 0.5, '0pt5'
    dt, eps, eps_str = 6, 0.5, '0pt5'
    # dt, eps, eps_str = 7, 0.5, '0pt5'
    # dt, eps, eps_str = 10, 0.5, '0pt5'
    degree_dist = 'binomial_binomial'
    closed_form = False
    mult_fact = 6
    
    # varsigma, tau = 1e7, 1e7
    varsigma, tau = 2e5, 8e5       ## dt, eps, eps_str = 6, 0.5, '0pt5'
    
    # min_scale = 0.5
    # max_scale = 0.8

    # min_scale = 0.1
    # max_scale = 0.9

    # min_scale = 0.1
    # max_scale = 0.9
    # min_scale = 0.001
    
    # min_scale = 0.02
    # max_scale = 0.5
     
    min_scale = 0.15 # dt, eps, eps_str = 6, 0.5, '0pt5'
    max_scale = 0.49    


    if closed_form:
        filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50_closed_form'    
    else:
        # filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50' 
        filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
    
    flops_slearnt_dict = np.load(filename+'.npy', allow_pickle='TRUE').item()

    # min_FLOPS = 1e5
    # max_FLOPS = 1e6
    # num_FLOPS = 5
    # # FLOPS_vec = np.logspace(np.log10(min_FLOPS), np.log10(max_FLOPS), num_FLOPS)    
    # FLOPS_vec = np.linspace(min_FLOPS, max_FLOPS, num_FLOPS)    

    # FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[10:40:4]
    # FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[10::4]
    # FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[15::5]
    FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[30:60:5]
    FLOPS_vec_scaled = FLOPS_vec*varsigma*tau*mult_fact  

    for ind_FLOPS, FLOPS in enumerate(FLOPS_vec):
        s_low = max(1.0, np.sqrt(FLOPS)*min_scale)
        s_high = np.sqrt(FLOPS)*max_scale        
        # args = (dt, FLOPS, eps)
        # s_opt = my_minimizer(num_skills_learnt, args, s_low, s_high, num_pts=30)
        print(f"ind_FLOPs={ind_FLOPS}, FLOPs = {FLOPS}")
        
        if closed_form:
            # biterr [TODO]
            plot_slearnt_vs_s_closed_form(s_low, s_high, FLOPS, eps, dt, fig_num=0)
        else:
            plot_slearnt_vs_s_biterr(s_low, s_high, FLOPS, eps, dt, fig_num=0)

        brkpnt1 = 1
            
    
    PB_vec = (1-np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_learnt'] for x in range(len(FLOPS_vec))])/np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_opt'] for x in range(len(FLOPS_vec))]))
    epsBP_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['eps_BP'] for x in range(len(FLOPS_vec))])
    s_opt_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_opt'] for x in range(len(FLOPS_vec))])
    s_learnt_opt_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_learnt'] for x in range(len(FLOPS_vec))])
    t_opt_vec = FLOPS_vec/np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_opt'] for x in range(len(FLOPS_vec))])
    alpha_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['alpha'] for x in range(len(FLOPS_vec))])

    D_opt_vec = t_opt_vec*tau
    N_opt_vec = s_opt_vec*varsigma

    plot_slearnt_vs_s_postproc(s_opt_vec, s_learnt_opt_vec, epsBP_vec)

    # plt.figure(1)
    # plt.plot(s_opt_vec, epsBP_vec - eps)
    # plt.xlabel("$R^*$")
    # plt.ylabel("$\epsilon^* - \epsilon$")
    # plt.xscale("log")
    # plt.yscale("log")

    brkpnt1 = 1

        

    
  