# 2-layer bipartite graph
# only one cluster

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
plt.rcParams['figure.figsize'] = 12, 5


# dt, eps, eps_str = 4, 0.5, '0pt5'
dt, eps, eps_str = 6, 0.5, '0pt5'
degree_dist = 'binomial_binomial'
varsigma, tau = 1e7, 1e7

filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
varsigma, tau = 2e5, 8e5       ## dt, eps, eps_str = 6, 0.5, '0pt5'
start_idx, end_idx = 20, 99
mult_fact = 6


flops_slearnt_dict = np.load(filename+'.npy', allow_pickle='TRUE').item()

flops_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[start_idx:end_idx]
flops_vec_adjusted = flops_vec*varsigma*tau*mult_fact
PB_vec = (1-np.array([flops_slearnt_dict[str(flops_vec[x])]['s_learnt'] for x in range(len(flops_vec))])/np.array([flops_slearnt_dict[str(flops_vec[x])]['s_opt'] for x in range(len(flops_vec))]))
epsBP_vec = np.array([flops_slearnt_dict[str(flops_vec[x])]['eps_BP'] for x in range(len(flops_vec))])
s_opt_vec = np.array([flops_slearnt_dict[str(flops_vec[x])]['s_opt'] for x in range(len(flops_vec))])
s_learnt_opt_vec = np.array([flops_slearnt_dict[str(flops_vec[x])]['s_learnt'] for x in range(len(flops_vec))])
t_opt_vec = flops_vec/np.array([flops_slearnt_dict[str(flops_vec[x])]['s_opt'] for x in range(len(flops_vec))])
alpha_vec = np.array([flops_slearnt_dict[str(flops_vec[x])]['alpha'] for x in range(len(flops_vec))])

D_opt_vec = t_opt_vec*tau
N_opt_vec = s_opt_vec*varsigma 

###################################################################
####################### Training loss scaling #####################
###################################################################

p_vec = dt/s_opt_vec
R_vec = s_opt_vec
del_vec = PB_vec


term_1a = 1
term_1b = (p_vec*(1-del_vec)+(1-p_vec))**R_vec
term_1c = del_vec*p_vec*R_vec*((p_vec*(1-del_vec)+(1-p_vec))**(R_vec-1))

term_2a = (1-p_vec)**R_vec + R_vec*p_vec*(1-p_vec)**(R_vec-1)
term_2b = (1-p_vec)**R_vec + R_vec*(1-del_vec)*p_vec*(1-p_vec)**(R_vec-1)
term_2c = del_vec*p_vec*R_vec*((1-p_vec)**(R_vec-1))

# train_err = (term_1a-term_2a)-(term_1b-term_2b)-(term_1c-term_2c) # correct
train_err = 1-((1-p_vec*del_vec)**(R_vec-1))*(1+p_vec*R_vec*del_vec) # correct, simplified
train_err_approx = 4*(dt*del_vec)**2

excess_entropy_lb = 0.5*(train_err**2)
# excess_entropy_lb_approx = -np.log(1-train_err_approx**2)


###################################################################
####################### Emergence & Plateau #######################
###################################################################

from scipy.special import comb
import sympy as smp
import scipy.special as scpspl

S, l, T, d, a, G, p = smp.symbols("S l T d a G p")

# p = (d*T/(S**2))**l
k = smp.simplify(S*p) # average degree of G_2^{(l)}

expr = G - (1-smp.exp(-a*G))
roots = smp.solve(expr, G)
G_sol = roots[0].subs(a, k)

gcc_ratio = smp.lambdify([p, S], G_sol)

lmd = 1
# Sl_vec = np.array([1e2, 1e2, 1e2])
# Sl_mix = np.array([0.6, 0.2, 0.2])
# num_Sl_vec = np.array([2, 4, 8])
# eta_l_vec = np.array([1, 10, 20])
# sigma_l_vec = np.array([0, 10, 30])

# Sl_vec = np.array([1e2, 1e2, 1e2, 1e2])
# Sl_mix = np.array([0.4, 0.2, 0.2, 0.2])
# num_Sl_vec = np.array([2, 4, 8, 16])
# eta_l_vec = np.array([1, 10, 20, 30])
# sigma_l_vec = np.array([0, 2, 4, 8])


# ## Smooth emergence
# L = 100
# Sl_vec = 1e3*np.ones(L)
# # Sl_mix = np.random.rand(L)
# x = np.linspace(-1, 1, L)
# Sl_mix = norm(loc=0, scale=0.5).pdf(x)
# Sl_mix = Sl_mix/np.sum(Sl_mix)
# num_Sl_vec = 100*np.ones(L)
# # eta_l_vec = np.round(np.linspace(1,10,L))
# eta_l_vec = np.round(np.exp(np.sqrt(np.linspace(1,10,L))))
# sigma_l_vec = np.round(np.linspace(0,10,L))

## Smooth emergence


## Plateau
L = 100
Sl_vec = 1e3*np.ones(L)
# Sl_mix = np.random.rand(L)
x = np.linspace(-1, 1, L)
y = norm(loc=-1, scale=0.1).pdf(x)
z = norm(loc=0.7, scale=0.1).pdf(x)
Sl_mix = (z+y)/np.sum(z+y)
num_Sl_vec = 100*np.ones(L)
# eta_l_vec = np.round(np.linspace(1,10,L))
eta_l_vec = np.round(np.exp(np.sqrt(np.linspace(1,10,L))))
sigma_l_vec = np.round(np.linspace(0,10,L))


gcc_ratio_avg = np.zeros(t_opt_vec.shape)
prereq_factor = np.ones(t_opt_vec.shape)

subtaskidx = 1

for ind_mix, mix_wt in enumerate(Sl_mix):
    Sl = Sl_vec[ind_mix]
    num_Sl = num_Sl_vec[ind_mix] # num skills reqd for subtask
    eta_l = eta_l_vec[ind_mix]
    Rc2 = s_opt_vec*(s_opt_vec-1)/2

    pss_vec = (1/Sl**2)*(1-(1-(dt/s_opt_vec)**2)**(t_opt_vec)) 
    # pss_vec = (1/Sl**2)*((dt/s_opt_vec)**2)*(t_opt_vec)
    
    # Chernoff bound
    term1_pl = (1-np.exp(-Rc2*KL_div(eta_l/Rc2, pss_vec)))*(eta_l/Rc2 < pss_vec) + (1/np.sqrt(2*Rc2))*np.exp(-Rc2*KL_div(eta_l/Rc2, pss_vec))*(eta_l/Rc2 > pss_vec) 
    term2_pl = prereq_factor**(2*sigma_l_vec[ind_mix]) #  factor of 2 - one for s_1 and another for s_2    
    pl_vec = term1_pl*term2_pl

    # >>> for debug
    # pl_vec = pss_vec

    # # ## Chebyshev # does not work
    # var_val = Rc2*pss_vec*(1-pss_vec)
    # mean_val = pss_vec*Rc2
    # bound_val = var_val/(eta_l-mean_val)**2
    # pl_vec = (1-bound_val)*(eta_l < mean_val) + (1-bound_val/(1+bound_val))*(eta_l > mean_val)
    # # pl_vec = (1-bound_val)*(eta_l < mean_val) + (bound_val)*(eta_l > mean_val)
    # # pl_vec = (bound_val)*(eta_l < mean_val) + (1-bound_val)*(eta_l > mean_val)

    gcc_ratio_vec = np.array([gcc_ratio(pl_vec[ind], Sl) for ind in range(len(pl_vec))])
    gcc_ratio_vec = np.nan_to_num(gcc_ratio_vec)
    gcc_ratio_avg = gcc_ratio_avg+mix_wt*gcc_ratio_vec**num_Sl
    if ind_mix == subtaskidx:
        gcc_ratio_onesubtask = gcc_ratio_vec**num_Sl
    
    prereq_factor = gcc_ratio_vec


###################################################################
############################ PLOTS ################################
###################################################################
plt.figure(0)

###################################################################
######################## R^*, T^* vs FLOPs ########################
###################################################################
plt.subplot(2,2,1)
plt.title("(a) Compute optimal size-scaling")

plt.plot(flops_vec_adjusted, N_opt_vec, 'r-', label="$N^*$ (model size)")
plt.plot(flops_vec_adjusted, D_opt_vec, 'b-', label="$D^*$ (dataset size)")
plt.ylabel("$N^*$ or $D^*$")
plt.plot(5.73*1e23, 63*1e9, "r*", markersize=10)
plt.plot(5.73*1e23, 1.4*1e12, "b*", markersize=10)

plt.xlabel("FLOPs ($C$)")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()

###################################################################
############################ P_B vs R* ############################
###################################################################
plt.subplot(2,2,2)
plt.title("(b) Excess entropy scaling")
plt.plot(N_opt_vec, excess_entropy_lb, 'b--', label="$\\frac{1}{2} P_{e, train}^2 \leq D_{KL}(p || q)$", linewidth=2) # good
# plt.plot(N_opt_vec, excess_entropy_lb_approx, 'r--', label="approx", linewidth=2) # approx

plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Excess entropy lower bound")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

###################################################################
###################################################################
###################################################################
plt.figure(1)

plt.subplot(2,2,1)
plt.title("(a) Emergence")
plt.plot(N_opt_vec, gcc_ratio_onesubtask, linewidth=2, label="$\gamma^{n_l}$")
plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.legend()
plt.grid()

plt.subplot(2,2,2)
plt.title("(b) Skill-level distribution")
plt.plot(N_opt_vec, gcc_ratio_onesubtask, linewidth=2, label="$\gamma^{n_l}$")
plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.legend()
plt.grid()

###################################################################
############################ P_B vs R* ############################
###################################################################
plt.subplot(2,2,4)
plt.title("(d) Plateauing and multiple emergences")
# plt.plot(N_opt_vec, (gcc_ratio_avg), "-o", linewidth=2, label="$\sum_{l} q_{n_l, l} \gamma^{n_l}_l$")
plt.plot(N_opt_vec, (gcc_ratio_avg), linewidth=2, label="$\sum_{l} q_{n_l, l} \gamma^{n_l}_l$")

plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Accuracy")
plt.legend()
plt.xscale("log")
plt.grid()

###################################################################

plt.tight_layout()

plt.savefig("size_loss_scaling_emerg_plateau.pdf", format='pdf')

plt.show(block=False)

brkpnt1 = 1
