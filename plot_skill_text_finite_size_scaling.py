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
# dt, eps, eps_str = 7, 0.5, '0pt5'
# dt, eps, eps_str = 10, 0.5, '0pt5'
# dt, eps, eps_str = 14, 0.5, '0pt5'
# dt, eps, eps_str = 100, 0.5, '0pt5'
# dt, eps, eps_str = 20, 0.5, '0pt5'
# dt, eps, eps_str = 4, 0.8, '0pt8'
# dt, eps, eps_str = 6, 0.7, '0pt7'
# dt, eps, eps_str = 4, 0.3, '0pt3'
# dt, eps, eps_str = 8, 0.5, '0pt5'
degree_dist = 'binomial_binomial'
varsigma, tau = 1e7, 1e7
# varsigma, tau = 1, 1

# filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
# varsigma, tau = 1e5, 2e5       ## dt, eps, eps_str = 7, 0.5, '0pt5'
# start_idx, end_idx = 40, 90

filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
# varsigma, tau = 1e5, 5e5       ## dt, eps, eps_str = 6, 0.5, '0pt5'
varsigma, tau = 2e5, 8e5       ## dt, eps, eps_str = 6, 0.5, '0pt5'
start_idx, end_idx = 20, 99

# filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
# # varsigma, tau = 3e6, 8e5       ## dt, eps, eps_str = 10, 0.5, '0pt5'
# varsigma, tau = 1e5, 5e4       ## dt, eps, eps_str = 10, 0.5, '0pt5'
# start_idx, end_idx = 20, 99

# filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
# varsigma, tau = 1e5, 2e5       ## dt, eps, eps_str = 4, 0.5, '0pt5'
# start_idx, end_idx = 40, 99

# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50'
# varsigma, tau = 1e7, 2e7

# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts100_closed_form'
# varsigma, tau = 4e6, 1e8

# filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
# varsigma, tau = 1e7, 6e7

mult_fact = 6

# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts5'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts10'
# filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts10'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts20'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts20_alphaconst'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts20_minscalar'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts20_minscalar_1e26' # dt, eps, eps_str = 4, 0.5, '0pt5'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts40_nflops1e28' # dt, eps, eps_str = 14, 0.5, '0pt5'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts40_nflops1e30' # dt, eps, eps_str = 7 or 4, 0.5, '0pt5'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50_minscalar'
# filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50_closed_form'
# filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
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

# curve_fit = s_opt_vec**(-0.5)/(10**(2.6))
# curve_fit = s_opt_vec**(-1)/(10**(5.2))

###################################################################
####################### Emergence & Plateau #######################
###################################################################

from scipy.special import comb
import sympy as smp
import scipy.special as scpspl

S, l, T, d, a, G = smp.symbols("S l T d a G")

p = (d*T/(S**2))**l
k = smp.simplify(S*p)

expr = G - (1-smp.exp(-a*G))
roots = smp.solve(expr, G)
G_sol = roots[0].subs(a, k)

gcc_ratio = smp.lambdify([T, S, d, l], G_sol)

# Sl, lmd = 10000, 0.7
# Sl, lmd = 1e2, 1
# lmd = 1
# Sl_vec = np.array([1e2, 2e3, 1e5])
# Sl_mix = np.array([0.6, 0.2, 0.2])
# num_Sl_vec = np.array([2, 4, 8])

lmd = 1
Sl_vec = np.array([1e4, 2e5, 1e7])
Sl_mix = np.array([0.6, 0.2, 0.2])
num_Sl_vec = np.array([2, 4, 8])

# Sl_vec = np.array([1e3])
# Sl_mix = np.array([1])
# num_Sl_vec = np.array([8])

# Sl, lmd = 1e3, 1
gcc_ratio_avg = np.zeros(t_opt_vec.shape)

subtaskidx = 1
gcc_ratio_onesubtask = np.array([gcc_ratio(TT, Sl_vec[subtaskidx], dt, lmd) for TT in t_opt_vec])

for ind_mix, mix_wt in enumerate(Sl_mix):
    Sl = Sl_vec[ind_mix]
    num_Sl = num_Sl_vec[ind_mix]
    gcc_ratio_vec = np.array([gcc_ratio(TT, Sl, dt, lmd) for TT in t_opt_vec])
    gcc_ratio_avg = gcc_ratio_avg+mix_wt*gcc_ratio_vec**num_Sl



# # max diameter of a random graph
# p_vec = (dt*t_opt_vec/(Sl**2))**lmd
# k_vec = p_vec*Sl
# d_max_vec = np.log(Sl)/np.log(k_vec) + 2*np.log(Sl)/np.log(-scpspl.lambertw(k_vec*np.exp(-k_vec)))


###################################################################
plt.figure(0)

###################################################################
######################## R^*, T^* vs FLOPs ########################
###################################################################
plt.subplot(2,2,1)
plt.title("Compute optimal size-scaling")
# plt.plot(flops_vec_adjusted, s_opt_vec, 'r-', label="$R^*$ (sub-skills)")
# plt.plot(flops_vec_adjusted, s_learnt_opt_vec, 'm-', label="$R^* (1-P_{B})$")
# plt.plot(flops_vec_adjusted, t_opt_vec, 'b-', label="$T^*$ (text pieces)")
# plt.ylabel("$R$ or $T$")

# plt.plot(flops_vec_adjusted, D_opt_vec/N_opt_vec, label="$N^*/D^*$")
# plt.ylabel("$N/D$")

plt.plot(flops_vec_adjusted, N_opt_vec, 'r-', label="$N^*$ (model size)")
# plt.plot(flops_vec_adjusted, s_learnt_opt_vec, 'm-', label="$R^* (1-P_{B})$")
plt.plot(flops_vec_adjusted, D_opt_vec, 'b-', label="$D^*$ (dataset size)")
plt.ylabel("$N$ or $D$")
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

# qmin = 1/1e100
# plt.plot(s_opt_vec, PB_vec*np.log(1+(PB_vec/qmin)), label="$-log(1-P_{B}^2)$", linewidth=2)

# # interpretation?
# plt.title(f"$P_B$ ($d_t=${dt}, $\epsilon=${eps}), Interpretation?")
# plt.plot(s_opt_vec, -np.log(1-PB_vec), label="$-\log(1-P_{B})$", linewidth=2) # good
# plt.plot(s_opt_vec, curve_fit, "r-.", label="$R^{* -0.5}$")

# 
# plt.title(f"$d_t=${dt}, $\epsilon=${eps}")
# plt.plot(s_opt_vec, -np.log(1-PB_vec**2), 'b-o', label="$-\log(1-P_{B}^2)$", linewidth=2) # good
# plt.plot(s_opt_vec, PB_vec, 'r-s', label="$P_{B}$", linewidth=2)
# plt.xlabel("$R^*$")
plt.title("Loss scaling")
plt.plot(N_opt_vec, -np.log(1-PB_vec**2), 'b--', label="$-\log(1-P_{B}^2) \leq D_{KL}(p || q)$", linewidth=2) # good
# plt.plot(N_opt_vec, PB_vec, 'r-.', label="$P_{B}$", linewidth=2)
plt.xlabel("$N^*$ (No. of parameters)")

# plt.plot(s_opt_vec, (epsBP_vec-eps), label="$\epsilon^*-\epsilon$")
# plt.plot(s_opt_vec, curve_fit, "r-.", label="$R^{* -1}$")


# plt.plot(s_opt_vec, PB_vec/(s_opt_vec), label="$1/(S(1-P_{B}))$") # good
# plt.plot(flops_vec_adjusted, epsBP_vec-eps, label="$\epsilon^*-\epsilon$")
# plt.plot(flops_vec_adjusted, (epsBP_vec-eps), label="$\epsilon^*-\epsilon$")
# plt.plot(s_opt_vec, (epsBP_vec-eps), label="$\epsilon^*-\epsilon$")
plt.plot()
# plt.xlabel("FLOPs")
# plt.ylabel("$P_B$ or $\epsilon^*$")
plt.ylabel("$-\log(1-P_{B}^2)$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

###################################################################
###################################################################
###################################################################

plt.subplot(2,2,3)
plt.title("Emergence")
plt.plot(N_opt_vec, gcc_ratio_onesubtask, linewidth=2, label="$\gamma^{n_l}$")
plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.legend()
plt.grid()

# plt.subplot(2,2,3)
# plt.plot(flops_vec_adjusted, t_opt_vec/s_opt_vec, linewidth=2)
# plt.xscale("log")
# plt.grid()

###################################################################
############################ P_B vs R* ############################
###################################################################
plt.subplot(2,2,4)
# plt.plot(s_opt_vec, gcc_ratio_vec*(1-PB_vec))
# plt.xlabel("$R^*$")
# plt.plot(N_opt_vec, (gcc_ratio_vec)*(1-PB_vec), label="$\\frac{S^{(l)}_G}{S^{(l)}} (1-P_B)$")
# plt.plot(N_opt_vec, gcc_ratio_vec, label="$\\frac{S^{(l)}_G}{S^{(l)}}$")

# >>>> for debug
# plt.plot(N_opt_vec, (gcc_ratio_vec)*(1-PB_vec), label="$\\frac{S^{(l)}_G}{S^{(l)}} (1-P_B)$")
# plt.plot(N_opt_vec, gcc_ratio_vec, label="$\\frac{S^{(l)}_G}{S^{(l)}}$")
plt.title("Plateauing and multiple emergences")
plt.plot(N_opt_vec, (gcc_ratio_avg), linewidth=2, label="$\sum_{l} q_{n_l, l} \gamma^{n_l}_l$")

plt.xlabel("$N^*$ (No. of parameters)")
# plt.ylabel("No. of nodes in giant\n connected component \n(normalized)")
plt.ylabel("Accuracy")
# plt.plot(t_opt_vec, p_vec)
plt.legend()
plt.xscale("log")
plt.grid()

###################################################################

# plt.suptitle(f"Compute optimal size scaling, plateauing and emergence ($d_t=${dt}, $\epsilon=${eps})")
plt.suptitle(f"Compute-optimal size scaling, plateauing and emergence")
plt.tight_layout()

# plt.savefig("finite_sz_scaling_and_emerg_ST_and_Q_vs_FLOPs.pdf", format='pdf')
plt.savefig("size_loss_scaling_emerg_plateau.pdf", format='pdf')

# plt.figure(1)
# # plt.plot(flops_vec_adjusted, alpha_vec)
# plt.plot(s_opt_vec, -np.log(1-PB_vec**2)*s_opt_vec)
# plt.xscale('log')
# # plt.yscale('log')


plt.show(block=False)


brkpnt1 = 1
