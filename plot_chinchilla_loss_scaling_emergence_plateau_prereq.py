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

# filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
# start_idx, end_idx = 30, 99
filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts200'
start_idx, end_idx, step_idx = 30, 170, 2

varsigma, tau = 2e5, 8e5       ## dt, eps, eps_str = 6, 0.5, '0pt5'
mult_fact = 6


flops_slearnt_dict = np.load(filename+'.npy', allow_pickle='TRUE').item()

flops_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[start_idx:end_idx:step_idx]
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
# train_err_approx = 4*(dt*del_vec)**2
train_err_approx = (dt*del_vec)**2

excess_entropy_lb = 0.5*(train_err**2)
# excess_entropy_lb_approx = -np.log(1-train_err_approx**2)

A = 406.4
B = 410.7
alpha_scale = 0.34
beta_scale = 0.28
chinchilla_xs_entropy_scaling = A/(N_opt_vec**alpha_scale) + B/(D_opt_vec**beta_scale)


# Fit a linear regression line to the data
x_vec = np.log(N_opt_vec[20:50])
y_vec = np.log(excess_entropy_lb[20:50])
slope, intercept, r_value, p_value, std_err = linregress(x_vec, y_vec)
slope_str = str(f"{slope:0.2f}")

excess_entropy_lb_fit = np.exp(np.log(N_opt_vec)*slope+intercept)

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

gcc_ratio_func = smp.lambdify([p, S], G_sol)

## Emergence - unimodal
L = 100
L_vec = np.arange(1, L+1, 1)
# Sl_vec = 1e3*np.ones(L)
Sl_vec = 1e3*np.ones(L)
min_nl, max_nl = 2, 8
q_nl_vec = np.ones(max_nl-min_nl)/(max_nl - min_nl)
nl_vec = np.arange(min_nl, max_nl+1, 1)
# eta_l_vec = np.round((L_vec*0.2)**2)
# eta_l_vec = np.round(np.log(L_vec)+1)
# eta_l_vec = np.round(np.exp(7*L_vec/L)) # works
eta_l_vec = np.round(np.exp(7*L_vec/L)) #
# eta_l_vec = np.round(7*L_vec)
sigma_l_vec = np.round(np.log2(L_vec))

# x = np.linspace(-1, 1, L)
# y = norm(loc=0.5, scale=0.3).pdf(x)
# q_l_vec = y/np.sum(y)
# q_l_vec_unimodal = q_l_vec.copy()

y = np.array([binom.pmf(a, L, 0.5) for a in L_vec])
# q_l_vec = y/np.sum(y)
q_l_vec = y
q_l_vec_unimodal = q_l_vec.copy()

gcc_ratio_avg_unimodal, gcc_ratio_1subtask = get_accuracy_curve(q_l_vec, q_nl_vec, nl_vec, Sl_vec, s_opt_vec, t_opt_vec, sigma_l_vec, eta_l_vec, dt, gcc_ratio_func)

######### Plateau = Emergence + multimodal #########
# x = np.linspace(-1, 1, L)
# y = norm(loc=-0.5, scale=0.05).pdf(x)
# z = norm(loc=0.5, scale=0.1).pdf(x)
# q_l_vec = (z+y)/np.sum(z+y)
# q_l_vec_multimodal = q_l_vec
# pmf_wt_vec = np.array([1/3, 1/3, 1/3])
# centroid_vec = np.array([0.3, 0.5, 0.9])

pmf_wt_vec = np.array([0.4, 0.4, 0.2])
centroid_vec = np.array([0.2, 0.6, 0.95])

pmf_mix_vec = np.zeros(L)
for ind_mix in range(len(pmf_wt_vec)):    
    this_pmf = np.array([binom.pmf(a, L, centroid_vec[ind_mix]) for a in L_vec])
    pmf_mix_vec = pmf_mix_vec+pmf_wt_vec[ind_mix]*this_pmf

# q_l_vec = pmf_mix_vec/np.sum(pmf_mix_vec)
q_l_vec = pmf_mix_vec
q_l_vec_multimodal = q_l_vec

gcc_ratio_avg_multi_modal, _ = get_accuracy_curve(q_l_vec, q_nl_vec, nl_vec, Sl_vec, s_opt_vec, t_opt_vec, sigma_l_vec, eta_l_vec, dt, gcc_ratio_func)


###################################################################
############################ PLOTS ################################
###################################################################
fig = plt.figure(0)
fig.set_size_inches(9, 3.5)

###################################################################
######################## R^*, T^* vs FLOPs ########################
###################################################################
plt.subplot(1,2,1)
plt.title("(a) Compute optimal size-scaling")

plt.axvline(x=5.73*1e23, color='#008080', linestyle='--')
plt.plot(flops_vec_adjusted, N_opt_vec, 'r-', label="$N^*$ (model size)")
plt.plot(flops_vec_adjusted, D_opt_vec, 'b-', label="$D^*$ (dataset size)")
plt.ylabel("$N^*$ or $D^*$")
plt.plot(5.73*1e23, 63*1e9, "r*", markersize=10)
plt.plot(5.73*1e23, 1.4*1e12, "b*", markersize=10)
plt.xlim((1e18, 1e30))
plt.ylim((5e7, 5e14))

plt.xlabel("FLOPs ($C$)")
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()

###################################################################
############################ P_B vs R* ############################
###################################################################
plt.subplot(1,2,2)
plt.title("(b) Excess entropy scaling")
# plt.plot(N_opt_vec, excess_entropy_lb, 'b-', label="$\\frac{1}{2} P_{e, train}^2 \leq D_{KL}(p || q)$", linewidth=2) # good
# plt.plot(N_opt_vec, excess_entropy_lb_fit, 'r--', label="$N^{"+slope_str+"}$", linewidth=1)
plt.plot(N_opt_vec, excess_entropy_lb, 'b-', label="Lower bound ($\\frac{1}{2} P_{e, train}^2$)", linewidth=2) # good
plt.plot(N_opt_vec, chinchilla_xs_entropy_scaling, 'm-.', label="Empirical", linewidth=2)


# plt.plot(N_opt_vec, excess_entropy_lb_approx, 'r--', label="approx", linewidth=2) # approx

plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Excess entropy")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("chinchilla_loss_scaling.svg", format='svg')

###################################################################
###################################################################
###################################################################
plt.figure(1)

plt.subplot(2,2,1)
plt.title("(a) Emergence (homogeneous)")
plt.plot(N_opt_vec, gcc_ratio_1subtask, linewidth=2)
plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.grid()

plt.subplot(2,2,2)
plt.title("(b) Skill-level distribution")
# plt.plot(L_vec, q_l_vec_unimodal, "rs", linewidth=2, label="unimodal")
# plt.plot(L_vec, q_l_vec_multimodal, "bo", linewidth=2, label="multimodal")
plt.plot(L_vec, q_l_vec_unimodal, "rs", markersize=5, label="unimodal")
plt.plot(L_vec, q_l_vec_multimodal, "bo", markersize=5, label="multimodal")
plt.xlabel("Skill level $l$")
plt.ylabel("$q(l)$")
plt.legend()
plt.grid()

###################################################################
############################ P_B vs R* ############################
###################################################################
plt.subplot(2,2,3)
plt.title("(c) Emergence (unimodal)")
# plt.plot(N_opt_vec, (gcc_ratio_avg), "-o", linewidth=2, label="$\sum_{l} q_{n_l, l} \gamma^{n_l}_l$")
plt.plot(N_opt_vec, gcc_ratio_avg_unimodal, "r-", linewidth=2)

plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Accuracy")
# plt.legend()
plt.xscale("log")
plt.grid()

###################################################################
############################ P_B vs R* ############################
###################################################################
plt.subplot(2,2,4)
plt.title("(d) Plateauing (multimodal)")
# plt.plot(N_opt_vec, (gcc_ratio_avg), "-o", linewidth=2, label="$\sum_{l} q_{n_l, l} \gamma^{n_l}_l$")
plt.plot(N_opt_vec, gcc_ratio_avg_multi_modal, "b-", linewidth=2)

plt.xlabel("$N^*$ (No. of parameters)")
plt.ylabel("Accuracy")
# plt.legend()
plt.xscale("log")
plt.grid()

###################################################################

plt.tight_layout()

# plt.savefig("emerg_plateau.svg", format='pdf')

plt.show(block=False)

# plt.figure(2)
# plt.plot(L_vec, eta_l_vec)
# plt.ylabel("$\eta_l$")
# plt.xlabel("Skill level ($l$)")


brkpnt1 = 1
