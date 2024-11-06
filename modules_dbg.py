
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import *
import scipy as scp
from modules_core import *

'''
For debug - plot f(x) vs x for a given epsilon
'''
def plot_fx_vs_x_dbg(eps, nv, nc, pv, pc, marker="-"):
    x_vec = np.linspace(0.01, 1-0.01, 1000)
    plt.figure(0)
    plt.plot(x_vec, [func_fx_pt(x,eps, nv, nc, pv, pc, neg=True) for x in x_vec], marker, label=f"eps={eps}")
    plt.grid()
    plt.legend()    
    brkpnt1 = 1


def plot_slearnt_vs_s(s_low, s_high, FLOPS, eps, dt, fig_num=0):
  s_vec = np.linspace(s_low, s_high, 50)
  # s_vec = np.logspace(np.log10(s_low), np.log10(s_high), 100)
  s_learnt_vec = np.zeros(s_vec.shape)
  eps_BP_vec = np.zeros(s_vec.shape)
  for ind_s, s in enumerate(s_vec):
    #print(f"s={s}, s/eps = {s/eps}")
    s_learnt, eps_BP, x_BP, Pb, PB, gamma_BP, alpha_val = num_skills_learnt(s, dt, FLOPS, eps, optimize_flag=False)
    s_learnt_vec[ind_s] = s_learnt
    eps_BP_vec[ind_s] = eps_BP

  # plt.figure(fig_num)
  plt.subplot(1,2,1)
  plt.plot(s_vec, s_learnt_vec, linewidth=2, label=f"FLOPs={FLOPS/1e5:0.2f}e5")
  plt.ylabel("$R (1-P_B)$")
  
  # plt.plot(s_vec, FLOPS/s_vec)
  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel("R")  
  plt.grid()

  plt.subplot(1,2,2)
  plt.plot(s_vec, eps_BP_vec, linewidth=2, label=f"FLOPs={FLOPS/1e5:0.2f}e5")
  # plt.plot(s_vec, FLOPS/s_vec)
  #plt.ylim((0.8, 1))
  plt.xscale("log")
  plt.xlabel("R")
  plt.ylabel("$\epsilon^*$")
  plt.grid()
  # plt.show(block=False)

  brkpnt1 = 1

def plot_slearnt_vs_s_biterr(s_low, s_high, FLOPS, eps, dt, fig, ax):
  s_vec = np.linspace(s_low, s_high, 50)
  # s_vec = np.logspace(np.log10(s_low), np.log10(s_high), 100)
  s_learnt_vec = np.zeros(s_vec.shape)
  eps_BP_vec = np.zeros(s_vec.shape)
  for ind_s, s in enumerate(s_vec):
    #print(f"s={s}, s/eps = {s/eps}")
    s_learnt, eps_BP, x_BP, Pb, PB, gamma_BP, alpha_val = num_skills_learnt_biterr(s, dt, FLOPS, eps, optimize_flag=False)
    s_learnt_vec[ind_s] = s_learnt
    eps_BP_vec[ind_s] = eps_BP

  # plt.figure(fig_num)
  line1, = ax[0].plot(s_vec, s_learnt_vec, linewidth=2)
  ax[0].set_ylabel("$R (1-\epsilon^{-1} P_{b, \lambda_T, \\tilde{\\rho}_R})$")
  
  # plt.plot(s_vec, FLOPS/s_vec)
  ax[0].set_xscale("log")
  ax[0].set_yscale("log")
  ax[0].set_xlabel("$R$")  
  ax[0].grid()
  
  line2, = ax[1].plot(s_vec, eps_BP_vec, linewidth=2, label=f"FLOPs={FLOPS/1e5:0.2f}e5")
  ax[1].set_xscale("log")
  ax[1].set_xlabel("$R$")
  ax[1].set_ylabel("$\epsilon^*$")
  ax[1].grid()  

  brkpnt1 = 1

  return line1, line2

def plot_slearnt_vs_s_closed_form(s_low, s_high, FLOPS, eps, dt, fig_num=0):
  s_vec = np.linspace(s_low, s_high, 50)
  # s_vec = np.logspace(np.log10(s_low), np.log10(s_high), 100)
  s_learnt_vec = np.zeros(s_vec.shape)
  eps_BP_vec = np.zeros(s_vec.shape)
  for ind_s, s in enumerate(s_vec):
    #print(f"s={s}, s/eps = {s/eps}")
    s_learnt, eps_BP, x_BP, Pb, PB, gamma_BP, alpha_val = num_skills_learnt_closed_form(s, dt, FLOPS, eps, optimize_flag=False)
    s_learnt_vec[ind_s] = s_learnt
    eps_BP_vec[ind_s] = eps_BP

  # plt.figure(fig_num)
  plt.subplot(1,2,1)
  plt.plot(s_vec, s_learnt_vec, linewidth=2, label=f"FLOPs={FLOPS/1e5:0.2f}e5")
  plt.ylabel("$R (1-P_B)$")
  
  # plt.plot(s_vec, FLOPS/s_vec)
  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel("R")  
  plt.grid()

  plt.subplot(1,2,2)
  plt.plot(s_vec, eps_BP_vec, linewidth=2, label=f"FLOPs={FLOPS/1e5:0.2f}e5")
  # plt.plot(s_vec, FLOPS/s_vec)
  #plt.ylim((0.8, 1))
  plt.xscale("log")
  plt.xlabel("R")
  plt.ylabel("$\epsilon^*$")
  plt.grid()
  # plt.show(block=False)

  brkpnt1 = 1

def plot_slearnt_vs_s_postproc(s_opt_vec, s_learnt_vec, eps_BP_vec, fig, ax, lines, legends):   
  ax[0].plot(s_opt_vec, s_learnt_vec, "ko", markersize=5)

  ax[1].plot(s_opt_vec, eps_BP_vec, "ko", markersize=5)
  ax[1].plot(np.linspace(1e1, 1e9, 100), 0.5*np.ones(100), "k--", linewidth=2)  
  
  ax[0].set_xlim((1e2, 7e4))   
  ax[1].set_xlim((1e2, 7e4))
  ax[1].set_ylim((0.49, 0.6))      

  ax[0].grid()
  ax[1].grid()

  #  fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='outside right upper')
  fig.legend(lines, legends, loc='outside right center')

def plot_slearnt_vs_t(t_low, t_high, s, eps, dt, fig_num=0):
  t_vec = np.linspace(t_low, t_high, 100)
  # s_vec = np.logspace(np.log10(s_low), np.log10(s_high), 100)
  s_learnt_vec = np.zeros(t_vec.shape)
  eps_BP_vec = np.zeros(t_vec.shape)
  for ind_s, t in enumerate(t_vec):
    #print(f"s={s}, s/eps = {s/eps}")
    s_learnt, eps_BP, x_BP, Pb, PB, gamma_BP = num_text_required(t, dt, s, eps, optimize_flag=False)
    s_learnt_vec[ind_s] = s_learnt
    eps_BP_vec[ind_s] = eps_BP

  # plt.figure(fig_num)
  plt.subplot(1,2,1)
  plt.plot(t_vec, s_learnt_vec, label=f"S={s/1e3:0.2f}e3")
  plt.ylabel("$S (1-P_B)$")    
  # plt.plot(s_vec, FLOPS/s_vec)
  plt.xscale("log")
  # plt.yscale("log")
  plt.xlabel("T")
  
  plt.grid()

  plt.subplot(1,2,2)
  plt.plot(t_vec, eps_BP_vec, label=f"S={s/1e3:0.2f}e3")
  # plt.plot(s_vec, FLOPS/s_vec)
  #plt.ylim((0.8, 1))
  plt.xscale("log")
  plt.xlabel("T")
  plt.ylabel("$\epsilon^{BP}$")
  plt.grid()
  # plt.show(block=False)

  brkpnt1 = 1
   

def plot_s_t_vs_flops(FLOPS_vec, s_opt_vec, s_learnt_vec):
    #flops_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])
    #flops_vec = flops_vec[:20:]
    plt.plot(FLOPS_vec, s_opt_vec, 'r-o', label="$S^*$")
    plt.plot(FLOPS_vec, s_learnt_vec, 'm-*', label="skills learnt")
    plt.plot(FLOPS_vec, FLOPS_vec/s_opt_vec, 'b-s', label="$T^*$")
    plt.xlabel("FLOPS")
    plt.ylabel("$S_{learnt} = $skills learnt")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()

    plt.show(block=False)
    
def plot_Pb_vs_s(s_low, s_high, FLOPS, eps, dt):
  s_vec = np.linspace(s_low, s_high, 100)
  Pb_vec = np.zeros(s_vec.shape)
  PB_vec = np.zeros(s_vec.shape)
  eps_BP_vec = np.zeros(s_vec.shape)
  gamma_BP_vec = np.zeros(s_vec.shape)
  x_BP_vec = np.zeros(s_vec.shape)
  for ind_s, s in enumerate(s_vec):
    #print(f"s={s}, s/eps = {s/eps}")
    s_learnt, eps_BP, x_BP, Pb, PB, gamma_BP = num_skills_learnt(s, FLOPS, eps, dt, optimize_flag=False)
    Pb_vec[ind_s] = Pb
    PB_vec[ind_s] = PB
    eps_BP_vec[ind_s] = eps_BP
    gamma_BP_vec[ind_s] = gamma_BP    
    x_BP_vec[ind_s] = x_BP

  #plt.subplot(1,2,1)
  plt.plot(s_vec, Pb_vec, "-o", label="$P_b$")
  plt.plot(s_vec, PB_vec, label="$P_B$")
  plt.plot(s_vec, eps_BP_vec, label="$\epsilon^{BP}$")
  plt.plot(s_vec, gamma_BP_vec, label="$\gamma^{BP}$")
  plt.plot(s_vec, x_BP_vec, label="$x^{BP}$")
  # plt.plot(s_vec, FLOPS/s_vec)
  plt.xlabel("S")
  plt.ylabel("$P_b$/$\epsilon^{BP}$/$P_B$/$\gamma^{BP}$/$x^{BP}$")
  plt.legend()
  plt.grid()