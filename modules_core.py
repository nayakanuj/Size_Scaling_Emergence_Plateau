import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import *
import scipy as scp

X_MIN = 0.1
NUM_PTS = 40

'''
Fixed point characterization of threshold
'''
## binomial only
def lambda_func_binomial(x, p, n):
  return (p*x+(1-p))**(n-1)

def lambda_prime_func_binomial(x, p, n):  # check this later [TODO]
  return (n-1)*p*((p*x+(1-p))**(n-2))

def L_func_binomial(x, p, n):  # check this later [TODO]
  denom = (1-(1-p)**n)
  numer = (p*x+(1-p))**(n)-(1-p)**n #(*)
  return numer/denom

def L_prime_func_binomial(x, p, n): # check this later [TODO]
  denom = (1-(1-p)**n)
  numer = n*p*(p*x+(1-p))**(n-1) # second term in (*) is a constant => vanishes after taking derivative
  return numer/denom

def rho_func_binomial(x, p, n):
  return (p*x+(1-p))**(n-1)

def rho_prime_func_binomial(x, p, n):  # check this later [TODO]
  return (n-1)*p*((p*x+(1-p))**(n-2))

def func_fx_pt(x, eps, nv, nc, pv, pc, neg=False):
    if neg:
      return -(x-eps*lambda_func_binomial(1-rho_func_binomial(1-x, pc, nc), pv, nv))
    else:
      return x-eps*lambda_func_binomial(1-rho_func_binomial(1-x, pc, nc), pv, nv)

'''
Computes \epsilon^{*}_{shannon} closed form
'''
def get_epsilon_BP_closed_form(nv, nc, pv, pc):  
  
  nv, nc, pv, pc = np.float128(nv), np.float128(nc), np.float128(pv), np.float128(pc)

  # Note: nc = R/e, and nv = T  
  r_val = 1-(nv/nc)*(1-(1-(pc))**(nc))/(1-(1-(pv))**(nv))
  
  # avg_degree_check_node = nv*pc
  avg_degree_check_node = nc*pc
  delta_val = (r_val**(avg_degree_check_node-1))*(1-r_val)/(1+(r_val**(avg_degree_check_node-1))*(1-r_val))
  print(f"r = {r_val}, ravg = {avg_degree_check_node}, delta = {delta_val}")

  eps_shannon = 1-r_val/(1-delta_val)

  eps_shannon = min(eps_shannon, 1.0)

  return eps_shannon

'''
Computes \epsilon^{BP} for regular LDPC codes with right (check node) degree = r and left (variable node) degree = l
'''
def get_epsilon_BP(nv, nc, pv, pc):
  eps_low = np.float128(1e-16)
  eps_high = np.float128(1-eps_low)
  eps_tolerance = np.float128(1e-16)
  num_iters = 100
  iter = 0
  eps_BP = eps_high
  # eps_vec = np.zeros(num_iters)
  while iter < num_iters and np.abs(eps_low-eps_high)>eps_tolerance:    

    eps_tmp = (eps_low+eps_high)/2
    if is_solution_in_0_1(eps_tmp, nv, nc, pv, pc) == 0:
      eps_low = eps_tmp
    else:
      eps_high = eps_tmp
    iter += 1
  
    # eps_vec[iter-1] = eps_tmp

  if iter>0.9*num_iters:
    print("max iters reached")

  eps_BP = eps_tmp
  while is_solution_in_0_1(eps_BP, nv, nc, pv, pc) == 0 and eps_BP+eps_tolerance<1.0:
      eps_BP = eps_BP+eps_tolerance
  
  # >>>> for debug
  if is_solution_in_0_1(eps_BP, nv, nc, pv, pc) == 0:
    print(f"eps_BP={eps_BP}")
    brkpnt1 = 1

  return eps_BP


'''
Optimize: grid search followed by scipy.optimize
'''
def my_minimizer(my_fun, args, x_low, x_high, num_pts=40):
  tol_val = 1e-18
  x_vec_tmp = np.linspace(x_low, x_high, num_pts)
  f_vec_tmp = np.zeros(x_vec_tmp.shape)        
  for ind_x, x in enumerate(x_vec_tmp):            
      f_min_tmp = my_fun(x, *args)
      f_vec_tmp[ind_x] = f_min_tmp
  x_init = x_vec_tmp[np.argmin(f_vec_tmp)] 

  x_lb = x_vec_tmp[max(np.argmin(f_vec_tmp)-1, 0)]
  x_ub =x_vec_tmp[min(np.argmin(f_vec_tmp)+1, len(f_vec_tmp)-1)]

  x_init = np.float128(x_init)

  # minimizer_kwargs = {'method':'SLSQP', 'bounds':[(x_lb, x_ub)], 'args':tuple(args)}  
  # minimizer_kwargs = {'method':'Nelder-Mead', 'bounds':[(x_lb, x_ub)], 'args':tuple(args)}  
  # minimizer_kwargs = {'method':'COBYLA', 'bounds':[(x_lb, x_ub)], 'args':tuple(args)}  
  # x_opt = scp.optimize.minimize(my_fun, x0=[x_init], bounds=minimizer_kwargs['bounds'], method=minimizer_kwargs['method'], tol=tol_val, args=minimizer_kwargs['args']).x[0]
  minimizer_kwargs = {'method':'Bounded', 'bounds':(x_lb, x_ub), 'args':tuple(args), 'tol':tol_val}
  x_opt = scp.optimize.minimize_scalar(my_fun, **minimizer_kwargs).x


  return x_opt

'''
Checking if x - \lambda(1-\rho(1-x)) intersects x-axis or not for any x \in (0, 1] (Note the exclusion of 0)
If it intersects, then the root corresponds to eps_tmp > \epsilon^{BP}
'''
def is_solution_in_0_1(eps_tmp, nv, nc, pv, pc):
  x_low = X_MIN #1e-16
  x_high = 1

  # min
  args = [eps_tmp, nv, nc, pv, pc, False] 
  min_x = my_minimizer(func_fx_pt, args, x_low, x_high, num_pts=NUM_PTS)
  # max
  args = [eps_tmp, nv, nc, pv, pc, True]
  max_x = my_minimizer(func_fx_pt, args, x_low, x_high, num_pts=NUM_PTS)

  if np.sign(func_fx_pt(min_x, eps_tmp, nv, nc, pv, pc)) == np.sign(func_fx_pt(max_x, eps_tmp, nv, nc, pv, pc)):
    return 0
  else:
    return 1


def get_x_BP(eps_BP, nv, nc, pv, pc):
  # x_low = X_MIN #1e-14 #1e-16
  # x_high = 1
  x_low = X_MIN #1e-14 #1e-16
  x_high = 1
  # eps_BP = eps_BP+1e-8
  eps_BP = eps_BP

  # args = [eps_BP, nv, nc, pv, pc, False]
  # min_x = my_minimizer(func_fx_pt, args, x_low, x_high, num_pts=NUM_PTS)
  # args = [eps_BP, nv, nc, pv, pc, True]
  # max_x = my_minimizer(func_fx_pt, args, x_low, x_high, num_pts=NUM_PTS)
  
  if is_solution_in_0_1(eps_BP, nv, nc, pv, pc) == 0:
    x_BP = X_MIN
  else:
    args = [eps_BP, nv, nc, pv, pc, False]
    # x_BP = min_x #scp.optimize.brentq(func_fx_pt, min_x, max_x, args=(eps_BP, nv, nc, pv, pc))
    x_BP = my_minimizer(func_fx_pt, args, x_low, x_high, num_pts=NUM_PTS)    

  return x_BP

'''
compute alpha for a given degree distribution
'''
def get_alpha(nv, nc, pv, pc, x_BP, eps_BP):
  if x_BP == X_MIN:
    alpha_val = 1e-5 # some small value so that P_B -> 0
  else:
    x_BP_bar = 1-x_BP
    y_BP = 1-rho_func_binomial(x_BP_bar, pc, nc)
    term1_numer1_2 = rho_func_binomial(x_BP_bar, pc, nc)**2-rho_func_binomial(x_BP_bar**2, pc, nc)
    term1_numer3 = rho_prime_func_binomial(x_BP_bar, pc, nc)*(1-2*x_BP*rho_func_binomial(x_BP_bar, pc, nc))
    term1_numer4 = (x_BP_bar**2)*rho_prime_func_binomial(x_BP_bar**2, pc, nc)
    term1_denom = L_prime_func_binomial(1, pv, nv)*(lambda_func_binomial(y_BP, pv, nv)**2)*(rho_prime_func_binomial(x_BP_bar, pc, nc)**2)

    term2_numer1 = (eps_BP**2)*(lambda_func_binomial(y_BP, pv, nv)**2)
    term2_numer2 = (eps_BP**2)*lambda_func_binomial(y_BP**2, pv, nv)
    term2_numer3 = (y_BP**2)*(eps_BP**2)*lambda_prime_func_binomial(y_BP**2, pv, nv)
    term2_denom = L_prime_func_binomial(1, pv, nv)*(lambda_func_binomial(y_BP, pv, nv)**2)

    alpha_val = ((term1_numer1_2+term1_numer3+term1_numer4)/term1_denom + (term2_numer1+term2_numer2+term2_numer3)/term2_denom)**(1/2)

  if alpha_val>2:
    brkpnt1 = 1

  return alpha_val

'''
Returns the number of skills learnt given max. number of skills learnable s (s depends on the model size)
'''
def num_skills_learnt(s, dt, FLOPS, eps, optimize_flag=True):
  t = FLOPS/s  
  sbar = s*(1-eps)/eps
  n = sbar+s # = s/eps
  nv, nc = t, n # these are not number of variable nodes and check nodes. These are n's of binomial distribution
  ds = dt*t/s # dtbar*t/n = (dt/eps)*(t/(s/eps)) = dt*t/s
  dtbar = ds*n/t # = dt/eps
  pv, pc = ds/nv, dtbar/nc

  pv = min(pv, 1.0)
  pc = min(pc, 1.0)
  
  eps_BP = get_epsilon_BP(nv, nc, pv, pc)

  x_BP, Pb, gamma_BP = 0.0, 0.0, 0.0 # initialization
  if eps_BP == 1:
    s_learnt = s
    Q_val = 0.0
    alpha = 1e-3 # some small value so that P_B -> 0
  else:    
    x_BP = get_x_BP(eps_BP, nv, nc, pv, pc)
    # alpha = 0.5791
    alpha = get_alpha(nv, nc, pv, pc, x_BP, eps_BP)      
    
    Q_val = 1-norm.cdf(float(np.sqrt(n)*(eps_BP - eps)/alpha), loc=0, scale=1)

    # # >>>> for debug - BEGIN
    # gamma_BP = eps_BP*L_func_binomial(1-rho_func_binomial(1-x_BP, pc, nc), pv, nv)
    # Q_val = gamma_BP*Q_val
    # # >>>> for debug - END

    s_learnt = s*(1-Q_val)
    #print(f"s={s}, s_learnt={s_learnt}, eps_BP = {eps_BP}, Q={Q}")

  if optimize_flag:
    return -s_learnt
  else:
    return s_learnt, eps_BP, x_BP, Pb, Q_val, gamma_BP, alpha

def num_skills_learnt_biterr(s, dt, FLOPS, eps, optimize_flag=True):
  t = FLOPS/s  
  sbar = s*(1-eps)/eps
  n = sbar+s # = s/eps
  nv, nc = t, n # these are not number of variable nodes and check nodes. These are n's of binomial distribution
  ds = dt*t/s # dtbar*t/n = (dt/eps)*(t/(s/eps)) = dt*t/s
  dtbar = ds*n/t # = dt/eps
  pv, pc = ds/nv, dtbar/nc

  pv = min(pv, 1.0)
  pc = min(pc, 1.0)
  
  eps_BP = get_epsilon_BP(nv, nc, pv, pc)

  x_BP, Pb, gamma_BP = 0.0, 0.0, 0.0 # initialization
  if eps_BP == 1:
    s_learnt = s
    Q_val = 0.0
    alpha = 1e-3 # some small value so that P_B -> 0
  else:    
    x_BP = get_x_BP(eps_BP, nv, nc, pv, pc)    
    alpha = get_alpha(nv, nc, pv, pc, x_BP, eps_BP)      
    
    Q_val = 1-norm.cdf(float(np.sqrt(n)*(eps_BP - eps)/alpha), loc=0, scale=1)

    gamma_BP = eps_BP*L_func_binomial(1-rho_func_binomial(1-x_BP, pc, nc), pv, nv)
    Q_val = gamma_BP*Q_val/eps

    s_learnt = s*(1-Q_val)
    #print(f"s={s}, s_learnt={s_learnt}, eps_BP = {eps_BP}, Q={Q}")

  if optimize_flag:
    return -s_learnt
  else:
    return s_learnt, eps_BP, x_BP, Pb, Q_val, gamma_BP, alpha
  

def num_skills_learnt_closed_form(s, dt, FLOPS, eps, optimize_flag=True):
  t = FLOPS/s  
  sbar = s*(1-eps)/eps
  n = sbar+s # = s/eps
  nv, nc = t, n # these are not number of variable nodes and check nodes. These are n's of binomial distribution
  ds = dt*t/s # dtbar*t/n = (dt/eps)*(t/(s/eps)) = dt*t/s
  dtbar = ds*n/t # = dt/eps
  pv, pc = ds/nv, dtbar/nc

  pv = min(pv, 1.0)
  pc = min(pc, 1.0)
  
  eps_BP = get_epsilon_BP_closed_form(nv, nc, pv, pc)

  x_BP, Pb, gamma_BP = 0.0, 0.0, 0.0 # initialization
  if eps_BP == 1:
    s_learnt = s
    Q_val = 0.0
    alpha = 1e-3 # some small value so that P_B -> 0
  else:
    
    alpha = np.sqrt(eps*(1-eps))
    
    Q_val = 1-norm.cdf(float(np.sqrt(n)*(eps_BP - eps)/alpha), loc=0, scale=1)

    s_learnt = s*(1-Q_val)

  if optimize_flag:
    return -s_learnt
  else:
    return s_learnt, eps_BP, x_BP, Pb, Q_val, gamma_BP, alpha

def num_text_required(t, dt, s, eps, optimize_flag=True):
  #t = FLOPS/s  
  sbar = s*(1-eps)/eps
  n = sbar+s # = s/eps
  nv, nc = t, n # these are not number of variable nodes and check nodes. These are n's of binomial distribution
  ds = dt*t/s # dtbar*t/n = (dt/eps)*(t/(s/eps)) = dt*t/s
  dtbar = ds*n/t # = dt/eps
  pv, pc = ds/nv, dtbar/nc
  
  eps_BP = get_epsilon_BP(nv, nc, pv, pc)

  x_BP, Pb, gamma_BP = 0.0, 0.0, 0.0 # initialization  
  x_BP = get_x_BP(eps_BP, nv, nc, pv, pc)
  # alpha = 0.5791
  alpha = get_alpha(nv, nc, pv, pc, x_BP, eps_BP)  
  # gamma_BP = eps_BP*lambda_func(1-(1-x_BP)**r, p, n)
  Q_val = 1-norm.cdf(np.sqrt(n)*(eps_BP - eps)/alpha, loc=0, scale=1)
  # s_learnt_div_t = s*np.exp(-(1-Q_val-0.95)**2/0.01)
  s_learnt = s*(1-Q_val)
  #print(f"s={s}, s_learnt={s_learnt}, eps_BP = {eps_BP}, Q={Q}")

  if optimize_flag:
    return -s_learnt
  else:
    return s_learnt, eps_BP, x_BP, Pb, Q_val, gamma_BP, alpha

def get_PB(n, nv, pv, nc, pc, eps):
    # alpha_val = 0.5791
    x_BP = get_x_BP(eps_BP, nv, nc, pv, pc)
    alpha_val = get_alpha(nv, nc, pv, pc, x_BP, eps_BP)  
    eps_BP = get_epsilon_BP(nv, nc, pv, pc)
    Q_val = 1-norm.cdf(np.sqrt(n)*(eps_BP - eps)/alpha_val, loc=0, scale=1)
    return Q_val, eps_BP

# def get_PB_epsthfixed(n, eps, eps_BP):
#    alpha_val = 0.5791    
#    Q_val = 1-norm.cdf(np.sqrt(n)*(eps_BP - eps)/alpha_val, loc=0, scale=1)
#    return Q_val


def print_fun(x, f, accepted):
  print("at minimum %.4f accepted %d" % (f, int(accepted)))


def generate_power_law_samples(alpha, xmin, size):
    """
    Generate samples from a power-law distribution.

    Parameters:
    - alpha: the exponent parameter of the power-law distribution.
    - xmin: the minimum value of x.
    - size: the number of samples to generate.

    Returns:
    - samples: an array of samples from the power-law distribution.
    """
    r = np.random.uniform(0, 1, size)
    samples = xmin * (1 - r) ** (-1 / (alpha - 1))
    
    return samples


def generate_zipf_samples(min_rank, max_rank, size, a=1.0):
    # Generate ranks in the specified range
    ranks = np.arange(min_rank, max_rank + 1)
    
    # Calculate the corresponding probabilities using Zipf's law
    probabilities = ranks ** (-a)
    probabilities /= probabilities.sum()  # Normalize to make it a probability distribution
    
    # Generate samples based on the Zipf distribution
    samples = np.random.choice(ranks, size=size, p=probabilities)
    
    return samples