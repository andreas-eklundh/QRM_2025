### Mandatory assignment
# Various utils. 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm ,binom, t, expon, invgamma, genpareto  
from scipy.optimize import minimize 
from scipy.stats import multivariate_normal as mnorm
from scipy.stats import multivariate_t as mt
from scipy.interpolate import UnivariateSpline # for smoothing
import pandas as pd
# Utils
import utils as u

## Set global seed.
np.random.seed(2025)
# read data and initial prep. 
data = pd.read_csv(f'Data/stock_data.csv').sort_values('Date')
N_obs = len(data['Date'])
# Get log returns. 
name_list = ['GOOG','MSFT','MRK','IDU']
for s in ['GOOG','MSFT','MRK','IDU']:
    data[s+'_shift'] = data[s].shift(1) # Shift once.
    data['X_'+s] = np.log(data[s]) - np.log(data[s+'_shift'])
    data['X_'+s] = data['X_'+s].fillna(0) # Zero return othjerwise.

# convert to arrays
goog,msft = np.array(data['X_GOOG']),np.array(data['X_MSFT'])
mrk,idu = np.array(data['X_MRK']),np.array(data['X_IDU'])
# Get negative returns
goog_neg,msft_neg,mrk_neg,idu_neg = goog*(-1),msft*(-1), mrk*(-1),idu*(-1)

### 1. Marginal distributions for each stock.
u.marginal_plots(goog_neg,name_list[0]+'_QQ',df1=2,df2=3,df3=4,df4=5,df5=6,markersize=5)
u.marginal_plots(msft_neg,name_list[1]+'_QQ',df1=2,df2=3,df3=4,df4=5,df5=6,markersize=5)
u.marginal_plots(mrk_neg,name_list[2]+'_QQ',df1=2,df2=3,df3=4,df4=5,df5=6,markersize=5)
u.marginal_plots(idu_neg,name_list[3]+'_QQ',df1=2,df2=3,df3=4,df4=5,df5=6,markersize=5)

# Google seems adequately fit by a t-distr. with df=3
# Msft df =3 is reasonable.
# Merck df=3 seems reasonable.
# IDU df=2 seems adequate (maybe df=3 if we want second moment)
dflist = [3,3,3,3]

### 2. upper tail distributions of losses using EVT. 
# Note this is only meaningfull for positive values of X
# i.e. the 'loss tail'

for X,name,df in zip([goog_neg,msft_neg,mrk_neg,idu_neg],name_list,dflist):
    x = X[X>0]
    n_obs = x.shape[0]
    ## Analysing thorugh Hill-Estimator (discard first few obs.):
    k_grid = np.array([i for i in range(5,200)])
    alpha_hat = []
    for k in k_grid:
        alpha_hat.append(u.HillEstimator(x,k))
    alpha_hat_50 = np.array(alpha_hat)

    plt.plot(k_grid-1,alpha_hat_50,color = 'black',label='Hill-plot')
    plt.hlines(y = df,xmin=np.min(k_grid),xmax = np.max(k_grid),color = 'red',label="t-distr index")
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Alpha estimate')
    plt.title(f'Hill Plot, {name}')
    plt.legend()
    plt.savefig(f'Figures/{name}_Hill.png')
    plt.show()

# Evidently from the plot, is that the indices obtained from the 
# QQ plot analysis are not suffucient
# google MRk idu somewhat easy to read. Difficult msft
H_k_goog,H_k_msft,H_k_mrk,H_k_idu = 105,165,130,100
# Note hor these k's, we may have values 
a_goog,a_msft = u.HillEstimator(goog_neg,H_k_goog), u.HillEstimator(msft_neg,H_k_msft)
a_mrk,a_idu = u.HillEstimator(mrk_neg,H_k_mrk),u.HillEstimator(idu_neg,H_k_idu)
print(f"Indixes: google:{a_goog.round(2)},Microsoft:{a_msft.round(2)}")
print(f"MRK:{a_mrk.round(2)},IDU:{a_idu.round(2)}")
hill_list = [a_goog,a_msft,a_mrk,a_idu ]
# This methodology indicates that the 
# choices from previously was not too much off. 

## POT-Methodolody.
for X,name in zip([goog_neg,msft_neg,mrk_neg,idu_neg],name_list):
    x_sorted = - np.sort(-X) # Sort decending. 
    n_obs = x_sorted.shape[0]
    k_list = [i for i in range(0,n_obs)]
    e_u = np.zeros(shape=n_obs)
    for k in k_list:
        e_u[k] = u.emp_mean_exess(x_sorted,k)
    plt.scatter(x_sorted[1:],e_u[1:],label='Mean-Exess plot', 
                color='green',s=1)
    plt.grid()
    plt.xlabel('x_k,n')
    plt.ylabel("Empirical-mean exess function")
    plt.title(f'Mean-Exess plotm {name}')
    plt.savefig(f'Figures/{name}_Mean_excess.png')
    plt.show()

# Get k's - difficult for google nad microsoft.. 
# u_goog,u_msft,u_MRK, u_IDU=0.025, 0.04, 0.015, 0.02
u_goog,u_msft,u_MRK, u_IDU=0.045, 0.04, 0.015, 0.015
u_list = [u_goog,u_msft,u_MRK, u_IDU]
pot_indices,beta_list,gamma_list = [],[],[]
# Do remainder of analysis. 
for X,thres,name in zip([goog_neg,msft_neg,mrk_neg,idu_neg],u_list,name_list):
    x_sorted = - np.sort(-X) # Sort decending. 
    k_idx = np.argmin(x_sorted>thres) - 1 
    k = k_idx + 1 # plus 1 as e.g. idx 3 corresponds to 4.
    print(f"k is {k} (in idx terms), val of k {x_sorted[k_idx]}")
    n_obs = x_sorted.shape[0]
    # tnumber exeeding threshold and retransform. 
    x_k =  x_sorted[k_idx]
    Z_i = (x_sorted[:k_idx] -x_k) # exess vals.
    N_u = len(Z_i)
    # Optimize numercally. 
    x0 = [2.6,0.7] # take params from similar problem
    res = minimize(fun=u.GPD_neg_loglike,x0=x0,args=(N_u,Z_i),
                method='Nelder-Mead',options={'fatol': 1e-20})
    beta_hat,gamma_hat = res.x
    beta_list.append(beta_hat)
    gamma_list.append(gamma_hat)
    print(f'Name {name}: Gamma_hat: {gamma_hat}, beta_hat: {beta_hat}')
    Z_orig_order = np.array(X[X>x_k] - x_k)
    G = u.GPD_fun(gamma_hat,beta_hat,Z_orig_order)
    gen_res = - np.log(G)
    gen_res_smooth = np.cumsum(gen_res) / np.cumsum(np.ones(N_u))
    # Sort residuals and Z
    Z_sorted = np.sort(Z_i)
    ges_res_sort = np.sort(gen_res)
    p_vec_tail =  np.array([(i)/(k) for i in range(1,k_idx+1)])
    q_expo = expon.ppf(p_vec_tail)

    F_n_bar = N_u / n_obs
    F_n_u_bar = 1 - u.GPD_fun(gamma_hat,beta_hat,Z_sorted)
    F_bar_upx = F_n_bar * F_n_u_bar

    # plot vs. Pareto counterpart. 
    F_bar_emp = 1 - np.cumsum(1/n_obs*np.ones(x_sorted.shape[0]))
    F_bar_emp_tail = F_bar_emp[n_obs-k_idx:]

    # plotting.
    fig, ax = plt.subplots(nrows=2,ncols=2)
    # Normal distributions
    ax1 =  ax[0,0]
    ax1.scatter([(i) for i in range(0,k_idx)],gen_res, 
                color='blue',s=1)
    ax1.plot([(i) for i in range(0,k_idx)],
             gen_res_smooth,color='black')
    ax1.grid()
    ax1.set_xlabel('Ordering (1:k)')
    ax1.set_ylabel('Generalized residuals')
    ax1.set_title(f'Residual plot')
    # t-distributions. 
    ax2 =  ax[0,1]
    ax2.scatter(q_expo,ges_res_sort, color='blue',s=1)
    ax2.plot(q_expo,q_expo,color='black',label='Reference line if distr true')
    ax2.grid()
    ax2.set_xlabel('Theoretical quantiles')
    ax2.set_ylabel('Empirical quantiles')
    ax2.set_title(f'Exponential quantiles')
    ax3 =  ax[1,0]
    ax3.plot((Z_sorted + x_k), F_bar_emp_tail, label='Empirical tail',color='red')
    ax3.plot((Z_sorted + x_k), F_bar_upx, label='Pareto tail',color='blue')
    ax3.grid()
    ax3.set_xlabel('Log returns (x)')
    ax3.legend()
    fig.tight_layout()
    plt.savefig(f'Figures/{name}_fits.png')
    plt.show()
    pot_indices.append(1/gamma_hat)


# Finally, compare all indeixes: 
print(f'Google t-index: {dflist[0]}, Hill: {hill_list[0].round(2)}, POT: {pot_indices[0].round(2)}')
print(f'Microsoft t-index: {dflist[1]}, Hill: {hill_list[1].round(2)}, POT: {pot_indices[1].round(2)}')
print(f'Merck t-index: {dflist[2]}, Hill: {hill_list[2].round(2)}, POT: {pot_indices[2].round(2)}')
print(f'IDU t-index: {dflist[3]}, Hill: {hill_list[3].round(2)}, POT: {pot_indices[3].round(2)}')
print(f'Google beta {beta_list[0].round(4)}, gamma: {gamma_list[0].round(3)}')
print(f'Microsoft beta {beta_list[1].round(4)}, gamma: {gamma_list[1].round(3)}')
print(f'Merck beta{beta_list[2].round(4)}, gamma: {gamma_list[2].round(3)}')
print(f'IDU beta {beta_list[3].round(4)}, gamma: {gamma_list[3].round(3)}')


### 3. Beginning on joint comparisons. 
X_pf1 = np.array([goog,msft ])
X_pf2 = np.array([mrk,idu  ])
X_pf1_neg = np.array([goog_neg,msft_neg ])
X_pf2_neg = np.array([mrk_neg,idu_neg  ])
# Do marginal plots: 
fig, ax = plt.subplots(nrows=1,ncols=2)
ax1 =  ax[0]
ax1.scatter(goog_neg,  msft_neg, color='blue',s=1)
ax1.grid()
ax1.set_xlabel('Google')
ax1.set_ylabel('Microsoft')
ax1.set_title('Negative Log returns')
ax2 =  ax[1]
ax2.scatter(mrk_neg,idu_neg, color='blue',s=1)
ax2.grid()
ax2.set_xlabel('Merck')
ax2.set_ylabel('IDU')
ax2.set_title(f'Negative Log returns')
fig.tight_layout()
plt.savefig('Figures/PF_plot.png')
plt.show()

# Propose eliptical distributions for the pair.
# Estimate mu and covariance matrix from empirical data.
# Then optimize over nu for the t-distribution.
# Portfolio 1
X1 = X_pf1_neg.T  # Shape: (n_samples, d)
mu_fixed1 = np.mean(X1, axis=0)  # Fixed mean vector
sigma_fixed1 = np.cov(X1, rowvar=False)  # Fixed covariance matrix
sigma_fixed1 += np.eye(sigma_fixed1.shape[0]) * 1e-6  # Regularization, so it is not singular

# Initial value for nu
nu_init = np.array([10])  # Starting value in an array format

# Optimize only over nu
result1 = minimize(
    u.multivariate_t_log_likelihood_dfs, nu_init,
    args=(X1, mu_fixed1, sigma_fixed1),
    method='Nelder-Mead',
    bounds=[(2.1, None)],  # Ensure nu > 2
    options={'fatol': 1e-10}
)

# Extract optimized degrees of freedom
nu_pf1 = result1.x[0]
dispersion_matrix1 = sigma_fixed1 * (nu_pf1 - 2) / nu_pf1

X2 = X_pf2_neg.T  # Shape: (n_samples, d)
mu_fixed2 = np.mean(X2, axis=0)  # Fixed mean vector
sigma_fixed2 = np.cov(X2, rowvar=False)  # Fixed covariance matrix
sigma_fixed2 += np.eye(sigma_fixed2.shape[0]) * 1e-6  # Regularization, so it is not singular

# Optimize only over nu
result2 = minimize(
    u.multivariate_t_log_likelihood_dfs, nu_init,
    args=(X2, mu_fixed2, sigma_fixed2),
    method='Nelder-Mead',
    bounds=[(2.1, None)],  # Ensure nu > 2
    options={'fatol': 1e-10}
)

# Extract optimized degrees of freedom
nu_pf2 = result2.x[0]
dispersion_matrix2 = sigma_fixed2 * (nu_pf2 - 2) / nu_pf2

# Print results
print(f'Estimated parameters for portfolio 1: nu={nu_pf1}, mu={mu_fixed1}, sigma={dispersion_matrix1}')
print(f'Estimated parameters for portfolio 2: nu={nu_pf2}, mu={mu_fixed2}, sigma={dispersion_matrix2}')

# Scatter plot of pairs with fitted eliptical distribution.
# Number of simulated points
n_sim = X_pf1_neg.shape[1] 
# Simulate from fitted Multivariate t-distribution
sim_pf1 = mt.rvs(df=nu_pf1, loc=mu_fixed1, shape=dispersion_matrix1, size=n_sim)
sim_pf2 = mt.rvs(df=nu_pf2, loc=mu_fixed2, shape=dispersion_matrix2, size=n_sim)

# Extract simulated values
sim_goog, sim_msft = sim_pf1[:, 0], sim_pf1[:, 1]
sim_mrk, sim_idu = sim_pf2[:, 0], sim_pf2[:, 1]

# Create scatter plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Plot Google & Microsoft
ax1 = ax[0]
ax1.scatter(goog_neg, msft_neg, color='blue', s=1, label="Empirical Data")
ax1.scatter(sim_goog, sim_msft, color='red', s=1, alpha=0.5, label=f"Simulated Data (df={nu_pf1:.2f})")
ax1.grid()
ax1.set_xlabel('Google')
ax1.set_ylabel('Microsoft')
ax1.set_title('Negative Log Returns: Google & Microsoft')
ax1.legend()

# Plot Merck & IDU
ax2 = ax[1]
ax2.scatter(mrk_neg, idu_neg, color='blue', s=1, label="Empirical Data")
ax2.scatter(sim_mrk, sim_idu, color='red', s=1, alpha=0.5, label=f"Simulated Data (df={nu_pf2:.2f})")
ax2.grid()
ax2.set_xlabel('Merck')
ax2.set_ylabel('IDU')
ax2.set_title('Negative Log Returns: Merck & IDU')
ax2.legend()

fig.tight_layout()
plt.savefig('Figures/Multivariate-t_plot.png')
plt.show()

### 4. Copula approach.

# Over u, use Pareto Tail.
i_list = [0,1,2,3]
for X,thres,name,i in zip([goog_neg,msft_neg,mrk_neg,idu_neg],u_list,name_list,i_list):
    x_sorted = - np.sort(-X) # Sort descending. 
    n = x_sorted.shape[0]
    beta_est = beta_list[i]
    gamma_est = gamma_list[i]
    # Get excesses across 
    x_k = thres
    F_marginal = u.GPD_emp(np.sort(x_sorted), x_k, beta_est, gamma_est)
    #F_emp = np.cumsum(np.ones(n) * 1 / n)

    # Add empirical
    plt.plot(np.sort(x_sorted), F_marginal, color='red', label='Marginal mixed emp and GPD')
    plt.axvline(x=x_k, color='blue', linestyle='--', label=f'Threshold x_k = {x_k:.3f}')
    plt.legend(loc='upper left')
    plt.xlabel("Log returns")
    plt.ylabel("Distribution function")
    plt.grid()
    plt.title(f"Marginal distribution {name}")
    plt.savefig(f"Figures/{name}_Marginal.png")
    plt.show()


### 5. Exploratory analysis, upper tail dependence
# Probably to be done for negative returns 
rho_tau_pf1 = u.kendalls_tau(X_pf1_neg.T)
rho_tau_pf2 = u.kendalls_tau(X_pf2_neg.T)
print(f'rho tau pf1 {rho_tau_pf1.round(3)}')
print(f'rho tau pf2 {rho_tau_pf2.round(3)}')
# Based on kendall's tau, it seems that especially 
# Microsoft and Google are comonotone. 
# But in general somewhat comonotone. 
copulas = u.Copulas()

# Assuming this, we can use Theorem 11.7 to determine the
# standard correlation .
N_sim = n_sim # 10**3
rho_gauss1 = np.sin(rho_tau_pf1*np.pi/2)
rho_gauss2 = np.sin(rho_tau_pf2*np.pi/2)
rho_mat_1 = np.array([[1,rho_gauss1],
                      [rho_gauss1,1]])

# First PF (microsoft and google)
# Fit Copulas.
U_gauss1 = copulas.simul_Gaussian(rho_mat_1,N_sim)
# Convert to returns:
X_gauss1 = U_gauss1.copy()
X_gauss1[:,0] = u.inverse_GPD_emp(U_gauss1[:,0],np.sort(goog_neg),
                                  u_list[0],beta_list[0],gamma_list[0])
X_gauss1[:,1] = u.inverse_GPD_emp(U_gauss1[:,1],np.sort(msft_neg),
                                  u_list[1],beta_list[1],gamma_list[1])
## Thm 7.43 in McNeil - Same rho_tau relation holds for t-copula.
# So in principle usable. Looks better? Choose nu? Go w. Gumbel?
U_t1 = copulas.simul_t_distr(rho_mat_1,nu_pf1,N_sim)
# Convert to returns:
X_t1 = U_t1.copy()
X_t1[:,0] = u.inverse_GPD_emp(U_t1[:,0],np.sort(goog_neg),
                                  u_list[0],beta_list[0],gamma_list[0])
X_t1[:,1] = u.inverse_GPD_emp(U_t1[:,1],np.sort(msft_neg),
                                  u_list[1],beta_list[1],gamma_list[1])
# Gumbel
theta1 = 1 / (1-rho_tau_pf1)

u_gumb = copulas.simul_gumbel(theta=theta1,dim=2,N_sim=N_sim)
# We then need to transform to X.
X_gumb1 = np.empty(shape=u_gumb.shape) 
# Overwrite a matrix -> faster for these large matrices
X_gumb1[:,0] = u.inverse_GPD_emp(u_gumb[:,0],np.sort(goog_neg),
                                  u_list[0],beta_list[0],gamma_list[0])
X_gumb1[:,1] = u.inverse_GPD_emp(u_gumb[:,1],np.sort(msft_neg),
                                  u_list[1],beta_list[1],gamma_list[1])

# Scond PF 
# FITTING COPULAS
rho_mat_2 = np.array([[1,rho_gauss2],
                      [rho_gauss2,1]])

U_gauss2 = copulas.simul_Gaussian(rho_mat_2,N_sim)
X_gauss2 = U_gauss2.copy()

X_gauss2[:,0] = u.inverse_GPD_emp(U_gauss2[:,0],np.sort(mrk_neg),
                                  u_list[2],beta_list[2],gamma_list[2])
X_gauss2[:,1] = u.inverse_GPD_emp(U_gauss2[:,1],np.sort(idu_neg),
                                  u_list[3],beta_list[3],gamma_list[3])
# t-copula 
U_t2 = copulas.simul_t_distr(rho_mat_2,nu_pf2,N_sim)
# Convert to returns:
X_t2 = U_t2.copy()
X_t2[:,0] = u.inverse_GPD_emp(U_t2[:,0],np.sort(goog_neg),
                                  u_list[2],beta_list[2],gamma_list[2])
X_t2[:,1] = u.inverse_GPD_emp(U_t2[:,1],np.sort(msft_neg),
                                  u_list[3],beta_list[3],gamma_list[3])
# Gumbel
theta2 = 1 / (1-rho_tau_pf2)
u_gumb = copulas.simul_gumbel(theta=theta2,dim=2,N_sim=N_sim)
# We then need to transform to X.
X_gumb2 = np.empty(shape=u_gumb.shape) 
# Overwrite a matrix -> faster for these large matrices
X_gumb2[:,0] = u.inverse_GPD_emp(u_gumb[:,0],np.sort(mrk_neg),
                                  u_list[2],beta_list[2],gamma_list[2])
X_gumb2[:,1] = u.inverse_GPD_emp(u_gumb[:,1],np.sort(idu_neg),
                                  u_list[3],beta_list[3],gamma_list[3])


### 6. Frechet Bounds
## Comonotonic copula. 
# Simulation could look something like:
u_com1 = copulas.simul_frechet_bound_M(N_sim=N_sim)
# We then need to transform to X.
X_com1 = np.empty(shape=u_com1.shape) 
# Overwrite a matrix -> faster for these large matrices
X_com1[:,0] = u.inverse_GPD_emp(u_com1[:,0],np.sort(goog_neg),
                                  u_list[0],beta_list[0],gamma_list[0])
X_com1[:,1] = u.inverse_GPD_emp(u_com1[:,1],np.sort(msft_neg),
                                  u_list[1],beta_list[1],gamma_list[1])
u_com2 = copulas.simul_frechet_bound_M(N_sim=N_sim)
# We then need to transform to X.
X_com2 = np.empty(shape=u_com2.shape) 
# Overwrite a matrix -> faster for these large matrices
X_com2[:,0] = u.inverse_GPD_emp(u_com2[:,0],np.sort(mrk_neg),
                                  u_list[2],beta_list[2],gamma_list[2])
X_com2[:,1] = u.inverse_GPD_emp(u_com2[:,1],np.sort(idu_neg),
                                  u_list[3],beta_list[3],gamma_list[3])

u_coum1 = copulas.simul_frechet_bound_W(N_sim=N_sim)
# We then need to transform to X.
X_coum1 = np.empty(shape=u_coum1.shape) 
# Overwrite a matrix -> faster for these large matrices
X_coum1[:,0] = u.inverse_GPD_emp(u_coum1[:,0],np.sort(goog_neg),
                                  u_list[0],beta_list[0],gamma_list[0])
X_coum1[:,1] = u.inverse_GPD_emp(u_coum1[:,1],np.sort(msft_neg),
                                  u_list[1],beta_list[1],gamma_list[1])
u_coum2 = copulas.simul_frechet_bound_W(N_sim=N_sim)
# We then need to transform to X.
X_coum2 = np.empty(shape=u_coum2.shape) 
# Overwrite a matrix -> faster for these large matrices
X_coum2[:,0] = u.inverse_GPD_emp(u_coum2[:,0],np.sort(mrk_neg),
                                  u_list[2],beta_list[2],gamma_list[2])
X_coum2[:,1] = u.inverse_GPD_emp(u_coum2[:,1],np.sort(idu_neg),
                                  u_list[3],beta_list[3],gamma_list[3])


### Plotting of various copula simulatinos
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
# Plot Google & Microsoft
ax1 = ax[0,0]
ax1.scatter(X_gauss1[:,0],X_gauss1[:,1],color='red',
            label = 'Gaussian Copula',s=1)
ax1.scatter(goog_neg, msft_neg, color='blue', s=1, label="Empirical Data")
ax1.legend()
ax1.grid()
ax1.set_xlabel('Google')
ax1.set_ylabel('Microsoft')
ax2 = ax[0,1]
ax2.scatter(X_t1[:,0],X_t1[:,1],color='red',
            label = 't- Copula',s=1)
ax2.scatter(goog_neg, msft_neg, color='blue', s=1, label="Empirical Data")
ax2.legend()
ax2.grid()
ax2.set_xlabel('Google')
ax2.set_ylabel('Microsoft')
ax3 = ax[1,0]
ax3.scatter(X_gumb1[:,0],X_gumb1[:,1],color='red',
            label = 'Gumbel Copula',s=1)
ax3.scatter(goog_neg, msft_neg, color='blue', s=1, label="Empirical Data")
ax3.legend()
ax3.grid()
ax3.set_xlabel('Google')
ax3.set_ylabel('Microsoft')
ax4 = ax[1,1]
ax4.scatter(X_com1[:,0],X_com1[:,1],color='red',
            label = 'Comonotonic Copula',s=1)
ax4.scatter(goog_neg, msft_neg, color='blue', s=1, label="Empirical Data")
ax4.legend()
ax4.grid()
ax4.set_xlabel('Google')
ax4.set_ylabel('Microsoft')
fig.tight_layout()
plt.show()

# plot for saving.
plt.scatter(X_gumb1[:,0],X_gumb1[:,1],color='red',
            label = 'Simulated data from Gumbel Copula',s=1)
plt.scatter(goog_neg, msft_neg, color='blue', s=1, label="Empirical Data")
plt.xlabel('Google')
plt.ylabel('Microsoft')
plt.legend()
plt.grid()
plt.savefig(f"Figures/Copula_pf1.png")
plt.show()


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
# Plot Google & Microsoft
ax1 = ax[0,0]
ax1.scatter(X_gauss2[:,0],X_gauss2[:,1],color='red',
            label = 'Gaussian Copula',s=1)
ax1.scatter(mrk_neg, idu_neg, color='blue', s=1, label="Empirical Data")
ax1.legend()
ax1.grid()
ax1.set_xlabel('Merck')
ax1.set_ylabel('IDU')
# t-copula
ax2 = ax[0,1]
ax2.scatter(X_t2[:,0],X_t2[:,1],color='red',
            label = 't-Copula',s=1)
ax2.scatter(mrk_neg, idu_neg, color='blue', s=1, label="Empirical Data")
ax2.set_xlabel('Merck')
ax2.set_ylabel('IDU')
ax2.legend()
ax2.grid()
## Try with a Gumbel. 
ax3 = ax[1,0]
ax3.scatter(X_gumb2[:,0],X_gumb2[:,1],color='red',
            label = 'Gumbel Copula',s=1)
ax3.scatter(mrk_neg, idu_neg, color='blue', s=1, label="Empirical Data")
ax3.set_xlabel('Merck')
ax3.set_ylabel('IDU')
ax3.legend()
ax3.grid()
ax4 = ax[1,1]
ax4.scatter(X_com2[:,0],X_com2[:,1],color='red',
            label = 'Comonotonic Copula',s=1)
ax4.scatter(mrk_neg, idu_neg, color='blue', s=1, label="Empirical Data")
ax4.set_xlabel('Merck')
ax4.set_ylabel('IDU')
ax4.legend()
ax4.grid()
fig.tight_layout()
plt.show()

# Plot for svaing.
plt.scatter(X_gumb2[:,0],X_gumb2[:,1],color='red',
            label = 'Simulated data from Gumbel Copula',s=1)
plt.scatter(mrk_neg, idu_neg, color='blue', s=1, label="Empirical Data")
plt.xlabel('Merck')
plt.ylabel('IDU')
plt.legend()
plt.grid()
plt.savefig(f"Figures/Copula_pf2.png")
plt.show()


### 7. Calculate VaR using varous approaches.
S_0 = np.array([10000,10000])
var_thres = 0.9999

def L_fun(X,S0):
    return - np.matmul(np.exp(X)-1,S0)

## Empirical values. 
# Tech portfolio
L_emp_pf1 = L_fun((-1)*X_pf1_neg.T,S_0)
VaR_emp_pf1 = u.VaR(L_emp_pf1,var_thres)
print(f"Empirical VaR PF1 {VaR_emp_pf1.round(3)}")

# Other index. 
L_emp_pf2 = L_fun((-1)*X_pf2_neg.T,S_0)
VaR_emp_pf2 = u.VaR(L_emp_pf2,var_thres)
print(f"Empirical VaR PF2 {VaR_emp_pf2.round(3)}")

## Elliptical approach. 
# Simulate again urins N_sim
np.random.seed(2025)
N_sim = 10**7
X_elliptical1 = mt.rvs(df=nu_pf1, loc=mu_fixed1, shape=dispersion_matrix1, size=N_sim)
X_elliptical2 = mt.rvs(df=nu_pf2, loc=mu_fixed2, shape=dispersion_matrix2, size=N_sim)

# VaR calc 
# Tech index.
L_el_pf1 = L_fun((-1)*X_elliptical1,S_0)
VaR_el_pf1 = u.VaR(L_el_pf1,var_thres)
print(f"Elliptical VaR PF1 {VaR_el_pf1.round(3)}")

# Other index. 
L_el_pf2 = L_fun((-1)*X_elliptical2,S_0)
VaR_el_pf2 = u.VaR(L_el_pf2,var_thres)
print(f"Elliptical VaR PF2 {VaR_el_pf2.round(3)}")

## Copula (Gumbel) Approach:
# NOTE: THESE COPULA APPROACHES ARE SOMEWHAT SLOW. 
# Simulate from Copula
u_gumb1 = copulas.simul_gumbel(theta=theta1,dim=2,N_sim=N_sim)
# We then need to transform to X.
X_gumb1 = np.empty(shape=u_gumb1.shape) 
# Overwrite a matrix -> faster for these large matrices
X_gumb1[:,0] = u.inverse_GPD_emp(u_gumb1[:,0],np.sort(goog_neg),
                                  u_list[0],beta_list[0],gamma_list[0])
X_gumb1[:,1] = u.inverse_GPD_emp(u_gumb1[:,1],np.sort(msft_neg),
                                  u_list[1],beta_list[1],gamma_list[1])

u_gumb2 = copulas.simul_gumbel(theta=theta2,dim=2,N_sim=N_sim)
# We then need to transform to X.
X_gumb2 = np.empty(shape=u_gumb2.shape) 
# Overwrite a matrix -> faster for these large matrices
X_gumb2[:,0] = u.inverse_GPD_emp(u_gumb2[:,0],np.sort(mrk_neg),
                                  u_list[2],beta_list[2],gamma_list[2])
X_gumb2[:,1] = u.inverse_GPD_emp(u_gumb2[:,1],np.sort(idu_neg),
                                  u_list[3],beta_list[3],gamma_list[3])

# Tech 
L_cop_pf1 = L_fun((-1)*X_gumb1,S_0)
VaR_cop_pf1 = u.VaR(L_cop_pf1,var_thres)
print(f"Gumbel Copula VaR PF1 {VaR_cop_pf1.round(3)}")

# Other index. 
L_cop_pf2 = L_fun((-1)*X_gumb2,S_0)
VaR_cop_pf2 = u.VaR(L_cop_pf2,var_thres)
print(f"Gumbel Copula VaR PF2 {VaR_cop_pf2.round(3)}")

## Comonotonic copula (worst case).
# Simul
u_com1 = copulas.simul_frechet_bound_M(N_sim=N_sim)
# We then need to transform to X.
X_com1 = np.empty(shape=u_com1.shape) 
# Overwrite a matrix -> faster for these large matrices
X_com1[:,0] = u.inverse_GPD_emp(u_com1[:,0],np.sort(goog_neg),
                                  u_list[0],beta_list[0],gamma_list[0])
X_com1[:,1] = u.inverse_GPD_emp(u_com1[:,1],np.sort(msft_neg),
                                  u_list[1],beta_list[1],gamma_list[1])
u_com2 = copulas.simul_frechet_bound_M(N_sim=N_sim)
# We then need to transform to X.
X_com2 = np.empty(shape=u_com2.shape) 
# Overwrite a matrix -> faster for these large matrices
X_com2[:,0] = u.inverse_GPD_emp(u_com2[:,0],np.sort(mrk_neg),
                                  u_list[2],beta_list[2],gamma_list[2])
X_com2[:,1] = u.inverse_GPD_emp(u_com2[:,1],np.sort(idu_neg),
                                  u_list[3],beta_list[3],gamma_list[3])
# Tech 
L_cop_pf1 = L_fun((-1)*X_com1,S_0)
VaR_cop_pf1 = u.VaR(L_cop_pf1,var_thres)
print(f"Comonotonic Copula VaR PF1 {VaR_cop_pf1.round(3)}")

# Other index. 
L_cop_pf2 = L_fun((-1)*X_com2,S_0)
VaR_cop_pf2 = u.VaR(L_cop_pf2,var_thres)
print(f"Comonotonic Copula VaR PF2 {VaR_cop_pf2.round(3)}")


## Copula (Gaussian) Approach: 
# L_cop_pf1 = L_fun((-1)*X_gauss1,S_0)
# VaR_cop_pf1 = u.VaR(L_cop_pf1,var_thres)
# print(f"Gaussian Copula VaR PF1 {VaR_cop_pf1.round(3)}")

# # Other index. 
# L_cop_pf2 = L_fun((-1)*X_gauss2,S_0)
# VaR_cop_pf2 = u.VaR(L_cop_pf2,var_thres)
# print(f"Gaussian Copula VaR PF2 {VaR_cop_pf2.round(3)}")

# ## t-copula
# L_cop_pf1 = L_fun((-1)*X_t1,S_0)
# VaR_cop_pf1 = u.VaR(L_cop_pf1,var_thres)
# print(f"t-Copula VaR PF1 {VaR_cop_pf1.round(3)}")

# # Other index. 
# L_cop_pf2 = L_fun((-1)*X_t2,S_0)
# VaR_cop_pf2 = u.VaR(L_cop_pf2,var_thres)
# print(f"t-Copula VaR PF2 {VaR_cop_pf2.round(3)}")

# ## Countermonotonic copula (worst case).
# L_cop_pf1 = L_fun((-1)*X_coum1,S_0)
# VaR_cop_pf1 = u.VaR(L_cop_pf1,var_thres)
# print(f"Countermonotonic Copula VaR PF1 {VaR_cop_pf1.round(3)}")

# # Other index. 
# L_cop_pf2 = L_fun((-1)*X_coum2,S_0)
# VaR_cop_pf2 = u.VaR(L_cop_pf2,var_thres)
# print(f"Countermonotonic Copula VaR PF2 {VaR_cop_pf2.round(3)}")
