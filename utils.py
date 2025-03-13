import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm ,binom, uniform, t, expon, invgamma,gamma,levy_stable
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mnorm
from scipy.stats import multivariate_t as mt
from scipy.special import factorial

def generel_inverse(L,F_L,alpha): 
    L_ordered = np.sort(L)
    # See definition in ex. 3.2.
    F_L_ualpha = np.where(F_L>=alpha,F_L,1)
    idx = np.argmin(F_L_ualpha) 
    # Will also find lowest in case of ties by default
    # Note, due to zero indexing, subtract one. 
    q_alpha = L_ordered[idx]

    return q_alpha

def VaR(L,alpha):
    L_order_desc = - np.sort(-L)
    var_idx=int(np.floor(len(L)*(1-alpha)))
    VaR = L_order_desc[var_idx]

    return VaR

def ES(L,alpha):
    L_order_desc = - np.sort(-L)
    var_idx = int(np.floor(len(L)*(1-alpha)))
    ES = np.sum(L_order_desc[:var_idx+1]) / (var_idx+1)

    return ES

### EXTREME VALUE THEORY

def HillEstimator(X,k):
    X_sorted = - np.sort(-X)
    k_0index = k - 1
    X_k = X_sorted[k_0index]
    
    mean =  np.sum(np.log(X_sorted[:k]) - np.log(X_k)) / k
    alpha = mean**(-1)
    return alpha

def emp_mean_exess(X,k):
    x_k = X[k] # get k'th entry - zero at first.
    return  np.sum(X[:k+1]-x_k) / (k+1) # due to zero idx.

def GPD_fun(gamma,beta,x):
    return 1- (1+gamma*x/beta)**(-1/gamma)
    
def GPD_neg_loglike(params,N_u,Z):
    beta,gamma = params
    log_like = -N_u * np.log(beta) - (gamma + 1) * np.sum(
        np.log(1+gamma*Z/beta))/gamma
    return - log_like

# Mandatory assignment -> Create fct combining emp. distr
# and Pareto.
def GPD_emp(X,u,beta,gamma):
    n_obs = X.shape[0]
    N_u = np.sum(X > u)
    F_margin = np.zeros(n_obs)
    for i in range(0,n_obs):
        # Empirical if below.
        if X[i] <= u:
            F_margin[i] = np.sum(X[:i+1] <= u) /  n_obs
        else: 
            F_n_u_bar = 1 - GPD_fun(gamma,beta,X[i]-u)
            F_n_bar = N_u / n_obs
            F_bar_upx = F_n_bar * F_n_u_bar
            F_bar_upx_upp = 1 - F_bar_upx
            F_margin[i] = F_bar_upx_upp

    return F_margin

def inverse_GPD_emp(U,X_emp,u,beta,gamma):
    n = X_emp.shape[0]
    x_sorted = - np.sort( - X_emp) # Sort decending. 
    k_idx = np.argmin(x_sorted>u) - 1 
    N_u = np.sum(X_emp > u)
    F_margin = GPD_emp(X_emp,u,beta,gamma)
    F_margin_sort = - np.sort(-F_margin) 
    out = np.zeros(U.shape[0])
    for i in range(out.shape[0]):
    # If slightly above empirical section -> Inverse GPD
        if U[i] > F_margin_sort[k_idx]:
            out[i] = u + beta*((n*(1-U[i])/N_u)**(-gamma)-1)/gamma
        else:
            out[i] = generel_inverse(X_emp,F_margin,U[i])
    return out

# Alternative approach for this CI section dessirable. 
def CI_lossRV(p,n,beta):
    idx_list = np.array([i for i in range(1,n)])
    P_Ygeqj = 1-binom.cdf(idx_list,n,p)
    j = np.argmax(np.where(P_Ygeqj<=beta/2,P_Ygeqj,0))
    P_Yleqj = binom.cdf(idx_list,n,p)
    i = np.argmax(np.where(P_Yleqj<=beta/2,P_Yleqj,0))
    return (j,i)

# Regarding Importance sampling 
def Emp_distr_tail_expo_shift(u,L,X,N,C,xi):
    F_L = np.sum(1/(C*np.exp(X @ xi)) * (L > u)) / N   
    return F_L

def Emp_tail_diff(u,L,X,N,C,xi,alpha):
    return np.abs(Emp_distr_tail_expo_shift(u,L,X,N,C,xi) - alpha)

def emp_tail_solve(L,X,N,C,xi, alpha):
    # Use relative difference to obtain tmore stable optimization.
    iv_num = minimize(fun=Emp_tail_diff,
                      x0=L[0],args = (L,X,N,C,xi, alpha),
                        method='Nelder-Mead')
    out = iv_num.x
    return out



### MULTIVARIATE T LOG LIKELIHOOD 
# TODO: REWRITE THIS.

def multivariate_t_log_likelihood(params, data):
    """
    Compute the negative log-likelihood of a multivariate t-distribution.
    """
    d = data.shape[1]  # Number of dimensions
    nu = params[0]  # Degrees of freedom
    mu = params[1:d+1]  # Location vector
    A = np.tril(np.reshape(params[d+1:], (d, d)))  # Lower-triangular Cholesky factor of Covariance
    #sigma = np.tril(np.reshape(params[d+1:], (d, d)))  # Lower-triangular Cholesky factor of Covariance
    sigma = A @ A.T  # Reconstruct covariance matrix
    dist = mt(loc=mu, shape=sigma, df=nu)
    
    return -np.sum(dist.logpdf(data))

def multivariate_t_log_likelihood_dfs(params, data, mu, sigma):
    """
    Compute the negative log-likelihood of a multivariate t-distribution 
    while keeping mu (mean) and sigma (covariance) fixed.

    Parameters:
        - params: Array containing only the degrees of freedom (nu).
        - data: Stock return data (n_samples, d).
        - mu: Fixed mean vector (d,).
        - sigma: Fixed covariance matrix (d, d).

    Returns:
        - Negative log-likelihood value.
    """
    
    d = data.shape[1]  # Number of dimensions
    nu = params[0]  # Degrees of freedom (only parameter being optimized)

    # Ensure nu > 2 for finite variance
    if nu <= 2:
        return np.inf  

    dispersion_matrix = sigma * (nu - 2) / nu  # Convert covariance to dispersion

    # Define Multivariate t-distribution with fixed mu and dispersion matrix
    dist = mt(df=nu, loc=mu, shape=dispersion_matrix)

    # Compute and return negative log-likelihood
    return -np.sum(dist.logpdf(data))

### FOR INVESTIGATING COMONOCITY AND COUNTERMONOCITY.

def kendalls_tau(X):
    '''
    Takes X \in R^(n,2) i.e. assumes bivariate R.V
    '''
    n = X.shape[0]
    k = 2 # bivariate comparisons. 
    binom_coeff = factorial(n,exact=True) / (
        factorial(k,exact=True)*factorial(n-k,exact=True))
    sign_sum = 0
    for i in range(0,n):
        sign_sum += np.sum(np.sign((X[i,0]-X[i:,0])*(X[i,1]-X[i:,1]))) 
    return sign_sum / binom_coeff

# Merge sort for efficient kendalls tau. 
def merge_sort_count(arr):
    """
    Modified merge sort to count the number of inversions (discordant pairs).
    """
    if len(arr) <= 1:
        return arr, 0
    
    mid = len(arr) // 2
    left, left_inv = merge_sort_count(arr[:mid])
    right, right_inv = merge_sort_count(arr[mid:])
    
    merged = []
    i = j = inv_count = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
            inv_count += len(left) - i
    
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged, left_inv + right_inv + inv_count

def kendalls_tau_eff(X):
    """
    Compute Kendall's Tau correlation coefficient efficiently using Merge Sort.
    """
    x,y = X[:,0], X[:,1]
    assert len(x) == len(y), "Both lists must have the same length."
    
    n = len(x)
    pairs = list(zip(x, y))
    pairs.sort()
    
    sorted_y = [val for _, val in pairs]
    _, discordant = merge_sort_count(sorted_y)
    
    total_pairs = (n * (n - 1)) // 2
    concordant = total_pairs - discordant
    
    tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
    return tau

### Copula class containing functionality to simulate from 

class Copulas():
    def __init__(self):        
        return None
    
    # Create Various copula functions as fct of params

    # Gumbel Copula.
    def C_gumbel(self,theta,U):
        return np.exp(-(np.sum((-np.log(U))**theta,axis=1))**(1/theta))

    def simul_gumbel(self,theta,dim,N_sim):
        # Step 1: Identify the distr. G having Laplace Transform 
        # psi = theta^{-1}
        # Step 2: Simulate V~G
        gamma = (np.cos(np.pi/(2*theta)))**theta
        V = levy_stable.rvs(alpha=1/theta,beta=1,scale=gamma,
                            size=N_sim).reshape((N_sim,1))
        # Step 3: Generate U_1,...,U_d
        U = uniform.rvs(size=dim*N_sim).reshape((N_sim,dim))
        # Get W
        t = - np.log(U) / V
        W = np.exp(-t**(1/theta))
        # Copula value
        #C_gumbel = self.C_gumbel(theta,U)

        return W #,C_gumbel 

    # Clayton Copula - > Make general version.
    def C_clayton(self,theta,U,dim):
        return (np.sum(U**(-theta),axis=1)-dim+1)**(-1/theta)

    # We also need to simulate from these. 
    # Note this follows Example 10.6 and prop 10.8
    def simul_clayton(self,theta,dim,N_sim):
        # Step 1: Identify the distr. G having Laplace Transform 
        # psi = theta^{-1}
        # Here G is Gamma w. alpha=beta=1/theta
        # See the documentation
        # Step 2: Simulate V~G
        V = gamma.rvs(a=1/theta,scale=1/theta,size=N_sim).reshape(
            (N_sim,1))
        # Step 3: Generate U_1,...,U_d
        U = uniform.rvs(size=dim*N_sim).reshape((N_sim,dim))
        # Get W
        t = - np.log(U) / V
        W = (theta * t + 1)**(-1/theta)
        # Copula value
        #C_clayton = self.C_clayton(theta,U,dim)

        return W #,C_clayton

    def simul_Gaussian(self,rho_mat,N_sim):
        d = rho_mat[0,:].shape[0]
        mu = np.zeros(d)
        X = mnorm.rvs(mean=mu,cov=rho_mat,size=N_sim)
        U = norm.cdf(X,0,1)
        cop_in = norm.ppf(U)
        #C_gauss = mnorm.cdf(cop_in,mean=mu,cov=rho_mat)
        # Return both obs and fct val. 
        return U #,C_gauss)
    
    def simul_t_distr(self,rho_mat,nu,N_sim):
        d = rho_mat[0,:].shape[0]
        mu = np.zeros(d)
        X = mt.rvs(loc=mu,shape=rho_mat,df=nu,size=N_sim)
        U = t.cdf(X,df=nu)
        # Then we need to tranform U in principle (will be x)
        cop_in = t.ppf(U,df = nu)
        #C_nu_sigma = mt.cdf(cop_in,loc=mu,shape=rho_mat,df=nu)
        # Return both obs and fct val. 
        return U #(U,C_nu_sigma)
    

    
### Marginal distribution plots -> 1 Gaussian and 3 t-distributions 
def marginal_plots(X,fig_name,df1,df2,df3,df4,df5,markersize):
    X_sorted = np.sort(X)
    #X_mean = np.mean(X_sorted)
    #X_std = np.std(X_sorted)
    n = X_sorted.shape[0]
    p_vec = np.array([(i)/(n+1) for i in range(1,n+1)])
    # NOTE: I DEVIATE FROM THE NOTES, BUT QQ-PLOTS GO FROM SMALLEST TO LARGETST. 
    # Do 4 plots. 1 for normal and three t-distr. 
    q_norm = norm.ppf(p_vec ) #,loc=X_mean,scale=X_std)
    q_t1 = t.ppf(p_vec,df=df1)#,loc=X_mean,scale=X_std)
    q_t2 = t.ppf(p_vec,df=df2)#,loc=X_mean,scale=X_std)
    q_t3= t.ppf(p_vec,df=df3)#,loc=X_mean,scale=X_std)
    q_t4= t.ppf(p_vec,df=df4)#,loc=X_mean,scale=X_std)
    q_t5= t.ppf(p_vec,df=df5)#,loc=X_mean,scale=X_std)


    # Plotting: 
    fig, ax = plt.subplots(nrows=3,ncols=2)
    # Normal distributions
    ax1 =  ax[0,0]
    ax1.scatter(q_norm,X_sorted, color='blue',s=markersize)
    #ax1.plot(q_norm,q_norm,color='black',label='Reference line if distr true')
    ax1.grid()
    ax1.set_xlabel('Theoretical quantiles')
    ax1.set_ylabel('Empirical quantiles')
    ax1.set_title(f'Normal distribution')
    # t-distributions. 
    ax2 =  ax[0,1]
    ax2.scatter(q_t1,X_sorted, color='blue',s=markersize)
    #ax2.plot(q_t1,q_t1,color='black',label='Reference line if distr true')
    ax2.grid()
    ax2.set_xlabel('Theoretical quantiles')
    ax2.set_ylabel('Empirical quantiles')
    ax2.set_title(f'T-distribution, df={df1}')
    ax3 =  ax[1,0]
    ax3.scatter(q_t2,X_sorted, color='blue',s=markersize)
    #ax3.plot(q_t2,q_t2,color='black',label='Reference line if distr true')
    ax3.grid()
    ax3.set_xlabel('Theoretical quantiles')
    ax3.set_ylabel('Empirical quantiles')
    ax3.set_title(f'T-distribution, df={df2}')
    ax4 =  ax[1,1]
    ax4.scatter(q_t3,X_sorted, color='blue',s=markersize)
    #ax4.plot(q_t3,q_t3,color='black',label='Reference line if distr true')
    ax4.grid()
    ax4.set_xlabel('Theoretical quantiles')
    ax4.set_ylabel('Empirical quantiles')
    ax4.set_title(f'T-distribution, df={df3}')
    ax5 =  ax[2,0]
    ax5.scatter(q_t4,X_sorted, color='blue',s=markersize)
    #ax5.plot(q_t4,q_t4,color='black',label='Reference line if distr true')
    ax5.grid()
    ax5.set_xlabel('Theoretical quantiles')
    ax5.set_ylabel('Empirical quantiles')
    ax5.set_title(f'T-distribution, df={df4}')
    ax6 =  ax[2,1]
    ax6.scatter(q_t5,X_sorted, color='blue',s=markersize)
    #ax6.plot(q_t5,q_t5,color='black',label='Reference line if distr true')
    ax6.grid()
    ax6.set_xlabel('Theoretical quantiles')
    ax6.set_ylabel('Empirical quantiles')
    ax6.set_title(f'T-distribution, df={df5}')
    fig.tight_layout()
    plt.savefig(f'Figures/{fig_name}.png')
    plt.show()