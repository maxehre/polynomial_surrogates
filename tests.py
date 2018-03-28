import numpy as np
import scipy.stats as ss
import surrogates_lib as MM
import matplotlib
import matplotlib.pyplot as plt
import example_applications as examples

# pick example ('beam'/'truss')

example = 'truss'

# ---------------------------------------------------------------------------
# input parameters
# ---------------------------------------------------------------------------

# global parameters
ned = 50     # # of points in design of experiments
nmc = int(2.5e5)  # # of points in MC reference
# parameters for PCE
ppce = 3      # max. polynomial order
q    = .4     # truncation exponent
Q    = .9999  # Stopping criterion: LOO Measure-of-fit

# parameters for LRA
plra = 3      # max. polynomial order
R    = 1      # max. rank

# load 1e6 reference samples & example dimension
data = np.load("data/{0}_validation_n_{1}.npz".format(example,nmc))
Ve   = data['Ve']
Ye   = data['Ye']
d    = data['d']

# ---------------------------------------------------------------------------
# standard-normal v < -> uniform space u
# ---------------------------------------------------------------------------

def V2U(v):
	return ss.norm.cdf(v, loc=0, scale=1)

def U2V(u):
	return ss.norm.ppf(u, loc=0, scale=1)

def X2V(x):
    if example == 'truss':
        return examples.X2V_truss(x)
    elif example == 'beam':
        return examples.X2V_beam(x)
 
# ---------------------------------------------------------------------------
# standard-normal space v < -> original space x
# ---------------------------------------------------------------------------

def V2X(v):
    if example == 'truss':
        return examples.V2X_truss(v)
    elif example == 'beam':
        return examples.V2X_beam(v)
    
def f(x):
    if example == 'truss':
        return examples.truss(x)
    elif example == 'beam':
        return examples.beam(x)


# ---------------------------------------------------------------------------
# run test
# ---------------------------------------------------------------------------

# Experimental Design
V = U2V(np.random.random((ned,d)))
Y = np.array([f(V2X(V[i,:])) for i in range(ned)])

# reformat if necessary
if np.size(np.shape(Y))==1:
    Y = np.expand_dims(Y, axis=1)

# build surrogates
#fpce     = MM.build_fpce(V, Y, ppce)
fpce     = MM.build_spce(V, Y, Q, ppce, q)
flra,_,_ = MM.build_cp(V, Y, R, plra)

# compute generalization error (MC estimate)
Ypce  = np.array([fpce(Ve[i,:]) for i in range(nmc)]).flatten()
Ylra  = np.array([flra(Ve[i,:]) for i in range(nmc)]).flatten()

err_pce = np.mean((Ye -  Ypce)**2)/np.var(Ye)
err_lra = np.mean((Ye -  Ylra)**2)/np.var(Ye)

print "\nGeneralization Error Comparison\n\n"\
        "PCE: g_e = {0}\n"\
        "LRA: g_e = {1}".format(err_pce,err_lra)

# ---------------------------------------------------------------------------
# density plots
# ---------------------------------------------------------------------------

de   = ss.kde.gaussian_kde(np.abs(Ye))
dpce = ss.kde.gaussian_kde(np.abs(Ypce))
dlra = ss.kde.gaussian_kde(np.abs(Ylra))
y    = np.linspace(min(np.abs(Ye)),max(np.abs(Ye)),100)


matplotlib.rcParams.update({'font.size': 18,'legend.fontsize': 18})
matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#plt.rc('text', usetex=True)

plt.semilogy(y, de(y),'-k',y, dpce(y),'or',y, dlra(y),'xg')
plt.ylim(1e-2, 1e3)
plt.title('$Densities$')
plt.xlabel('$u_{out}$')
plt.ylabel('$f_{u_{out}}$')
plt.legend(['DMC','PCE','LRA'])
plt.tight_layout()
