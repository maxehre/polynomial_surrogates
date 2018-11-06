import numpy as np
import scipy.stats as ss
import surrogates_lib_polyval as MM
import matplotlib
import matplotlib.pyplot as plt
import example_applications as examples
import pdb
# pick example ('beam'/'truss')

example = 'truss'

np.random.seed(1)
# ---------------------------------------------------------------------------
# input parameters
# ---------------------------------------------------------------------------

# global parameters
ned = int(5e1)     # # of points in design of experiments
nmc = int(1e6)  # # of points in MC reference
# parameters for PCE
ppce = 5      # max. polynomial order
q    = .4     # truncation exponent
Q    = .9999  # Stopping criterion: LOO Measure-of-fit

# parameters for LRA
plra = 5      # max. polynomial order
R    = 5      # max. rank


# load 1e6 reference samples & example dimension
# Ve = U2V(np.random.random((nmc,d)))
# Ye = np.array([f(V2X(Ve[i,:])) for i in range(nmc)])

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
    elif example == 'ishigami':
        return examples.X2V_ishigami(x)
 
# ---------------------------------------------------------------------------
# standard-normal space v < -> original space x
# ---------------------------------------------------------------------------

def V2X(v):
    if example == 'truss':
        return examples.V2X_truss(v)
    elif example == 'beam':
        return examples.V2X_beam(v)
    elif example == 'ishigami':
        return examples.V2X_ishigami(v)
    
def f(x):
    if example == 'truss':
        return examples.truss(x)
    elif example == 'beam':
        return examples.beam(x)
    elif example == 'ishigami':
        return examples.ishigami(x)

# ---------------------------------------------------------------------------
# run test
# ---------------------------------------------------------------------------

# Experimental Design
V = U2V(np.random.random((ned,d)))
Y = np.array([f(V2X(V[i,:])) for i in range(ned)])

# reformat if necessary
if np.size(np.shape(Y))==1:
    Y = np.expand_dims(Y, axis=1)

spce     = MM.build_spce(V, Y, Q, ppce, q)
flra,z,b = MM.build_cp(V, Y, R, plra,'R')

# build surrogates
# fpce,_   = MM.build_fpce(V, Y, ppce)
#fpce     = MM.build_spce(V, Y, Q, ppce, q)

pdb.set_trace()

# compute generalization error (MC estimate)
Ypce = spce[0](Ve).flatten()
Ylra = flra[0](Ve).flatten()

err_pce = np.mean((Ye -  Ypce)**2)/np.var(Ye)
err_lra = np.mean((Ye -  Ylra)**2)/np.var(Ye)

print("\nGeneralization Error Comparison\n\n"\
        "PCE: g_e = {0}\n"\
        "LRA: g_e = {1}".format(err_pce,err_lra))


# ---------------------------------------------------------------------------
# density plots
# ---------------------------------------------------------------------------

de   = ss.kde.gaussian_kde(np.abs(Ye))
dpce = ss.kde.gaussian_kde(np.abs(Ypce))
dlra = ss.kde.gaussian_kde(np.abs(Ylra))
y    = np.linspace(min(np.abs(Ye)),max(np.abs(Ye)),100)


matplotlib.rcParams.update({'font.size': 18,'legend.fontsize': 18})
plt.semilogy(y, de(y),'-k',y, dpce(y),'or',y, dlra(y),'xg')
plt.ylim(1e-2, 1e3)
plt.title('Densities')
plt.xlabel('$u_{out}$')
plt.ylabel('$\pi_{u_{out}}$')
plt.legend(['DMC','PCE','LRA'])
plt.tight_layout()
plt.show()