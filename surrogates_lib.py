import numpy as np
import scipy.special as sss
import sympy as sym
from sklearn import linear_model as sklm
from sympy.utilities.lambdify import lambdify
import itertools as itt
<<<<<<< HEAD
=======
import pdb
>>>>>>> a62e4489bba6d36502bbe86eec6d8800c23789f7


# ---------------------------------------------------------------------------
# Get index set of d-dimensional,tensorized polynomials of max. total order p
# according to fractional q-norm
# ---------------------------------------------------------------------------

def truncated_multi_index(d, p, q_trunc):
    N = int(sss.binom(d+p,p))
    alpha = np.zeros((N,d))
    if d == 1:
        alpha = range(p+1)
        return np.reshape(np.array(alpha),(-1,1))
    else:
        lblast = 1
        for q in range(1,p+1):
            s = np.array(list(itt.combinations(range(1,d+q), d-1)))
            slen = np.shape(s)[0]
            s1 = np.zeros((slen, 1))
            s2 = (d + q) + s1
            tmp = np.flipud(np.diff(np.concatenate((s1,s,s2),axis=1),n=1,axis=1)) - 1
            q_order = np.sum(tmp**q_trunc,1)**(1/q_trunc)
            tmp_trunc = tmp[(q_order - 1e-15 <= p),:]
            slen_trunc = np.shape(tmp_trunc)[0]
            alpha[range(lblast,lblast+slen_trunc),:] = tmp_trunc
            lblast = lblast + slen_trunc
        
        alpha = alpha[:lblast,:]
        return alpha.astype(int)
    
# ---------------------------------------------------------------------------
# Computes 1-dimensional stochastic, normalized Hermite polynomials
# ---------------------------------------------------------------------------

def scalar_Hermite_polynomials(p):

    # 1-dimensional recursive & symbolic Hermite polynomials
    x = sym.symbols('x')

    He = [sym.symbols('1')]
    He.append(x)

    for i in range(2,int(p)+1):
        He.append(sym.simplify((x*He[i-1] - (i-1)*He[i-2])))

    return He

# ---------------------------------------------------------------------------
# Computes d-dimensional tensorized Hermite polynomials
# ---------------------------------------------------------------------------

def multi_Hermite_polynomials(d, p, q):

    # 1-dimensional recursive & symbolic Hermite polynomials
    x = sym.symbols('x')

    # get 1-D base polynomials (Hermite)
    He = scalar_Hermite_polynomials(p)

    # get truncated index set
    alpha_trunc = truncated_multi_index(d,p,q)
    
    # d-dimensional Hermite polynomials according to alpha_trunc
    n_psi = np.shape(alpha_trunc)[0]
    X = sym.symbols('x0:%d'%d)

    #np.sqrt(np.math.factorial(i))

    psi = []
    #f_psi = []
    for i in range(n_psi):
        norm = np.prod([np.math.factorial(alpha_trunc[i, j]) for j in range(d)])**.5
        psi.append(sym.prod([He[alpha_trunc[i, j]].subs(x,X[j]) for j in range(d)])/norm)
       # f_psi.append(lambdify([X], psi[i], 'numpy'))
    return (np.array(psi),alpha_trunc)

# ---------------------------------------------------------------------------
# construct full PCE with hermitian polyonmial base and OLS
# ---------------------------------------------------------------------------

def build_fpce(X, Y, p):

    N,d = np.shape(X)
        
    # get polynomial base
    psi,alpha = multi_Hermite_polynomials(d, p, 1)

    X_sym = sym.symbols('x0:%d' % d)

    #f_psi = lambdify([X_sym], psi, 'numpy')
    #Psi = np.array([f_psi(X[i, :]) for i in range(N)])

    # build data matrix Psi
    
    P = np.shape(psi)[0]
    f_psi = np.array([lambdify([X_sym], psi[i], 'numpy') for i in range(P)])
    Psi = np.array([[f_psi[j](X[i, :])for j in range(P)] for i in range(N)])
    
    # information matrix
    M = np.dot(Psi.T,Psi)

    # inverse information matrix
    MI = np.linalg.inv(M)
    
    # projection matrix
    PP = Psi.dot(MI).dot(Psi.T)
    
    # annihilator matrix
    AA = np.eye(N) - PP
    
    # LOO-measure of fit Q^2
    h = np.diag(PP);
    
    eps_loo = np.mean(((AA.dot(Y))/(1-h))**2)/np.var(Y) * (1 + np.trace(MI))*N/(N-P);
    
    coeffs = np.dot(MI , np.dot(Psi.T , Y))
        
    f_fpce = lambdify([X_sym],np.dot(coeffs.flatten(),psi), 'numpy')
    
    return f_fpce,eps_loo


# ---------------------------------------------------------------------------
# driver for sparse polynomial chaos expansions
# ---------------------------------------------------------------------------

def build_spce(X, Y, Q_target, p, q):

    N,dx = np.shape(X)
    N,dy = np.shape(Y)
    
    spce_vec = []
    
    for rowY in Y.T:
        
        spce = spce_main(X, rowY.T, Q_target, p, q)
                
        spce_vec.append(spce)
            
    X_sym = sym.symbols('x0:%d' % dx)

    f_spce = lambdify([X_sym], spce_vec, 'numpy')

    # # version for linux cluster anaconda env: cannot handle array lambdification   
    # P = np.shape(psi)[0]
    # f_spce = np.array([lambdify([X_sym], spce_vec[i], 'numpy') for i in range(P)])
    
    return f_spce


# ---------------------------------------------------------------------------
# driver for low-rank app in canonical polyadics format
# ---------------------------------------------------------------------------

def build_cp(X, Y, R, p):

    N,dx = np.shape(X)
    N,dy = np.shape(Y)

    if np.shape(p) == ():
        p = [p] * dx
        
    lra_vec = []
    
    for rowY in Y.T:
       # lra_list = cp(X, rowY.T, R, p)
        cv3_opt,R_opt = cvn(X,rowY.T,R,p,3)
        
        lra_list,z,b = cp_main(X,rowY.T,R_opt,p)
        
        lra_vec.append(lra_list[-1])
            
    X_sym = sym.symbols('x0:%d' % dx)

    f_lra = lambdify([X_sym], lra_vec, 'numpy')

    # version for linux cluster anaconda env: cannot handle array lambdification
    # f_lra = np.array([lambdify([X_sym], lra_vec[i], 'numpy') for i in range(dy)])
    
    return (f_lra,z,b)


# ---------------------------------------------------------------------------
# construct sparse PCE with Hermitian polynomial base and least-angle reg:
#
# Blatman, G. and B. Sudret (2011). 
# Adaptive sparse polynomial chaos expansion based on least-angle regression.
# Journal of Computational Physics 230(6), 2345 - 2367.                                             
# ---------------------------------------------------------------------------

def spce_main(X, Y, Q_target, pmax, q):

    Y = np.expand_dims(Y, axis=1)
    
    N,dx = np.shape(X)

    Psi = np.empty(pmax, dtype=object)
    psi = np.empty(pmax, dtype=object)
    A_p = np.empty(pmax, dtype=object)
    
    eps_LOO_min = np.ones(pmax)*987654321
    I = np.empty(pmax, dtype = int)

    for j in range(pmax):

        p = j+1
            
        # get polynomial base
        psi[j],alpha = multi_Hermite_polynomials(dx, p, q)
        
        P = np.shape(psi[j])[0]

        # build data matrix Psi
        X_sym = sym.symbols('x0:%d' % dx)
    
        #  f_psi  = lambdify([X_sym], psi[j], 'numpy')
        #  Psi[j] = np.array([f_psi(X[i, :]) for i in range(N)])
        
        # version for linux cluster anaconda env: cannot handle array lambdification
        f_psi  = np.array([lambdify([X_sym], psi[j][k], 'numpy') for k in range(P)])
        Psi[j] = np.array([[f_psi[k](X[l, :])for k in range(P)] for l in range(N)])
        
        # least-angle regression
        reg = sklm.Lars(verbose=False)
        reg.fit((Psi[j] - np.mean(Psi[j],axis=0))/np.linalg.norm(Psi[j],axis=0), Y - np.mean(Y))
        A_p[j] = np.array([0] + reg.active_)

    
        # successively add regressors for optimal sparse expansion
        nA = np.shape(A_p[j])[0]
    
        eps_LOO = np.empty(nA)
    
        for i in range(nA):
            
            # current active set
            A_tmp = A_p[j][:(i+1)]
                       
            # information matrix
            M  = Psi[j][:,A_tmp].T.dot(Psi[j][:,A_tmp])
            
            # inverse information matrix
            MI = np.linalg.inv(M)
            
            # projection matrix
            PP  = Psi[j][:,A_tmp].dot(MI).dot(Psi[j][:,A_tmp].T)
            
            # annihilator matrix
            AA  = np.eye(N) - PP
            
            # LOO-measure of fit Q^2
            h          = np.diag(PP);
            eps_LOO[i] = np.mean(((AA.dot(Y))/(1-h))**2)/np.var(Y) * (1 + np.trace(MI))*N/(N-P);
            
                   
        eps_LOO_min[j] = np.min(abs(eps_LOO))
        I[j]           = np.argmin(abs(eps_LOO)) + 1
           
        if np.abs(eps_LOO_min[j]) < 1 - Q_target:
            print('Target accuracy Qtgt_sq reached.')
            break
        elif (j > 1) & (j < nA - 1):
            if (eps_LOO_min[j] > eps_LOO_min[j-1]) & (eps_LOO_min[j-1] > eps_LOO_min[j-2]):
                print('Overfitting: Forward loop stopped at poylnomial order '+str(j))
                break
        elif j == nA - 1:
            print('All terms ('+str(i)+') included in expansion: Forward loop stopped')
    
    #err_loo_opt = np.min(np.abs(eps_LOO_min))
    J           = np.argmin(np.abs(eps_LOO_min)).astype(int)

    # compute optimal PCE expansion
    A_opt   = A_p[J][:I[J]]
    Psi_opt = Psi[J]
    psi_opt = psi[J]
    a       = np.linalg.solve(Psi_opt[:,A_opt].T.dot(Psi_opt[:,A_opt]) , Psi_opt[:,A_opt].T.dot(Y))
    fpce    = a.T.dot(psi_opt[A_opt])
               
    return fpce



# ---------------------------------------------------------------------------
# main: construct CP with Hermitian polyonmial base and ALS
#
# Konakli, K. and B. Sudret (2016b). 
# Polynomial meta-models with canonical low-rank approximations:
# Numerical insights and comparison to sparse polynomial chaos expansions. 
# Journal of Computational Physics 321, 1144 - 1169.
# ---------------------------------------------------------------------------

def cp_main(X, Y, R, p):

    #Y = np.expand_dims(Y, axis=1)
    N,dx = np.shape(X)
    
    x = sym.symbols('x')
    He = scalar_Hermite_polynomials(max(p))

    # convert to lambda functions
    f_He = np.array([lambdify(x, He[index], 'numpy') for index in range(max(p) + 1)])

    # Alternating least squares

    # loop parameters
    err_tol = 1e-8
    Imax = 50 * dx

    # Initialization
    v = np.ones((N, dx, R))
    z = np.empty((dx, R), dtype=object)
    Psi = np.empty(dx, dtype=object)
    w = np.empty((N,R))
    
    lra_list = []
    # Precompute regression matrices Psi
    for i in range(dx):
        Psi[i] = np.array([f_He[order](X[:, i]) for order in range(1, p[i] + 1)]).T
        Psi[i] = np.concatenate((np.ones((N, 1)), Psi[i]), axis=1)

    # Residual
    Yres = Y
    
    # Loop
    for r in range(R):

        err0 = 1e4
        I = 0

        while True:

            # build index i that runs in between 1:d during the loop
            i = np.mod(I, dx)
            if (i == 0 & I>0):
                i = dx

            # fix all components but i-th
            c = np.reshape(np.prod(v[:, np.setdiff1d(range(dx), i), r], axis=1)[:],(-1,1))

            # find polynomial coefficients z
            CPsi = c*Psi[i]
            z[i,r] = np.linalg.solve(CPsi.T.dot(CPsi), CPsi.T.dot(Yres)).flatten()

            # compute ith factor of the rth rank - 1 tensor
            v[:, i, r] = Psi[i].dot(z[i, r])

            # compute the normalized empirical error
#            err = np.mean((Yres - (c * v[:, i, r]))**2)/np.var(Yres)
            err = np.mean((Yres - (c.T * v[:, i, r]).T)**2)/np.var(Yres)
            # update
            if (((I < Imax) & (abs(err0 - err) > err_tol)) | (I < dx)):
                err0 = err
                I = I + 1
            else:
                w[:, r] = c.flatten() * v[:, i, r]
                break

        b = np.linalg.solve(w[:,:r+1].T.dot(w[:,:r+1]) , w[:,:r+1].T.dot(Y))

        X_sym = sym.symbols('x0:%d' % dx)

        lra = 0
        for j in range(r+1):
            f_w = 1
            for k in range(dx):
                f_w = f_w * np.dot(z[k,j],[He[order].subs('x', X_sym[k]) for order in range(p[k]+1)])
            
            lra = lra + b[j]*f_w
 
        lra_list.append(lra)
       
        hatY = np.dot(w[:,:r+1], b)
        # compute new residual
        Yres = Y - hatY
    
    return (lra_list,z,b)


# ---------------------------------------------------------------------------
# compute n-fold cross validation error for CP format
# ---------------------------------------------------------------------------

def cvn(X,Y,R,p,n):
    
    N,dx = np.shape(X)
    
    X_sym = sym.symbols('x0:%d' % dx)
    
    edges = np.round(np.linspace(0, N, num=n+1)).astype(int)
    
    err = np.empty((R,n))
    
    for i in range(n):
        
        idval  = range(edges[i],edges[i+1])
        idpart = list(set(range(N)) - set(idval))

        Xpart  = X[idpart,:]
        Ypart  = Y[idpart]
        Xval   = X[idval,:]
        Yval   = Y[idval]
        
        lra_list,_,_ = cp_main(Xpart, Ypart, R, p)
        
        f_lra    = lambdify([X_sym], lra_list, 'numpy')
        hatY     = [f_lra(Xval[j,:]) for j in range(np.shape(Xval)[0])]    
        
        # # version for linux cluster anaconda env: cannot handle array lambdification
        # f_lra = np.array([lambdify([X_sym], lra_list[i], 'numpy') for i in range(R)])
        # hatY = [[f_lra[k](Xval[j,:]) for k in range(R)]for j in range(np.shape(Xval)[0])] 

        err[:,i] = np.mean((np.expand_dims(Yval, axis=1) - hatY)**2,axis=0)/np.var(Yval)
        
    
    mean_err = np.mean(err,axis=1)    
    R_opt    = np.argmin(np.array(mean_err)) + 1
    err_opt  = mean_err[R_opt - 1]
    
    return (err_opt,R_opt)
        
        
# ---------------------------------------------------------------------------
# computes variance-based polynomial chaos sensitivites
#
# Sudret, B. (2008). 
# Global sensitivity analysis using polynomial chaos expansions.
# Reliability Engineering And System Safety 93(7), 964 - 979.
# ---------------------------------------------------------------------------

def pce_sensitivities(alpha,coeffs):

    P, d = np.shape(alpha)

    # truncate 0-order index (output sample mean -> contributes no variance)
    coeffs = coeffs[1:]
    alpha = alpha[1:,:]
    indicator_set = alpha[:,:] != 0

    unique, unique_indices, inverse_indices = np.unique(indicator_set, axis=0, return_index=True, return_inverse=True)

    total_variance = np.sum(coeffs ** 2)
    V = np.bincount(unique_indices[inverse_indices], weights=coeffs**2)/total_variance
    S = np.dot(coeffs ** 2 , indicator_set)/total_variance
    # total Sobol indices
    return (V,S)
