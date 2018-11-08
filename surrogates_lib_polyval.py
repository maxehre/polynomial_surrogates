import numpy as np
import scipy.special as sss
from sklearn import linear_model as sklm
import itertools as itt
import pdb
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

def Hermite_polynomials(p):

    He = np.zeros((p+1,p+1))
    He[0,-1] = 1
    He[1,-2] = 1

    for n in range(2,int(p)+1):
        He[n,:] = np.append(He[n-1][1:],0) - (n-1)*He[n-2]

    return He


# ---------------------------------------------------------------------------
# driver for sparse polynomial chaos expansions
# ---------------------------------------------------------------------------

def build_fpce(X, Y, Q_target, p):

    N,dx = np.shape(X)
    N,dy = np.shape(Y)
    
    spce_vec = []
    
    for rowY in Y.T:
        
        spce,coeffs,alpha = fpce_main(X, rowY.T, Q_target, p)
                
        spce_vec.append(spce)
            
    return spce_vec,coeffs,alpha

# ---------------------------------------------------------------------------
# driver for sparse polynomial chaos expansions
# ---------------------------------------------------------------------------

def build_spce(X, Y, Q_target, p, q):

    N,dx = np.shape(X)
    N,dy = np.shape(Y)
    
    spce_vec = []
    
    for rowY in Y.T:
        
        spce,coeffs,alpha = spce_main(X, rowY.T, Q_target, p, q)
                
        spce_vec.append(spce)
            
    return spce_vec,coeffs,alpha


# ---------------------------------------------------------------------------
# driver for low-rank app in canonical polyadics format
# ---------------------------------------------------------------------------

def build_cp(X, Y, R, p,adaptation):

    N,dx = np.shape(X)
    N,dy = np.shape(Y)

    if np.shape(p) == ():
        p = np.array([p] * dx).astype(int)
        
    lra_vec = []
    
    k = 0
    for rowY in Y.T:
        
        k = k + 1
        
       # lra_list = cp(X, rowY.T, R, p)
        cvn_opt,r_opt,p_opt = cvn(X,rowY.T,R,p,3,adaptation)
        
        print("For component {0}, selected rank {1} & degree {2} with CV score {3}".format(k,r_opt,p_opt,cvn_opt))

        lra,z,b,_,Yhat = cp_main(X,rowY.T,r_opt,np.array([p_opt] * dx).astype(int),[],[],[])
        
        lra_vec.append(lra)
    
    return (lra_vec,z,b)


# ---------------------------------------------------------------------------
# construct full PCE with Hermitian base polynomials              
# ---------------------------------------------------------------------------

def fpce_main(X, Y, Q_target, pmax):

    Y = np.expand_dims(Y, axis=1)
    
    N,dx = np.shape(X)

    Psi = np.empty(pmax, dtype=object)
    alpha = np.empty(pmax, dtype=object)
    
    eps_LOO = np.ones(pmax)*987654321

    for j in range(pmax):

        p = j+1
        
        # get polynomial base
        He = Hermite_polynomials(p)
            
        # get truncated index set
        alpha[j] = truncated_multi_index(dx, p, 1)
        P        = np.shape(alpha[j])[0]
        
        # compute data matrix
        norm   = np.sqrt(np.prod(sss.factorial(alpha[j]),axis=1))
        Psi[j] = np.prod([np.array([np.polyval(He[entry,:],X[:,dim]) 
                 for dim,entry in enumerate(row)]) for row in alpha[j]],axis=1).T/norm

        # information matrix
        M  = Psi[j].T.dot(Psi[j])
        
        # inverse information matrix
        MI = np.linalg.inv(M)
        
        # projection matrix
        PP  = Psi[j].dot(MI).dot(Psi[j].T)
        
        # annihilator matrix
        AA  = np.eye(N) - PP
        
        # LOO-measure of fit Q^2
        h          = np.diag(PP);
        eps_LOO[j] = np.mean(((AA.dot(Y))/(1-h))**2)/np.var(Y) * (1 + np.trace(MI))*N/(N-P);
           
        J = j
        if np.abs(eps_LOO[j]) < 1 - Q_target:
            print('Target accuracy Qtgt_sq reached.')
            break
        elif (j > 1):
            if (eps_LOO[j] > eps_LOO[j-1]) & (eps_LOO[j-1] > eps_LOO[j-2]):
                print('Overfitting: Stopped at poylnomial order '+ str(j))
                J = j - 2
                break
        elif j == pmax - 1:
            print('All terms included in expansion: pmax ('+str(p)+') polynomial order used')

    # compute optimal PCE expansion
    Psi_opt   = Psi[J]
    alpha_opt = alpha[J]
    a         = np.linalg.solve(Psi_opt.T.dot(Psi_opt) , Psi_opt.T.dot(Y))
    
    def fpce(arg):
        
        N,_ = np.shape(arg)

        norm_opt = np.sqrt(np.prod(sss.factorial(alpha_opt),axis=1))
        out = (np.prod([np.array([np.polyval(He[entry,:],arg[:,dim]) 
              for dim,entry in enumerate(row)]) for row in alpha_opt],axis=1).T/norm_opt).dot(a) 
        
        return out
               
    return fpce,a,alpha_opt


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
    A_p = np.empty(pmax, dtype=object)
    alpha = np.empty(pmax, dtype=object)
    
    eps_LOO_min = np.ones(pmax)*987654321
    I = np.empty(pmax, dtype = int)

    for j in range(pmax):

        p = j+1
        
        # get polynomial base
        He = Hermite_polynomials(p)
            
        # get truncated index set
        alpha[j] = truncated_multi_index(dx, p, q)
        P        = np.shape(alpha[j])[0]
        
        # compute data matrix
        norm   = np.sqrt(np.prod(sss.factorial(alpha[j]),axis=1))
        Psi[j] = np.prod([np.array([np.polyval(He[entry,:],X[:,dim]) 
                 for dim,entry in enumerate(row)]) for row in alpha[j]],axis=1).T/norm
        
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
                print('Overfitting: Forward loop stopped at poylnomial order '+ str(j))
                break
        elif j == nA - 1:
            print('All terms ('+str(i)+') included in expansion: Forward loop stopped')
    
    #err_loo_opt = np.min(np.abs(eps_LOO_min))
    J           = np.argmin(np.abs(eps_LOO_min)).astype(int)

    # compute optimal PCE expansion
    A_opt     = A_p[J][:I[J]]
    Psi_opt   = Psi[J]
    alpha_opt = alpha[J]
    a         = np.linalg.solve(Psi_opt[:,A_opt].T.dot(Psi_opt[:,A_opt]) , Psi_opt[:,A_opt].T.dot(Y))
    
    def spce(arg):
        
        N,_ = np.shape(arg)

        norm_opt = np.sqrt(np.prod(sss.factorial(alpha_opt[A_opt,:]),axis=1))
        out = (np.prod([np.array([np.polyval(He[entry,:],arg[:,dim]) 
              for dim,entry in enumerate(row)]) for row in alpha_opt[A_opt,:]],axis=1).T/norm_opt).dot(a) 
        
        return out
               
    return spce,a,alpha_opt[A_opt,:]

# ---------------------------------------------------------------------------
# main: construct CP with Hermitian polyonmial base and ALS
#
# Konakli, K. and B. Sudret (2016b). 
# Polynomial meta-models with canonical low-rank approximations:
# Numerical insights and comparison to sparse polynomial chaos expansions. 
# Journal of Computational Physics 321, 1144 - 1169.
# ---------------------------------------------------------------------------

#def cp_main(X, Y, R, p):
def cp_main(X,Y,R,p,z0,w0,Y0):
    
    p = p.astype(int)

    N,dx = np.shape(X)
        
    He = Hermite_polynomials(max(p))

    # Alternating least squares

    # loop parameters
    err_tol = 1e-6
    Imax = 50 * dx

    # check for initialization
    if len(z0) > 0 & len(w0) > 0 & len(Y0) > 0:
        _,r0 = np.shape(z0)
        z    = np.concatenate((z0, np.empty((dx, R - r0), dtype=object)),axis=1)
        w    = np.concatenate((w0, np.empty((N,R - r0))),axis=1)
    else:
        Y0 = 0   
        r0 = 0
        z   = np.empty((dx, R), dtype=object)
        w   = np.empty((N,R))
    
    Psi = np.empty(dx, dtype=object)

    # Initialization
    v   = np.ones((N, dx, R))

    
    # Residual
    Yres = Y - Y0
    
    # precompute Psi matrix from poylnomial orders and X-components
    for i in range(dx):
        Psi[i] = np.array([np.polyval(He[j,:],X[:,i].T) for j in range(p[i]+1)]).T
        
    # rank-loop
    for r in range(r0,R):

        err0 = 1e4
        I = 0
        
        # component-loop
        while True:

            # build index i that runs in between 1:d during the loop
            i = np.mod(I, dx)
            if (i == 0 & I>0):
                i = dx

            # fix all components but i-th
            c = np.reshape(np.prod(v[:, np.setdiff1d(range(dx), i), r], axis=1)[:],(-1,1))
                   
            # find polynomial coefficients z
            CPsi   = c*Psi[i]
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
                
        Yhat = np.dot(w[:,:r+1], b)
        # compute new residual
        Yres = Y - Yhat
    
#    def build_lra(z,b):
        
#        dx, R = np.shape(z)
        
    def lra(arg):
        #out = np.array([np.prod([np.polyval(np.ones((1,p[id]+1))
        #      .dot(He[:,np.max(p)-p[id]:]*z[id,ir]).flatten(),X[:,id]) for id in range(dx)]
        #      ,axis=0) for ir in range(R)]).T.dot(np.expand_dims(b, axis=1))
        
        N,_ = np.shape(arg)
        out = np.zeros((N,1))
        
        for ir in range(R):
            
            w_ir = np.ones((N,1))
            
            for id in range(dx):
                
                v_id = np.zeros((N,1))
                
                for ip in range(p[id]+1):
                    
                    v_id = v_id + np.expand_dims(z[id,ir][ip]*np.polyval(He[ip,:],arg[:,id]),axis=1)

                w_ir = w_ir*v_id
            
            out = out + b[ir]*w_ir
        
        return(out)
            
    #return (lra,z,b)
    return (lra,z,b,w,Yhat)

# ---------------------------------------------------------------------------
# - compute n-fold cross validation error for CP format
# - run tensor format adaptation strategy based on cvn score
# ---------------------------------------------------------------------------

def cvn(X,Y,R,p,n,adapt):
    
    cvn_p = []
    R_opt = []

    N,dx = np.shape(X)
        
    # find n equally-sized disjunct subsets of X and Y
    edges = np.round(np.linspace(0, N, num=n+1)).astype(int)
    
    for i in range(n):
        idval  = range(edges[i],edges[i+1])
        idpart = list(set(range(N)) - set(idval))
        
        Xpart  = X[idpart,:]
        Ypart  = Y[idpart]
        Xval   = X[idval,:]
        Yval   = Y[idval]
            
    for k in range(max(p)):

        cvn_r = [] 
        
        z0 = n*[[]]
        w0 = n*[[]]
        Y0 = n*[[]]
                                
        for j in range(R):    
                 
            cvn = 0
            
            for i in range(n):
        
                p_equi = np.ones(np.shape(p))*(k+1)
                #p_true = p_equi + (p - p_equi)*(p < p_equi).flatten()
                
                ftmp,z0[i],_,w0[i],Y0[i] = cp_main(Xpart, Ypart, j+1, p_equi,z0[i],w0[i],Y0[i])                  
                Yhat = ftmp(Xval)
                cvn  = cvn + np.mean((np.expand_dims(Yval, axis=1) - Yhat)**2,axis=0)/np.var(Yval)/n
        
            cvn_r.append(cvn)
            
            print("Degree {0} - Rank {1}: CV score {2}\n".format(k+1,j+1,cvn_r[j]))
            
            # rank selection
            if  adapt in {'R','pR'}:
                # rank overfitting test
                if j > 2:
                    if cvn_r[j] >= cvn_r[j-1] and cvn_r[j-1] >= cvn_r[j-2]:
                        break
                    else:
                        continue
                else:
                    continue
                        
        cvn_p.append(min(cvn_r))
        R_opt.append(np.argmin(np.array(cvn_r))+1)
        
        
        if (cvn_p[k] < cvn_p[:k]).all():
            print("Selected Rank {0} & Degree {1} with CV score {2}\n".format(R_opt[k],k+1,cvn_p[k]))
        
        # polynomial order selection
        if  adapt in {'p','pR'}:

            # polynomial order overfitting test
            if k > 2:
                if cvn_p[k] >= cvn_p[k-1] and cvn_p[k-1] >= cvn_p[k-2]:
                    break
                else:
                    continue
            else:
                continue
        
    
    # cv3 score and optimal order of k-th component
    cvn_opt = min(cvn_p)
    p_opt   = np.argmin(np.array(cvn_p)) + 1

    # optimal rank of k-th component
    r_opt = R_opt[p_opt - 1];
    
    return (cvn_opt,r_opt,p_opt)
        
        
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
    indicator_set = (alpha[:,:] != 0).astype(int)
    total_variance = np.sum(coeffs ** 2)

    id_var=list(range(1,d+1))
    
    # first-order total-effect indices
    S = (np.dot(indicator_set.T,coeffs ** 2)/total_variance).flatten().tolist()
    Sout = [id_var,S]
    
    # first-order Sobol indices
    id_1 = np.where(np.sum(indicator_set,axis=1)==1)
    coeffs = coeffs[id_1]
    indicator_set = indicator_set[id_1]
    V = (np.dot(indicator_set.T,coeffs ** 2)/total_variance).flatten().tolist()
    Vout = [id_var,V]

    #unique, unique_indices, inverse_indices = np.unique(indicator_set, axis=0, return_index=True, return_inverse=True)

    # Sobol indices    
    #id_V=np.array([np.where(indicator_set[unique_indices[i],:]==1) for i in range(len(unique_indices))])+1
    #V = np.bincount(unique_indices[inverse_indices], weights=coeffs.flatten()**2)/total_variance
    #V = V[V>0]
    #Vout = np.concatenate(id_V, np.expand_dims(V, axis=1))
 

    
    
    
    return (Vout,Sout)
