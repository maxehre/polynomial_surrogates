import numpy as np
import scipy.stats as ss

'''
#===========================================================================
1D Bernoulli beam
#===========================================================================
Inputs:
X = [Load P, Beam Length L, Young's Modulus E, Beam Width B, Beam Height H]
all log-normally distributed

Output:    
Beam tip deflection U = PL^3/(4EBH^3)
log-normally distributed
#===========================================================================
'''

# Input variable means and std. dev. 
muX_beam  = np.array([1e4, 5, 3e10, .15, .3])
stdX_beam = np.array([2e3, .05, .45e10, .0075, .015])

# lognormal parameters (mean and std.dev. of underlying Gaussian)
stdlnX_beam = np.sqrt(np.log(stdX_beam**2./muX_beam**2 +1))
mulnX_beam  = np.log(muX_beam) - .5*stdlnX_beam**2

                    
# ---------------------------------------------------------------------------                    
# tandard-normal space v <=> original space x
# ---------------------------------------------------------------------------

def X2V_beam(x):
    return (np.log(x) - mulnX_beam)/stdlnX_beam

def V2X_beam(v):
    return np.exp(mulnX_beam + stdlnX_beam*v)


# ---------------------------------------------------------------------------
# beam model 
# ---------------------------------------------------------------------------

def beam(x):
    return   x[0]*x[1]**3/(4*x[2]*x[3]*x[4]**3)

'''
#===========================================================================
Truss 2D example
#===========================================================================
Inputs:
X = [E1 ,E2, A1, A2, P1, P2, P3, P4, P5, P6]

Youngs Moduli E1, E2:  log-normally distributed
Cross-Sections A1, A2: log-normally distributed
Loads P1 -P6:          Gumbel distributed

Output:    
vertical truss deflection at bottom center (N04)
#===========================================================================
truss description taken from
Lee, S.H. and B.M Kwak (2006).
Response surface augmented moment method for efficient reliability analysis.
Structural Safety 28(3), 261 - 272.
                       
example code based on
elementosfinitosunalmzl.wikispaces.com
#===========================================================================

       N13__R4___N12__R8___N11__R12__N10__R16__N09__R20__N09  
        /\        /\        /\        /\        /\        /\
       /  \      /  \      /  \      /  \      /  \      /  \
      R1  R3    R5   R7   R9  R11   R13 R15  R17  R19  R21  R23
     /      \  /      \  /      \  /      \  /      \  /      \
    /___R2___\/___R6___\/__R10___\/__R14___\/__R18___\/__R22___\
  N01        N02      N03       N04        N05       N06       N07
                                 
                                 ||
                                 \/
                                uout = u_y @ N04
                                
#===========================================================================
'''

# Input variable parameters 

muX_truss  = np.array([2.1e11, 2.1e11, 2e-3, 1e-3, 5e4, 5e4, 5e4, 5e4, 5e4, 5e4])
stdX_truss = np.array([2.1e10, 2.1e10, 2e-4, 1e-4, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3, 7.5e3])

# lognormal parameters (mean and std.dev. of underlying Gaussian)
stdlnX_truss = np.sqrt(np.log(stdX_truss[:4]**2./muX_truss[:4]**2 +1))
mulnX_truss  = np.log(muX_truss[:4]) - .5*stdlnX_truss**2

# Gumbel parameters
eulergamma = 0.5772157
an         = stdX_truss[4:]*np.sqrt(6)/np.pi
bn         = muX_truss[4:] - eulergamma*an

                      
# ---------------------------------------------------------------------------
# Transformation: Standard-Normal v <=> Original Space x
# ---------------------------------------------------------------------------

def X2V_truss(x):
    v     = np.empty(np.shape(x)) 
    v[:4] = (np.log(x[:4]) - mulnX_truss)/stdlnX_truss
    v[4:] = ss.norm.ppf(np.exp(-np.exp((x[4:] - bn)/an)), loc=0, scale=1)
    
    return v
 
def V2X_truss(v):
     x     = np.empty(np.shape(v))
     x[:4] = np.exp(mulnX_truss + stdlnX_truss*v[:4])
     x[4:] = bn - an*np.log(-np.log(ss.norm.cdf(v[4:], loc=0, scale=1)))
     
     return x


# ---------------------------------------------------------------------------
# 2d truss model
# ---------------------------------------------------------------------------

def truss(X):
    
    # vector inputs            
    E1 = X[0]
    E2 = X[1]
    A1 = X[2]
    A2 = X[3]
    P  = X[4:]
        
    # element, nodes and dofs association
    # IEN: connectivity matrix, nfe x nodes
    IEN = np.array(
           [[1 , 13],   # bar 1 has nodes 1 and 3
           [1 , 2 ],    # bar 2 has nodes 1 and 4 ...
           [13, 2 ],
           [13, 12], 
           [2 , 12], 
           [2 , 3 ],
           [12, 3 ],
           [12, 11],
           [3 , 11],
           [3 , 4 ],
           [11, 4 ],
           [11, 10],
           [4 , 10],
           [4 , 5 ],
           [10, 5 ],
           [10, 9 ],
           [5 , 9 ],
           [5 , 6 ],
           [9 , 6 ],
           [9 , 8 ],
           [6 , 8 ],
           [6 , 7 ],
           [8 , 7 ]]) - 1
    
    # deterministic rod properties
    ang       = np.arctan2(200,200)              # inclination angle of the truss [deg]
    theta     = np.array(6*[ang, 0, -ang, 0])    # inclination angle [deg]
    leng      = np.array(12*[4/np.sqrt(2), 4])   # bar length [cm]
    theta     = np.delete(theta,-1)
    leng      = np.delete(leng,-1)
    
    # FEM constants
    ned  = 2                  # number of dof per node
    nnp  = 13                 # number of nodal points
    nfe  = np.shape(IEN)[0]   # number of bars
    ndof = ned*nnp            # number of degrees of freedom (dof)
    
    # dof: degrees of freedom (rows are the nodes, cols are dofs)
    dof = np.array(range(ndof)).reshape((nnp,ned))
          
    # stochastic rod properties
    area     = np.array(12*[A2, A1]) # bar cross sectional area [cm2]
    E        = np.array(12*[E2, E1]) # young's modulus [ton/cm^2]
    area     = np.delete(area,-1)
    E        = np.delete(E,-1)
    
    # material properties
    k = E*area/leng  # stiffness of each bar
    
    # boundary conditions (supports)
    cc = np.array([1, 2, 14])-1                      # dof fixed/supports
    dd = np.array(list(set(range(ndof)) - set(cc)))  # dof free
    
    # boundary conditions (applied force) 
    fc        = np.zeros(ndof)
    fc[15::2] = -P
    fc        = np.delete(fc,cc)
    fc        = np.expand_dims(fc, axis=1)
    
    # global stiffness matrix
    K = np.zeros((ndof,ndof))
    T = nfe*[None]        # MATLAB = cell(5,1) -> memory alloc
    for e in range(nfe):
        idx  = np.r_[dof[IEN[e,0],:], dof[IEN[e,1],:]]   # extract the dofs of the element
        c    = np.cos(theta[e])    # cosine inclination angle
        s    = np.sin(theta[e])    # sinus inclination angle
        T[e] = np.array([[ c,  s,  0,  0],         # coordinate transformation matrix
                         [-s,  c,  0,  0],
                         [ 0,  0,  c,  s], 
                         [ 0,  0, -s,  c]])
        Kloc = k[e]*np.array([[ 1,  0, -1,  0],    # local stiffness matrix 
                              [ 0,  0,  0,  0],                 
                              [-1,  0,  1,  0],
                              [ 0,  0,  0,  0]])
        K[np.ix_(idx,idx)] += T[e].T.dot(Kloc).dot(T[e])     # add to K global
    
    # solve systems of equations
    # f = vector of equivalent nodal forces
    # q = vector of equilibrium nodal forces 
    # a = displacements
    #| qd |   | Kcc Kcd || ac |   | fd |
    #|    | = |         ||    | - |    |
    #| qc |   | Kdc Kdd || ad |   | fc |
    
    Kcc = K[cc,:][:,cc]
    Kcd = K[cc,:][:,dd]
    Kdc = K[dd,:][:,cc]
    Kdd = K[dd,:][:,dd]
    
    ac = np.array([[0], [0], [0]])                # displacements for fixed/support nodes
    ad = np.linalg.solve(Kdd, fc - Kdc.dot(ac))   # solution
    qd = Kcc.dot(ac) + Kcd.dot(ad)
    
    # assemble vectors of displacements (a) and forces (q)
    a = np.zeros((ndof,1)) 
    q = np.zeros((ndof,1))
    a[cc] = ac    
    q[cc] = qd
    a[dd] = ad    # q[dd] = qc = 0
    
    # compute axial loads
    N = np.zeros(nfe)
    for e in range(nfe):
        idx  = np.r_[dof[IEN[e,0],:], dof[IEN[e,1],:]] 
        N[e] = k[e]*np.array([-1, 0, 1, 0]).dot(T[e]).dot(a[idx])
    
    # output vertical deflection
    uout = a[7]
    
    return uout