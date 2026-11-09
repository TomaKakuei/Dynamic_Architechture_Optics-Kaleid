"""Small convex local subproblem; constraints apply to the actual update."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def solve_local(G, gradient, B, x, prior, center_gram, radius=.2, center_limit=.01, step_limit=.05):
    d=x-prior; scale=step_limit; A=scale*B
    lo=np.maximum.reduce([np.full(35,-step_limit),-radius-d,-1-x])
    hi=np.minimum.reduce([np.full(35,step_limit),radius-d,1-x])
    Q=(G+G.T)*.5+1e-8*np.eye(G.shape[0])
    objective_scale=max(float(np.linalg.norm(scale*gradient)),1e-6)
    def fun(z):
        delta=scale*z
        return float((delta@Q@delta+2*gradient@delta)/objective_scale)
    def jac(z):return 2*scale*(Q@(scale*z)+gradient)/objective_scale
    constraints=[LinearConstraint(A,lo,hi)]
    if center_limit:
        constraints.append(dict(type='ineq',fun=lambda z:float(1-(d+A@z)@center_gram@(d+A@z)/center_limit**2),
            jac=lambda z:-2*A.T@center_gram@(d+A@z)/center_limit**2))
    solved=minimize(fun,np.zeros(B.shape[1]),jac=jac,constraints=constraints,method='SLSQP',
                    options=dict(maxiter=150,ftol=1e-10,disp=False))
    dz=scale*solved.x; step=B@dz
    center=float(np.sqrt(max((d+step)@center_gram@(d+step),0)))
    violation=max(float(np.max(lo-step)),float(np.max(step-hi)),0.)
    feasible=violation<=1e-7 and (not center_limit or center<=center_limit*(1+1e-5))
    info=dict(success=bool(solved.success),message=str(solved.message),iterations=int(solved.nit),
        feasible=bool(feasible),max_box_violation=violation,center_change=center,
        predicted_gain=float(-(dz@Q@dz+2*gradient@dz)),step_max=float(abs(step).max()))
    return dz,info
