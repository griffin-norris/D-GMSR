import numpy as np
import cvxpy as cp
import typing as T

def rl_cons_fcn(X : np.ndarray,
                 U : np.ndarray,
                 X_last : np.ndarray,
                 U_last : np.ndarray,
                 params : T.Dict[str, T.Any],
                 npy : bool,
                 ) -> T.Tuple[T.Any]:
    
    vehicle_cons = []
    
    vehicle_cons += [X[:, 0] == params['x_init']]
    vehicle_cons += [X[0, -1] >= params['m_f']]
    vehicle_cons += [X[1:-1, -1] == params['x_final'][1:-1]] # Except mass and the gimbal angle rate
    
    vehicle_cons += [X[1, 1:-1] >= params['min_alt']]
    vehicle_cons += [cp.abs(X[7, 1:]) <= params['delta_max']]
    
    vehicle_cons += [U[0, :] <= params['T_max']]
    vehicle_cons += [U[0, :] >= params['T_min']]

    vehicle_cons += [U[1:3, :] <= params['T_max']]
    vehicle_cons += [U[1:3, :] >= 0.]
    
    # ------------------------------------------
    
    vehicle_cons += [cp.abs(U[3, :]) <= params['delta_dot_max']]
    
    return vehicle_cons