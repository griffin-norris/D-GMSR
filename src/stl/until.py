import numpy as np
import typing as T

from .gmsr import gmsr_and, gmsr_or, verify_gmsr_fcn

def UNTIL(eps : float, 
          p : int, 
          w_f : np.ndarray, 
          w_g : np.ndarray,
          w_fg : np.ndarray, 
          f : np.ndarray,
          g : np.ndarray) -> T.Tuple[T.Any]:
    
    s = []
    ds_dg = []
    ds_df = []
    
    K = f.shape[0]
    
    for i in range(K):
        y_i, dyi_dfi = gmsr_and(eps, p, w_f[0:i+1], f[0:i+1])
        s_i, dsi_dyi__dsi_dgi = gmsr_and(eps, p, w_fg[0:2], np.array([y_i, g[i]]))

        dsi_dyi = dsi_dyi__dsi_dgi[0]
        dsi_dgi = dsi_dyi__dsi_dgi[1]
        dsi_dfi = dsi_dyi * dyi_dfi

        s.append(s_i)
        ds_dg.append(dsi_dgi)
        ds_df.append(dsi_dfi)
    
    z, dz_ds = gmsr_or(eps, p, w_g, np.array(s))
    
    dz_df = np.zeros(K)
    for i, dsi_dfi in enumerate(ds_df):
        dz_df[:i+1] += dz_ds[i] * dsi_dfi
    
    dz_dg = dz_ds * np.array(ds_dg)

    return z, dz_df, dz_dg

def verify_UNTIL_grads(ITE : int):

    for i in range(ITE):
        K = np.random.randint(10, 20)
        p = np.random.randint(1, 5)
        eps = 0.1 * np.random.rand()

        w_phi_1 = np.random.randint(1, 5, size=K)
        w_phi_2 = np.random.randint(1, 5, size=K)
        w_phi_12  = np.random.randint(1, 5, size=2)

        pos_neg_1 = [0., 0., -0.5, -0.5]
        pos_neg_2 = [0., -0.5, 0., -0.5]
        
        for case in range(4):

            phi_1 = np.random.rand(K) + pos_neg_1[case]
            phi_2 = np.random.rand(K) + pos_neg_2[case]
            phi_all = np.concatenate((phi_1, phi_2))

            eps_abs = 1e-8
            eps_rel = 1e-6
            del_x = np.maximum(eps_abs,eps_rel*np.abs(phi_all))  

            # Analitic Gradients
            phi_12_k, phi_1_grad, phi_2_grad = UNTIL(eps, p, w_phi_1, w_phi_2, w_phi_12, phi_1, phi_2)

            # Numerical Gradients
            grad_all = np.zeros(2*K)
            for idx in range(2*K):

                x_back = phi_all.copy()
                x_next = phi_all.copy()

                x_back[idx] -= del_x[idx]
                x_next[idx] += del_x[idx]
            
                phi_12_k1, _, _ = UNTIL(eps, p, w_phi_1, w_phi_2, w_phi_12, x_back[:K], x_back[K:])
                phi_12_k2, _, _ = UNTIL(eps, p, w_phi_1, w_phi_2, w_phi_12, x_next[:K], x_next[K:])

                grad_all[idx] = (phi_12_k2 - phi_12_k1) / (2*del_x[idx])

            err_1 = np.linalg.norm(phi_1_grad-grad_all[:K])
            err_2 = np.linalg.norm(phi_2_grad-grad_all[K:])

            if err_1 > 1e-7 or err_2 > 1e-7:
                print('Analytic gradient calculation is wrong!')
                break

    print('UNTIL - Successful')