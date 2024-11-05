import numpy as np
import scipy.stats as st
import torch
import matplotlib.pyplot as plt

relu = torch.nn.functional.relu

class ModifiedRipleysK:
    """Calculate the modified Ripley's K as in 
    
    - [1] Lagache, T., Lang, G., Sauvonnet, N., & Olivo-Marin, J.-C. (2013). Analysis of the Spatial
    Organization of Molecules with Robust Statistics. PLoS ONE, 8(12), e80914.
    https://doi.org/10.1371/journal.pone.0080914
    - [2] Amgad, M., Itoh, A., & Tsui, M. M. K. (2015). Extending Ripley’s K-Function to Quantify
    Aggregation in 2-D Grayscale Images. PLOS ONE, 10(12), e0144404.
    https://doi.org/10.1371/journal.pone.0144404
    - [3] Article [1] Supplementary Material S1 https://doi.org/10.1371/journal.pone.0144404.s001
    - [4] Article [1] Supplementary Material S2 https://doi.org/10.1371/journal.pone.0144404.s002
    """
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.area = abs(x1 - x2) * abs(y1 - y2)
        self.peri = 2*abs(x1 - x2) + 2*abs(y1 - y2)

    def __call__(self, z1, z2, r, log=False, plot=False):
        n = len(z2)
        # preliminaries
        # compute the distances between points in z1 and z2
        d = z1.unsqueeze(0) - z2.unsqueeze(0).permute((1, 0, 2))
        d = d.pow(2).sum(dim=-1).sqrt()
        # compute beta
        beta = np.pi * r.pow(2) / self.area
        # compute gamma
        gamma = self.peri * r / self.area
        # calculate K [1, eq (1)]
        # calculate which other points are within each radius of each point
        indicator = d.unsqueeze(-1) <= r.unsqueeze(0).unsqueeze(0)
        # calculate the circle area for every radius
        area_circle = np.pi * r.pow(2).unsqueeze(0).unsqueeze(0)
        # calculate the intersection area of the circle with the square for
        # every point and every radius
        # find distance from corners (as designed, these will always be positive)
        d_l = z1[:, 0] - self.x1
        d_r = self.x2 - z1[:, 0]
        d_b = z1[:, 1] - self.y1
        d_t = self.y2 - z1[:, 1]
        # find the heights of the circular sector on each side
        R = r.unsqueeze(0)
        h_l = relu(R - d_l.unsqueeze(-1))
        h_r = relu(R - d_r.unsqueeze(-1))
        h_b = relu(R - d_b.unsqueeze(-1))
        h_t = relu(R - d_t.unsqueeze(-1))
        # now find the areas
        A_l = R.pow(2) * torch.arccos((R - h_l) / (R)) \
            - (R - h_l) * torch.sqrt(2 * R * h_l - h_l.pow(2))
        A_r = R.pow(2) * torch.arccos((R - h_r) / (R)) \
            - (R - h_r) * torch.sqrt(2 * R * h_r - h_r.pow(2))
        A_b = R.pow(2) * torch.arccos((R - h_b) / (R)) \
            - (R - h_b) * torch.sqrt(2 * R * h_b - h_b.pow(2))
        A_t = R.pow(2) * torch.arccos((R - h_t) / (R)) \
            - (R - h_t) * torch.sqrt(2 * R * h_t - h_t.pow(2))
        # correct for corners (right now this is just a triangular approximation)
        c_lt = h_l * h_t / 2.0
        c_rt = h_r * h_t / 2.0
        c_lb = h_l * h_b / 2.0
        c_rb = h_r * h_b / 2.0
        area_circle_in_square = area_circle \
            - A_l.unsqueeze(1) - A_r.unsqueeze(1) - A_b.unsqueeze(1) - A_t.unsqueeze(1) \
            + c_lt.unsqueeze(1) + c_rt.unsqueeze(1) + c_lb.unsqueeze(1) + c_rb.unsqueeze(1)
        if log:
            print(f'{z1}')
            print(f'{d_l}')
            print(f'{h_l}')
            print(f'{A_l}')
            print(area_circle.shape, area_circle_in_square.shape)
            print(area_circle)
            print(area_circle_in_square)
        # calculate the correction from ripley [2, method I]
        f = (area_circle / area_circle_in_square) * n
        # finally calculate the whole matrix K_rn
        K_rn = (self.area / (n * (n - 1))) * (f.sum(1, keepdim=True) * indicator).sum(1)
        # calculate the variance of K_rn
        var_hat_K_rn = K_rn.var(dim=0)
        if log:
            print(f'{f.shape}')
            print(f'{indicator.shape}')
            print(f'{K_rn.shape}')
            print(f'{var_hat_K_rn.shape}')
        # calculate the modified ripley's function, K_tilde_rn [2, eq (4)]
        K_tilde_rn = (K_rn - np.pi * R.pow(2)) / var_hat_K_rn.sqrt()
        if plot:
            plt.plot(r, K_rn.mean(0), label='K_rn')
            plt.plot(r, np.pi * r.pow(2), label='area')
            plt.plot(r, var_hat_K_rn.sqrt().squeeze(), label='var_K_rn')
            #plt.plot(r, K_tilde_rn.mean(0), label='K_tilde_rn')
            plt.legend()
            plt.show()
            print(f'{var_hat_K_rn.shape}')
        # average over all the points
        K_tilde_m = K_tilde_rn.mean(0)
        # calculate CRs under CSR in closed form
        M = K_rn.shape[0]
        # calculate the variance of K_rn
        var_K_rn = ((2 * self.area * self.area * beta) / (n * n)) \
            * (1 + 0.305 * gamma + beta * (-1 + 0.0132 * n * gamma)).unsqueeze(0)
        # [2, eq (6)]
        a = self.area
        skewness = ((4 * np.power(a, 3) * beta) / (np.power(n, 4) * var_K_rn.pow(3/2))) \
            * (1 + 0.76 * gamma + n * beta * (1.173 + 0.414 * gamma) 
               + n * beta * beta * (-2 + 0.012 * n * gamma))
        # [3, eq (172)]
        skewness_tilde = skewness.mean(0) / np.sqrt(M)
        # [2, eq (7)]
        kurtosis = ((np.power(a, 4) * beta) / (np.power(n, 6) * var_K_rn.pow(2))) \
            * (8 + 11.52 * gamma + n * beta * ((104.3 + 12 * n) + (78.7 + 7.32 * n) * gamma
                                               + 1.116 * n * gamma.pow(2))
               + n * beta * beta * ((-304.3 - 1.92 * n)
                                    + (-97.9 + 2.69 * n + 0.317 * n * n) * gamma)
               + 0.0966 * n * n * gamma.pow(2)
               + n * n * beta.pow(3) * (-36 + 0.0021 * n * n * gamma.pow(2)))
        # [3, eq (177)]
        kurtosis_tilde = (kurtosis.mean(0) / M) + (3 * (M - 1) / M)
        # quantiles under CSR
        z_1 = st.norm.ppf(0.025)
        q_1 = z_1 + (1 / 6) * (z_1 * z_1 - 1) * skewness_tilde \
            + (1 / 24) * (np.power(z_1, 3) - 3 * z_1) * (kurtosis_tilde - 3) \
            - (1 / 36) * (2 * np.power(z_1, 3) - 5 * z_1) * skewness_tilde.pow(2)
        z_99 = st.norm.ppf(0.975)
        q_99 = z_99 + (1 / 6) * (z_99 * z_99 - 1) * skewness_tilde \
            + (1 / 24) * (np.power(z_99, 3) - 3 * z_99) * (kurtosis_tilde - 3) \
            - (1 / 36) * (2 * np.power(z_99, 3) - 5 * z_99) * skewness_tilde.pow(2)
        if log:
            print(f'{gamma}')
            print(f'{beta}')
            print(f'{kurtosis}')
            print(f'{skewness}')
            print(f'{q_1}')
            print(f'{q_99}')
        return K_tilde_m, q_1, q_99