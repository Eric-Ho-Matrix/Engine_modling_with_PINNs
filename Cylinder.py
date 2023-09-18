import torch

class Cylinder():
        def __init__(self, c_volVec, V_d, R_a, T_im, n_cyl, q_HV, x_cv, c_pa, c_va, r_c, gamma_a, eta_sc, T_amb,
                     h_tot, d_pipe, l_pipe, n_pipe, c_pe):
                self.c_vol_1 = c_volVec[0]
                self.c_vol_2 = c_volVec[1]
                self.c_vol_3 = c_volVec[2]
                self.V_d = V_d
                self.R_a = R_a
                self.T_im = T_im
                self.n_cyl = n_cyl
                self.q_HV = q_HV
                self.x_cv = x_cv
                self.c_pa = c_pa
                self.c_va = c_va
                self.r_c = r_c
                self.gamma_a = gamma_a
                self.eta_sc = eta_sc
                self.T_amb = T_amb
                self.h_tot = h_tot
                self.d_pipe = d_pipe
                self.l_pipe = l_pipe
                self.n_pipe = n_pipe
                self.c_pe = c_pe

        def Cylinder_flow(self, p_im, n_e, u_delta):  # check
                # Cylinder flow
                eta_vol = self.c_vol_1 * torch.sqrt(p_im) + self.c_vol_2 * torch.sqrt(n_e) + self.c_vol_3
                W_ei = (eta_vol * p_im * n_e * self.V_d) / (120 * self.R_a * self.T_im)                   # check
                W_f = (1e-6 * u_delta * n_e * self.n_cyl) / 120
                W_eo = W_ei + W_f
                return W_f, W_eo, W_ei


        def Cylinder_temperature(self, p_im, p_em, x_r, T_1, W_eo, W_ei, W_f):  # check
                # Cylinder temperature
                pi_e = p_em / (p_im+1e-9)
                q_in = (W_f * self.q_HV) * (1 - x_r) / (W_ei + W_f)
                x_p = 1 + (q_in * self.x_cv / (self.c_va*T_1*self.r_c**(self.gamma_a - 1)))
                x_v = 1 + (q_in * (1 - self.x_cv) / (self.c_pa * ((q_in * self.x_cv)/(self.c_va) + T_1 * self.r_c**(self.gamma_a-1))))

                T_e = self.eta_sc * pi_e**(1-1/self.gamma_a) * self.r_c**(1-self.gamma_a) * x_p**(1/self.gamma_a -1) \
                    * (q_in * ((1-self.x_cv)/(self.c_pa) + (self.x_cv)/(self.c_va)) + T_1*self.r_c**(self.gamma_a -1))            # check

                T_em = self.T_amb + (T_e - self.T_amb)*torch.exp(-(self.h_tot*torch.pi*self.d_pipe*self.l_pipe*self.n_pipe)/(W_eo*self.c_pe))   # check

                return T_em, x_v, T_e, x_p


                

                


