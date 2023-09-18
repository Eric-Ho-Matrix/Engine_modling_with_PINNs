import torch


class Compressor():
    def __init__(self, p_amb, gamma_a, c_pa, T_amb, R_c, c_psi_2, c_phi_2, R_a, w_copt,
                 c_pi, pi_copt, Q_c, eta_cmax):
        self.p_amb = p_amb
        self.gamma_a = gamma_a
        self.c_pa = c_pa
        self.T_amb = T_amb
        self.R_c = R_c
        self.c_psi_2 = c_psi_2
        self.c_phi_2 = c_phi_2
        self.R_a = R_a
        self.w_copt = w_copt
        self.c_pi = c_pi
        self.pi_copt = pi_copt
        self.Q_c = Q_c
        self.eta_cmax = eta_cmax

    def phi_c_fun(self, c_psi_1, c_phi_1, psi_c):
        ratio = (1-c_psi_1*(psi_c-self.c_psi_2)**2)/c_phi_1
        condition = ratio<0
        first_term = torch.where(condition, 0, ratio)
        return torch.sqrt(first_term)+self.c_phi_2

    def comp_mass_flow(self, p_im, omega_t, c_omega_psi_1, c_omega_psi_2, c_omega_psi_3,
                       c_omega_phi_1, c_omega_phi_2, c_omega_phi_3):
        pi_c = p_im / (self.p_amb+1e-9)         # check
        # print(pi_c)
        psi_c = 2*self.c_pa*self.T_amb*(pi_c**(1-1/self.gamma_a) -1)/(self.R_c**2 * omega_t**2)  # check
        # print(psi_c)
        c_psi_1 = c_omega_psi_1*omega_t**2 + c_omega_psi_2*omega_t + c_omega_psi_3               # check
        # print(c_psi_1)
        c_phi_1 = c_omega_phi_1*omega_t**2 + c_omega_phi_2*omega_t + c_omega_phi_3               # check
        # print(c_phi_1)
        phi_c = self.phi_c_fun(c_psi_1, c_phi_1, psi_c)                                          # check
        # print(phi_c)
        W_c = self.p_amb*torch.pi*self.R_c**3*omega_t*phi_c / (self.R_a * self.T_amb)            # check
        # print(W_c)
        
        return W_c
    
    def comp_mass_flow_pre_tra_nn(self, omega_t, phi_c):                    
        W_c = self.p_amb*torch.pi*self.R_c**3*omega_t*phi_c / (self.R_a * self.T_amb)            # check
        
        return W_c         

    def comp_effi(self, W_c, p_im):     # check
        pi_c = p_im / (self.p_amb+1e-9) 
        small_pi_c = (pi_c - 1)**(self.c_pi)
        
        # make dimension right
        ele1 = W_c - self.w_copt      # dim ?x1  "?" represents sample number
        ele2 = small_pi_c - self.pi_copt
        temp = torch.concat((ele1, ele2), axis=1)
        X = torch.reshape(temp, (W_c.shape[0], 2, 1))
        X_trans = torch.reshape(temp, (W_c.shape[0], 1, 2)) # dim ?x1x2

        Q_c = self.Q_c.repeat(W_c.shape[0],1,1)  # dim ?x2x2

        step1 = torch.einsum('bij,bjk->bik', X_trans, Q_c)
        step2 = torch.einsum('bij,bjk->bik', step1, X)

        eta_c = torch.reshape((self.eta_cmax - step2), (W_c.shape[0],-1))   # should have dim ?x1

        p_c = W_c * self.c_pa*self.T_amb*(pi_c**(1-1/self.gamma_a) -1) / eta_c

        return p_c


