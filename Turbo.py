import torch

# input: p_im p_em T_em u_vgt
# output: W_c T_c W_t

class Turbo():
    def __init__(self,c_pe, c_f1, c_f2, c_vgt1, c_vgt2, p_amb, K_t, R_e, 
                 c_m, R_t, gamma_e, eta_tmmax, BSR_opt, A_vgtmax):
        self.c_pe  =  c_pe 
        self.c_f1  = c_f1
        self.c_f2  = c_f2
        self.c_vgt1 = c_vgt1
        self.c_vgt2 = c_vgt2
        self.p_amb  = p_amb
        self.K_t   = K_t
        self.R_e = R_e
        self.c_m1 = c_m[0]
        self.c_m2 = c_m[1]
        self.c_m3 = c_m[2]
        self.R_t  = R_t
        self.gamma_e = gamma_e
        self.eta_tmmax = eta_tmmax
        self.BSR_opt = BSR_opt
        self.A_vgtmax = A_vgtmax
    
    def f_vgt(self,u_vgt_bar):
        ratio = 1 - ((u_vgt_bar - self.c_vgt2)/ self.c_vgt1)**2
        condition = ratio <= 0
        result = torch.where(condition, 0, ratio)
        f = self.c_f2 + self.c_f1 * torch.sqrt(result)
        return f
    
    def c_m(self, omega_t):
        minus_term = omega_t - self.c_m2
        condition = minus_term <= 0
        result = torch.where(condition, 0, minus_term)
        return self.c_m1 * result**self.c_m3

    def Turbo_charger_dynamics(self, omega_t, u_vgt_bar, p_em, T_em):   # check
        # W_t calculation 
        pi_t = self.p_amb / (p_em+1e-9)
        f_pi_t = torch.sqrt(1-pi_t**self.K_t)
        W_t = self.A_vgtmax * p_em * f_pi_t * self.f_vgt(u_vgt_bar) / torch.sqrt(T_em * self.R_e)

        BSR = (self.R_t * omega_t) / torch.sqrt(2*self.c_pe*T_em*(1-pi_t**(1-1/self.gamma_e)))
        eta_tm = self.eta_tmmax - self.c_m(omega_t) * (BSR - self.BSR_opt)**2
        Pt_eta_m = eta_tm * W_t * self.c_pe * T_em * (1-pi_t**(1-1/self.gamma_e))

        return Pt_eta_m, W_t


