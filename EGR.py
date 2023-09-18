import torch

class EGR():
        def __init__(self, PI_egropt, c_egr, R_e, K_egr):
                self.PI_egropt = PI_egropt
                self.c_egr1 = c_egr[0]
                self.c_egr2 = c_egr[1]
                self.c_egr3 = c_egr[2]
                self.R_e = R_e
                self.K_egr = K_egr

        
        def pi_egr(self, p_im, p_em):
                ratio = p_im / p_em
                # Check the first condition
                condition1 = ratio < self.PI_egropt
                result = torch.where(condition1, self.PI_egropt, ratio)
                
                # Check the second condition
                condition2 = torch.logical_and(self.PI_egropt <= ratio, ratio <= 1)
                result = torch.where(condition2, ratio, result)
                
                # Check the third condition
                condition3 = ratio > 1
                result = torch.where(condition3, torch.ones_like(ratio), result)
                
                return result     
        
        def f_egr(self,u_egr_bar):
                ratio = -(self.c_egr2)/(2*self.c_egr1)
                condition1 = u_egr_bar <= ratio
                result = torch.where(condition1, self.c_egr1*u_egr_bar**2 + self.c_egr2*u_egr_bar + self.c_egr3, ratio)

                condition2 = u_egr_bar > ratio
                result = torch.where(condition2, self.c_egr3 - (self.c_egr2**2)/(4*self.c_egr1), result)

                return result

                
        def EGR_dynamics(self, p_im, p_em, u_egr_bar1, u_egr_bar2, T_em, A_egrmax):  # check
                # EGR_valve flow
                psi_egr = 1 - ((1-self.pi_egr(p_im, p_em))/(1- self.PI_egropt) -1)**2

                u_egr_bar = self.K_egr * u_egr_bar1 - (self.K_egr-1)*u_egr_bar2

                A_egr = A_egrmax * self.f_egr(u_egr_bar)

                W_egr = (A_egr * p_em * psi_egr)/ torch.sqrt(T_em*self.R_e)

                return W_egr
                


                

                

                


