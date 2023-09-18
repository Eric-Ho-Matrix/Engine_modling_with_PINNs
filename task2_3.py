import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
from scipy.io import loadmat
import matplotlib.pyplot as plt
from EGR import EGR
from Compressor import Compressor
from Cylinder import Cylinder
from Turbo import Turbo
from torch.nn.functional import softplus
from torch import sigmoid
from torch import relu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("current device: ", device, "\n")

torch.autograd.set_detect_anomaly(True)

# define the model use Tanh activation function
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_neurons, output_dim):
        super(PINN, self).__init__()
        
        self.net = nn.Sequential()
        
        # add input layer
        linear1 = nn.Linear(input_dim, num_neurons)
        self.net.add_module("lin_layer_0", linear1)
        # self.net.add_module("Batch_norm_0", nn.BatchNorm1d(num_neurons))
        self.net.add_module("Tanh_layer_0", nn.Tanh())
        
        # add hidden layer
        for num in range(1, hidden_layers + 1):
            linear = nn.Linear(num_neurons, num_neurons)
            self.net.add_module("lin_layer_%d" %(num), linear)
            # self.net.add_module("Batch_norm_%d"%(num), nn.BatchNorm1d(num_neurons))
            self.net.add_module("Tanh_layer_%d" % (num), nn.Tanh())
            
        # add output layer
        linear3 = nn.Linear(num_neurons, output_dim)
        self.net.add_module("lin_layer_output", linear3)

    def time_mapping(self, t_tensor):
        t_min = torch.min(t_tensor)
        t_max = torch.max(t_tensor)
        t_tensor_ret = -1 + (2*(t_tensor - t_min) / (t_max - t_min))
        return t_tensor_ret
        
    def forward(self, x):
        x_norm = self.time_mapping(x)
        return self.net(x_norm)
    

# Define the phyics-informed loss function
def p_im_physics_loss(p_im, t,R_a, T_im, V_im, W_c, W_egr, W_ei):
    d_pim = grad(p_im, t, grad_outputs=torch.ones_like(p_im), retain_graph=True)
    equation = d_pim[0] - R_a*T_im*(W_c + W_egr - W_ei)/V_im 
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def p_em_physics_loss(p_em, t, R_e, T_em, V_em, W_eo, W_egr, W_t):
    d_pem = grad(p_em, t, grad_outputs=torch.ones_like(p_em), retain_graph=True)
    equation = d_pem[0]- R_e*T_em*(W_eo - W_t - W_egr)/V_em 
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def omega_t_physics_loss(omega_t, t, Pt_eta_m, p_c, J_t):
    d_omega_t = grad(omega_t, t, grad_outputs=torch.ones_like(omega_t), retain_graph=True)
    equation = d_omega_t[0] - (Pt_eta_m-p_c)/(J_t*omega_t)
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def u_egr1_physics_loss(u_egr1_bar, t, tau_egr_1, u_egr, tau_degr):
    d_u_egr1_bar = grad(u_egr1_bar, t, grad_outputs=torch.ones_like(u_egr1_bar), retain_graph=True)
    equation = d_u_egr1_bar[0] - (u_egr*(t-tau_degr)-u_egr1_bar)/tau_egr_1
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def u_egr2_physics_loss(u_egr2_bar, t, tau_egr_2, u_egr, tau_degr):
    d_u_egr2_bar = grad(u_egr2_bar, t, grad_outputs=torch.ones_like(u_egr2_bar), retain_graph=True)
    equation = d_u_egr2_bar[0] - (u_egr*(t-tau_degr)-u_egr2_bar)/tau_egr_2
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def u_vgt_bar_physics_loss(u_vgt_bar, t, tau_vgt, u_vgt, tau_dvgt):
    d_u_vgt_bar = grad(u_vgt_bar, t, grad_outputs=torch.ones_like(u_vgt_bar), retain_graph=True)
    equation = d_u_vgt_bar[0] - (u_vgt*(t-tau_dvgt)-u_vgt_bar)/tau_vgt
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def T_1_physics_loss(T_1, x_r, T_e, T_im):
    equation = T_1 - x_r*T_e - (1-x_r)*T_im
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def x_r_physics_loss(x_r, p_im, p_em, gamma_a, x_p, r_c, x_v):
    pi_e = p_em/p_im
    equation = x_r - (pi_e**(1/gamma_a) * x_p**(-1/gamma_a))/(r_c*x_v)
    # residual = torch.mean(torch.sum(torch.square(equation)))
    return equation

def weighted_loss(lambdaa, loss_):
    weighted_loss_ = lambdaa * loss_
    return torch.mean(torch.square(weighted_loss_))

def normalize(x_tensor):
    x_min = torch.min(x_tensor)
    x_max = torch.max(x_tensor)
    x_tensor_ret = (x_tensor - x_min) / (x_max - x_min)
    return x_tensor_ret

################ End of Definition ##################

# Data Extraction
model_parameter = loadmat('WahlstromErikssonTCDI_EGR_VGT/parameterData.mat')['model']
workspace_sim = loadmat('WahlstromErikssonTCDI_EGR_VGT/Workspace.mat')

p_im   = workspace_sim['simp_im']
# T_em   = workspace_sim['simT_em']
p_em   = workspace_sim['simp_em']
W_egr  = workspace_sim['simW_egr']
t      = workspace_sim['simTime']
n_e    = workspace_sim['simn_e']     #input
u_delta= workspace_sim['simu_delta'] #input
u_egr  = workspace_sim['simu_egr']   #
u_vgt  = workspace_sim['simu_vgt']   #
omega_t = workspace_sim['sim_omega_t']
T_1     = workspace_sim['simT_1']
x_r     = workspace_sim['simx_r']
uegr_1_bar = workspace_sim['u_egr_bar_1']
uegr_2_bar = workspace_sim['u_egr_bar_2']
u_vgt_bar = workspace_sim['simu_vgtact']
uInit_egr = model_parameter["uInit_egr"][0][0][0]
PI_egropt = model_parameter["PI_egropt"][0][0][0]
c_egr     = model_parameter["c_egr"][0][0]
A_egrmax_gt  = model_parameter["A_egrmax"][0][0][0]
R_e       = model_parameter["R_e"][0][0][0]
tau_degr  = model_parameter["tau_degr"][0][0][0]
tau_egr1  = model_parameter["tau_egr1"][0][0][0]
tau_egr2  = model_parameter["tau_egr2"][0][0][0]
K_egr     = model_parameter["K_egr"][0][0][0]
R_a       = model_parameter["R_a"][0][0][0]
T_im      = model_parameter["T_im"][0][0][0]
V_im      = model_parameter["V_im"][0][0][0]
c_pe      = model_parameter["c_pe"][0][0][0]
c_f1      = model_parameter["c_f1"][0][0][0]
c_f2      = model_parameter["c_f2"][0][0][0]
c_vgt1    = model_parameter["c_vgt1"][0][0][0]
c_vgt2    = model_parameter["c_vgt2"][0][0][0]
p_amb     = model_parameter["p_amb"][0][0][0]
K_t       = model_parameter["K_t"][0][0][0]
c_m       = model_parameter["c_mVec"][0][0][0]
R_t       = model_parameter["R_t"][0][0][0]
gamma_e   = model_parameter["gamma_e"][0][0][0]
eta_tmmax = model_parameter["eta_tmmax"][0][0][0]
BSR_opt   = model_parameter["BSR_opt"][0][0][0]
c_volVec  = model_parameter["c_volVec"][0][0]        
V_d       = model_parameter["V_d"][0][0][0] 
n_cyl     = model_parameter["n_cyl"][0][0][0]   
q_HV      = model_parameter["q_HV"][0][0][0]  
x_cv      = model_parameter["x_cv"][0][0][0]  
c_pa      = model_parameter["c_pa"][0][0][0]  
c_va      = model_parameter["c_va"][0][0][0]  
r_c       = model_parameter["r_c"][0][0][0] 
gamma_a   = model_parameter["gamma_a"][0][0][0]     
eta_sc    = model_parameter["eta_sc"][0][0][0]    
T_amb     = model_parameter["T_amb"][0][0][0]   
h_tot     = model_parameter["h_tot"][0][0][0]   
d_pipe    = model_parameter["d_pipe"][0][0][0]    
l_pipe    = model_parameter["l_pipe"][0][0][0]    
n_pipe    = model_parameter["n_pipe"][0][0][0]    
R_c       = model_parameter["R_c"][0][0][0]  
c_psi_2   = model_parameter["c_psi2"][0][0][0]  
c_phi_2   = model_parameter["c_phi2"][0][0][0]  
w_copt    = model_parameter["w_copt"][0][0][0]  
c_pi      = model_parameter["c_pi"][0][0][0]  
pi_copt   = model_parameter["pi_copt"][0][0][0]  
Q_c       = model_parameter["Q_c"][0][0]  
eta_cmax  = model_parameter["eta_cmax"][0][0][0]  
V_em      = model_parameter["V_em"][0][0][0]  
J_t       = model_parameter["J_t"][0][0][0]  
tau_vgt   = model_parameter["tau_vgt"][0][0][0]  
tau_dvgt  = model_parameter["tau_dvgt"][0][0][0] 
A_vgtmax     = model_parameter["A_vgtmax"][0][0][0]
c_wpsiVec = model_parameter["c_wpsiVec"][0][0][0]
c_wphiVec = model_parameter["c_wphiVec"][0][0][0]


# Compute the predicted temperature distribution
p_im_tensor = torch.from_numpy(p_im).float().to(device)
p_em_tensor = torch.from_numpy(p_em).float().to(device)
t_tensor    = torch.from_numpy(t).float().to(device)
n_e         = torch.from_numpy(n_e.astype(np.float64)).float().to(device)
u_delta     = torch.from_numpy(u_delta).float().to(device)
u_egr       = torch.from_numpy(u_egr).float().to(device)
u_vgt       = torch.from_numpy(u_vgt).float().to(device)

# uInit_egr = torch.from_numpy(uInit_egr).float().to(device)
PI_egropt = torch.from_numpy(PI_egropt).float().to(device)
c_egr     = torch.from_numpy(c_egr    ).float().to(device)
A_egrmax_gt  = torch.from_numpy(A_egrmax_gt).float().to(device)
R_e       = torch.from_numpy(R_e.astype(np.float64)).float().to(device)
tau_degr  = torch.from_numpy(tau_degr ).float().to(device)
tau_egr1  = torch.from_numpy(tau_egr1 ).float().to(device)
tau_egr2  = torch.from_numpy(tau_egr2 ).float().to(device)
K_egr     = torch.from_numpy(K_egr    ).float().to(device)
R_a       = torch.from_numpy(R_a.astype(np.float64)).float().to(device)
T_im      = torch.from_numpy(T_im     ).float().to(device)
V_im      = torch.from_numpy(V_im     ).float().to(device)
c_pe      = torch.from_numpy(c_pe.astype(np.float64)).float().to(device)
c_f1      = torch.from_numpy(c_f1     ).float().to(device)
c_f2      = torch.from_numpy(c_f2     ).float().to(device)
c_vgt1    = torch.from_numpy(c_vgt1   ).float().to(device)
c_vgt2    = torch.from_numpy(c_vgt2   ).float().to(device)
p_amb     = torch.from_numpy(p_amb    ).float().to(device)
K_t       = torch.from_numpy(K_t      ).float().to(device)
c_m       = torch.from_numpy(c_m      ).float().to(device)
R_t       = torch.from_numpy(R_t      ).float().to(device)
gamma_e   = torch.from_numpy(gamma_e  ).float().to(device)
eta_tmmax = torch.from_numpy(eta_tmmax).float().to(device)
BSR_opt   = torch.from_numpy(BSR_opt  ).float().to(device)
c_volVec  = torch.from_numpy(c_volVec ).float().to(device)
V_d       = torch.from_numpy(V_d      ).float().to(device)
n_cyl     = torch.from_numpy(n_cyl    ).float().to(device)
q_HV      = torch.from_numpy(q_HV     ).float().to(device)
x_cv      = torch.from_numpy(x_cv     ).float().to(device)
c_pa      = torch.from_numpy(c_pa.astype(np.float64)).float().to(device)
c_va      = torch.from_numpy(c_va.astype(np.float64)).float().to(device)
r_c       = torch.from_numpy(r_c      ).float().to(device)
gamma_a   = torch.from_numpy(gamma_a  ).float().to(device)
eta_sc    = torch.from_numpy(eta_sc   ).float().to(device)
T_amb     = torch.from_numpy(T_amb    ).float().to(device)
h_tot     = torch.from_numpy(h_tot    ).float().to(device)
d_pipe    = torch.from_numpy(d_pipe   ).float().to(device)
l_pipe    = torch.from_numpy(l_pipe   ).float().to(device)
n_pipe    = torch.from_numpy(n_pipe   ).float().to(device)
R_c       = torch.from_numpy(R_c      ).float().to(device)
c_psi_2   = torch.from_numpy(c_psi_2  ).float().to(device)
c_phi_2   = torch.from_numpy(c_phi_2  ).float().to(device)
w_copt    = torch.from_numpy(w_copt   ).float().to(device)
c_pi      = torch.from_numpy(c_pi     ).float().to(device)
pi_copt   = torch.from_numpy(pi_copt  ).float().to(device)
Q_c       = torch.from_numpy(Q_c      ).float().to(device)
eta_cmax  = torch.from_numpy(eta_cmax ).float().to(device)
V_em      = torch.from_numpy(V_em     ).float().to(device)
J_t       = torch.from_numpy(J_t      ).float().to(device)
tau_vgt   = torch.from_numpy(tau_vgt  ).float().to(device)
tau_dvgt  = torch.from_numpy(tau_dvgt ).float().to(device)
omega_t   = torch.from_numpy(omega_t ).float().to(device)
T_1       = torch.from_numpy(T_1).float().to(device)
A_vgtmax  = torch.from_numpy(A_vgtmax).float().to(device)
x_r       = torch.from_numpy(x_r).float().to(device)
uegr_1_bar = torch.from_numpy(uegr_1_bar).float().to(device)
uegr_2_bar = torch.from_numpy(uegr_2_bar).float().to(device)
u_vgt_bar = torch.from_numpy(u_vgt_bar).float().to(device)
c_wpsiVec_tensor = torch.from_numpy(c_wpsiVec).float().to(device)
c_wphiVec_tensor = torch.from_numpy(c_wphiVec).float().to(device)
W_egr_tensor     = torch.from_numpy(W_egr).float().to(device)

p_im_Init = p_im_tensor[0]
p_em_Init = p_em_tensor[0]
omega_t_Init = omega_t[0]
T_1_Init =  T_1[0]
x_r_Init =  x_r[0]
uegr_1_bar_Init = uegr_1_bar[0]
uegr_2_bar_Init = uegr_2_bar[0]
u_vgt_bar_Init = u_vgt_bar[0]

# define four main parts
Turbo_ins = Turbo(c_pe, c_f1, c_f2, c_vgt1, c_vgt2, p_amb, K_t, R_e, 
                 c_m, R_t, gamma_e, eta_tmmax, BSR_opt, A_vgtmax)
EGR_ins = EGR(PI_egropt, c_egr, R_e, K_egr)
Cylinder_ins = Cylinder(c_volVec, V_d, R_a, T_im, n_cyl, q_HV, x_cv, c_pa, 
                        c_va, r_c, gamma_a, eta_sc, T_amb,
                        h_tot, d_pipe, l_pipe, n_pipe, c_pe)
Compressor_ins = Compressor(p_amb, gamma_a, c_pa, T_amb, R_c, c_psi_2, c_phi_2, R_a, w_copt,
                            c_pi, pi_copt, Q_c, eta_cmax)

# define pim\pem network
input_dim = 1
hidden_layers = 3
num_neurons = 10
output_dim = 2
pim_pem_nn = PINN(input_dim, hidden_layers, num_neurons, output_dim).to(device)

# define omega_t nn
input_dim = 1
hidden_layers = 2
num_neurons = 10
output_dim = 1
omega_t_nn = PINN(input_dim, hidden_layers, num_neurons, output_dim).to(device)

# define uegr nn
input_dim = 1
hidden_layers = 3
num_neurons = 10
output_dim = 2
uegr_1_2_nn = PINN(input_dim, hidden_layers, num_neurons, output_dim).to(device)

# define u_vgt nn
input_dim = 1
hidden_layers = 2
num_neurons = 10
output_dim = 1
u_vgt_nn = PINN(input_dim, hidden_layers, num_neurons, output_dim).to(device)

# define T_1 nn
input_dim = 1
hidden_layers = 3
num_neurons = 15
output_dim = 1
T_1_nn = PINN(input_dim, hidden_layers, num_neurons, output_dim).to(device)

# define x_r nn
input_dim = 1
hidden_layers = 2
num_neurons = 10
output_dim = 1
x_r_nn = PINN(input_dim, hidden_layers, num_neurons, output_dim).to(device)

# adaptive weights for loss term
lambda_p_im_data_l     = torch.full((t_tensor.shape[0],1), 1000.0,device=device,requires_grad=True)
lambda_p_em_data_l     = torch.full((t_tensor.shape[0],1), 1000.0,device=device,requires_grad=True)
lambda_omega_t_data_l  = torch.full((t_tensor.shape[0],1), 1000.0,device=device,requires_grad=True)
lambda_w_egr_data_l    =torch.full((t_tensor.shape[0],1), 1000.0,device=device,requires_grad=True)

lambda_p_im_data     = torch.nn.Parameter(lambda_p_im_data_l   )
lambda_p_em_data     = torch.nn.Parameter(lambda_p_em_data_l   )
lambda_omega_t_data  = torch.nn.Parameter(lambda_omega_t_data_l)
lambda_w_egr_data    = torch.nn.Parameter(lambda_w_egr_data_l  )
# lambda_p_im_phy      = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 1.0)).detach().to(device).requires_grad_(True)
# lambda_p_em_phy      = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 1.0)).detach().to(device).requires_grad_(True)
# lambda_omega_t_phy   = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 1.0)).detach().to(device).requires_grad_(True)
# lambda_u_egr1_phy    = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 1.0)).detach().to(device).requires_grad_(True)
# lambda_u_egr2_phy    = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 1.0)).detach().to(device).requires_grad_(True)
# lambda_u_vgt_bar_phy = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 1.0)).detach().to(device).requires_grad_(True)
# lambda_T_1_phy       = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 1000.0)).detach().to(device).requires_grad_(True)
# lambda_x_r_phy       = torch.nn.Parameter(torch.full((t_tensor.shape[0],1), 10.0)).detach().to(device).requires_grad_(True)

A_egrmax_train_l       = torch.tensor([3.0], requires_grad=True,device=device)
c_omega_psi_1_train_l  = torch.tensor([1.0], requires_grad=True,device=device)
c_omega_psi_2_train_l  = torch.tensor([1.5], requires_grad=True,device=device)
c_omega_psi_3_train_l  = torch.tensor([1.0], requires_grad=True,device=device)
c_omega_phi_1_train_l  = torch.tensor([1.0], requires_grad=True,device=device)
c_omega_phi_2_train_l  = torch.tensor([1.0], requires_grad=True,device=device)
c_omega_phi_3_train_l  = torch.tensor([2.5], requires_grad=True,device=device)

A_egrmax_train       = torch.nn.Parameter(A_egrmax_train_l     )
c_omega_psi_1_train  = torch.nn.Parameter(c_omega_psi_1_train_l)
c_omega_psi_2_train  = torch.nn.Parameter(c_omega_psi_2_train_l)
c_omega_psi_3_train  = torch.nn.Parameter(c_omega_psi_3_train_l)
c_omega_phi_1_train  = torch.nn.Parameter(c_omega_phi_1_train_l)
c_omega_phi_2_train  = torch.nn.Parameter(c_omega_phi_2_train_l)
c_omega_phi_3_train  = torch.nn.Parameter(c_omega_phi_3_train_l)


# optimizer
optimizer = torch.optim.Adam([{'params': pim_pem_nn.parameters()},
                            {'params': omega_t_nn.parameters()},
                            {'params': uegr_1_2_nn.parameters()},
                            {'params': u_vgt_nn.parameters()},
                            {'params': T_1_nn.parameters()},
                            {'params': x_r_nn.parameters()},
                            {'params':A_egrmax_train},
                            # {'params':c_omega_psi_1_train},
                            # {'params':c_omega_psi_2_train},
                            # {'params':c_omega_psi_3_train},
                            # {'params':c_omega_phi_1_train},
                            # {'params':c_omega_phi_2_train},
                            # {'params':c_omega_phi_3_train},
                            ], lr=1e-4)
optimizer_para = torch.optim.Adam([
                            {'params':c_omega_psi_1_train},
                            {'params':c_omega_psi_2_train},
                            {'params':c_omega_psi_3_train},
                            {'params':c_omega_phi_1_train},
                            {'params':c_omega_phi_2_train},
                            {'params':c_omega_phi_3_train},
                            ], lr=1e-3)

optimizer_adp_weights = torch.optim.Adam([{'params': lambda_p_im_data   },
                                        {'params': lambda_p_em_data     },
                                        {'params': lambda_omega_t_data  },
                                        {'params': lambda_w_egr_data     },
                                        # {'params': lambda_p_em_phy      },
                                        # {'params': lambda_omega_t_phy   },
                                        # {'params': lambda_u_egr1_phy    },
                                        # {'params': lambda_u_egr2_phy    },
                                        # {'params': lambda_u_vgt_bar_phy },
                                        # {'params': lambda_T_1_phy       },
                                        # {'params': lambda_x_r_phy       },
                                        # {'params': lambda_p_im_init     },
                                        # {'params': lambda_p_em_init     },
                                        # {'params': lambda_omega_t_init  },
                                        # {'params': lambda_u_egr1_init   },
                                        # {'params': lambda_u_egr2_init   },
                                        # {'params': lambda_u_vgt_bar_init},
                                        # {'params': lambda_T_1_init      },
                                        # {'params': lambda_x_r_init      }
                                        ], lr=1e-4)

t_tensor.requires_grad_(True)


LOSS1 = []
LOSS2 = []
LOSS3 = []
LOSS4 = []
LOSS5 = []
LOSS6 = []
LOSS7 = []
LOSS8 = []
LOSS9 = []
LOSS10 = []
LOSS11 = []
LOSS12 = []
LOSS13 = []
LOSS14 = []
LOSS15 = []
LOSS16 = []
LOSS17 = []
LOSS18 = []
LOSS19 = []
LOSSALL = []

A_EGRMAX      = []
C_OMEGA_PSI_1 = []
C_OMEGA_PSI_2 = []
C_OMEGA_PSI_3 = []
C_OMEGA_PHI_1 = []
C_OMEGA_PHI_2 = []
C_OMEGA_PHI_3 = []

LAMBDA1 = []
LAMBDA2 = []
LAMBDA3 = []
LAMBDA4 = []

WATCH = []

Epoch = 50000
epoch_for_adp_weights = 1000
for epoch in range(Epoch):
    
    # predict data
    p_im_em_pred_list = pim_pem_nn(t_tensor)
    # transfer to phy domain
    p_im_pred = p_im_em_pred_list[:,0].reshape(-1,1).requires_grad_(True)
    p_em_pred = p_im_em_pred_list[:,1].reshape(-1,1).requires_grad_(True)
    A_egrmax_pred_phy = softplus(A_egrmax_train)*1e-4
    p_im_pred_phy = (sigmoid(p_im_pred)+1.3)*1e5
    p_em_pred_phy = (sigmoid(p_em_pred)+1.6)*1e5


    omega_t_pred = omega_t_nn(t_tensor).reshape(-1,1)
    # transfer to phy domain
    omega_t_pred_phy = (sigmoid(omega_t_pred)+6.8)* 1e3
    c_omega_psi_1_pred_phy = softplus(c_omega_psi_1_train) * 1e-8
    c_omega_psi_2_pred_phy = -softplus(c_omega_psi_2_train) * 1e-4
    c_omega_psi_3_pred_phy = softplus(c_omega_psi_3_train) * 0.1
    c_omega_phi_1_pred_phy = -softplus(c_omega_phi_1_train) * 1e-8
    c_omega_phi_2_pred_phy = -softplus(c_omega_phi_2_train) * 1e-3
    c_omega_phi_3_pred_phy = softplus(c_omega_phi_3_train) * 10

    uegr_bar_list_pred  = uegr_1_2_nn(t_tensor)
    uegr_1_bar_pred     = uegr_bar_list_pred[:,0].reshape(-1,1)
    uegr_2_bar_pred     = uegr_bar_list_pred[:,1].reshape(-1,1)
    # transfer to phy domain
    uegr_1_bar_pred_phy = sigmoid(uegr_1_bar_pred) * 100
    uegr_2_bar_pred_phy = sigmoid(uegr_2_bar_pred) * 100

    u_vgt_bar_pred     = u_vgt_nn(t_tensor).reshape(-1,1)
    T_1_pred           = T_1_nn(t_tensor).reshape(-1,1)
    x_r_pred           = x_r_nn(t_tensor).reshape(-1,1)
    # transfer to phy domain
    u_vgt_bar_pred_phy = sigmoid(u_vgt_bar_pred) * 100
    T_1_pred_phy = (sigmoid(T_1_pred)+312/3) * 3
    x_r_pred_phy = (sigmoid(x_r_pred)+0.6)*0.03    # 0.031 - 0.035

    # used for calculate pim physics loss
    W_c_physics = Compressor_ins.comp_mass_flow(p_im_pred_phy, omega_t_pred_phy, 
                                            c_omega_psi_1_pred_phy, c_omega_psi_2_pred_phy, c_omega_psi_3_pred_phy, 
                                             c_omega_phi_1_pred_phy, c_omega_phi_2_pred_phy, c_omega_phi_3_pred_phy)
 

    W_f_physics, W_eo_physics, W_ei_physics = Cylinder_ins.Cylinder_flow(p_im_pred_phy, n_e, u_delta)
    T_em_physics, x_v_physics, T_e_physics, x_p_physics = Cylinder_ins.Cylinder_temperature(p_im_pred_phy, p_em_pred_phy, x_r, T_1, 
                                                                                            W_eo_physics, W_ei_physics, W_f_physics)

    W_egr_physics = EGR_ins.EGR_dynamics(p_im_pred_phy, p_em_pred_phy, uegr_1_bar, uegr_2_bar, T_em_physics, A_egrmax_gt)

    # used for calculate pem physics loss
    Pt_eta_m_physics, W_t_physics = Turbo_ins.Turbo_charger_dynamics(omega_t_pred_phy, u_vgt_bar, p_em_pred_phy, T_em_physics)
    
    # used for calculating omega_t physics loss
    p_c_physics = Compressor_ins.comp_effi(W_c_physics, p_im_pred_phy)

    loss1  = weighted_loss(lambda_p_im_data    , (p_im_pred_phy - p_im_tensor))
    loss2  = weighted_loss(lambda_p_em_data    , (p_em_pred_phy - p_em_tensor))
    loss3  = weighted_loss(lambda_omega_t_data , (omega_t_pred_phy - omega_t))
    loss4  = weighted_loss(lambda_w_egr_data   , (W_egr_physics - W_egr_tensor))
    loss5  = weighted_loss(1                   , p_im_physics_loss(p_im_pred_phy, t_tensor, R_a, T_im, V_im, W_c_physics, W_egr_physics, W_ei_physics) )
    loss6  = weighted_loss(1                   , p_em_physics_loss(p_em_pred_phy, t_tensor, R_e, T_em_physics, V_em, W_eo_physics, W_egr_physics, W_t_physics)) 
    loss7  = weighted_loss(1                   , omega_t_physics_loss(omega_t_pred_phy, t_tensor, Pt_eta_m_physics, p_c_physics, J_t))
    loss8  = weighted_loss(1                   , u_egr1_physics_loss(uegr_1_bar_pred_phy, t_tensor, tau_egr1, u_egr, tau_degr))
    loss9  = weighted_loss(1                   , u_egr2_physics_loss(uegr_2_bar_pred_phy, t_tensor, tau_egr2, u_egr, tau_degr))
    loss10 = weighted_loss(1                   , u_vgt_bar_physics_loss(u_vgt_bar_pred_phy, t_tensor, tau_vgt, u_vgt, tau_dvgt))
    loss11 = weighted_loss(10**3               , T_1_physics_loss(T_1_pred_phy, x_r_pred_phy, T_e_physics, T_im))
    loss12 = weighted_loss(10                  , x_r_physics_loss(x_r_pred_phy, p_im_pred_phy, p_em_pred_phy, gamma_a, x_p_physics, r_c, x_v_physics))
    loss13 = weighted_loss(1                   , (p_im_pred_phy[0] - p_im_Init))
    loss14 = weighted_loss(1                   , (p_em_pred_phy[0] - p_em_Init))
    loss15 = weighted_loss(1                   , (omega_t_pred_phy[0] - omega_t_Init)) 
    loss16 = weighted_loss(1                   , (uegr_1_bar_pred_phy[0] - uegr_1_bar_Init)) 
    loss17 = weighted_loss(1                   , (uegr_2_bar_pred_phy[0] - uegr_2_bar_Init))
    loss18 = weighted_loss(1                   , (u_vgt_bar_pred_phy[0] - u_vgt_bar_Init)) 
    loss19 = weighted_loss(1                   , (x_r_pred_phy[0] - x_r_Init)) 
    loss20 = weighted_loss(100                 , (T_1_pred_phy[0] - T_1_Init))
    
    LOSS1.append(loss1.detach().cpu().numpy())
    LOSS2.append(loss2.detach().cpu().numpy())
    LOSS3.append(loss3.detach().cpu().numpy())
    LOSS4.append(loss4.detach().cpu().numpy())
    LOSS5.append(loss5.detach().cpu().numpy())
    LOSS6.append(loss6.detach().cpu().numpy())
    LOSS7.append(loss7.detach().cpu().numpy())
    LOSS8.append(loss8.detach().cpu().numpy())
    LOSS9.append(loss9.detach().cpu().numpy())
    LOSS10.append(loss10.detach().cpu().numpy())
    LOSS11.append(loss11.detach().cpu().numpy())
    LOSS12.append(loss12.detach().cpu().numpy())
    LOSS13.append(loss13.detach().cpu().numpy())
    LOSS14.append(loss14.detach().cpu().numpy())
    LOSS15.append(loss15.detach().cpu().numpy())
    LOSS16.append(loss16.detach().cpu().numpy())
    LOSS17.append(loss17.detach().cpu().numpy())
    LOSS18.append(loss18.detach().cpu().numpy())
    LOSS19.append(loss18.detach().cpu().numpy())

    A_EGRMAX.append(A_egrmax_pred_phy.detach().cpu().numpy())
    C_OMEGA_PSI_1.append(c_omega_psi_1_pred_phy.detach().cpu().numpy())
    C_OMEGA_PSI_2.append(c_omega_psi_2_pred_phy.detach().cpu().numpy())
    C_OMEGA_PSI_3.append(c_omega_psi_3_pred_phy.detach().cpu().numpy())
    C_OMEGA_PHI_1.append(c_omega_phi_1_pred_phy.detach().cpu().numpy())
    C_OMEGA_PHI_2.append(c_omega_phi_2_pred_phy.detach().cpu().numpy())
    C_OMEGA_PHI_3.append(c_omega_phi_3_pred_phy.detach().cpu().numpy())

    # WATCH.append(W_c_physics.detach().cpu().numpy())
    
    if epoch % 10 == 9:  # save memory
        LAMBDA1.append(lambda_p_im_data.detach().cpu().numpy())
        LAMBDA2.append(lambda_p_em_data.detach().cpu().numpy())
        LAMBDA3.append(lambda_omega_t_data.detach().cpu().numpy())
        LAMBDA4.append(lambda_w_egr_data.detach().cpu().numpy())

    # loss
    loss =  loss1 + loss2 + loss3 + loss4 + loss5 + loss6 +\
            loss7 + loss8 + loss9 + loss10 + loss11 + loss12 +\
            loss13 + loss14 + loss15 + loss16 + loss17 + loss18 + loss19 + loss20
    
    LOSSALL.append(loss.detach().cpu().numpy())
    

    # back prop
    if epoch < epoch_for_adp_weights:  # train lambda

        # loss_adp_weights = -loss.clone()
        
        optimizer.zero_grad()
        optimizer_para.zero_grad()
        optimizer_adp_weights.zero_grad()

        loss.backward()


        optimizer.step()
        optimizer_para.step()

        for param in optimizer_adp_weights.param_groups[0]['params']:
            param.grad *= -1

        optimizer_adp_weights.step()


    else:
        optimizer.zero_grad()
        optimizer_para.zero_grad()

        loss.backward()

        optimizer.step()
        optimizer_para.step()


    if epoch % 1000 == 999:
        print('Epoch {}, Loss: {:.8e}'.format(epoch+1, loss.item()))
        print(c_omega_psi_1_train)
        print(c_omega_psi_1_pred_phy)
        print(c_omega_psi_1_train.grad)
        print(W_c_physics)

lossss = [loss1,loss2 ,loss3 , loss4 , loss5 , loss6 ,
            loss7 ,loss8 , loss9 , loss10, loss11 , loss12 ,
            loss13 , loss14 , loss15 , loss16 , loss17 , loss18]
for i in range(18):
    print("loss{} = ".format(i+1),lossss[i])
    print(" ")

save_flag = 1
### plots generation
LOSSES = [LOSS1, LOSS2, LOSS3, LOSS4, LOSS5, LOSS6, LOSS7, LOSS8, LOSS9, LOSS10,
          LOSS11, LOSS12, LOSS13, LOSS14, LOSS15, LOSS16, LOSS17, LOSS18]
# loss plot
xx = np.linspace(1,Epoch, Epoch)
for i in range(18):
    plt.figure()
    plt.plot(xx, LOSSES[i])
    plt.title("Loss{}".format(i+1))
    if save_flag == None:
        plt.show()
    else:
        plt.savefig("pic_task2_3/Loss{}.png".format(i+1))

plt.figure()
plt.plot(xx, LOSSALL)
plt.title("Total Loss")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/Total_Loss.png")

# Adaptive weights plot
plot_th = np.zeros((len(LAMBDA1), LAMBDA1[0].shape[0]))
for i in range(len(LAMBDA1)):
    plot_th[i,:] = LAMBDA1[i].reshape(1,LAMBDA1[0].shape[0])
x_time = np.arange(0,len(LAMBDA1),1)
fig2, ax2 = plt.subplots(1,1,figsize=(20,8))
axins = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.01, 0., 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0)
ax2.set_xlabel('Epoch/10', fontsize=30)
ax2.set_ylabel('Location', fontsize=30)
ax2.tick_params(axis='y', labelsize=25)
ax2.tick_params(axis='x', labelsize=25)
ax2.set_title('Lambda1', fontsize=30, pad=20)
im = ax2.imshow(plot_th.T, cmap='jet', interpolation='nearest', aspect='auto')
ax2.autoscale(False)
cbar = fig2.colorbar(im, cax=axins)
cbar.ax.tick_params(labelsize=20)
# fig2.tight_layout()  # otherwise the right y-label is slightly clipped
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/Lambda1.png")

plot_th = np.zeros((len(LAMBDA2), LAMBDA2[0].shape[0]))
for i in range(len(LAMBDA2)):
    plot_th[i,:] = LAMBDA2[i].reshape(1,LAMBDA2[0].shape[0])
x_time = np.arange(0,len(LAMBDA2),1)
fig2, ax2 = plt.subplots(1,1,figsize=(20,8))
axins = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.01, 0., 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0)
ax2.set_xlabel('Epoch/10', fontsize=30)
ax2.set_ylabel('Location', fontsize=30)
ax2.tick_params(axis='y', labelsize=25)
ax2.tick_params(axis='x', labelsize=25)
ax2.set_title('Lambda2', fontsize=30, pad=20)
im = ax2.imshow(plot_th.T, cmap='jet', interpolation='nearest', aspect='auto')
ax2.autoscale(False)
cbar = fig2.colorbar(im, cax=axins)
cbar.ax.tick_params(labelsize=20)
# fig2.tight_layout()  # otherwise the right y-label is slightly clipped
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/Lambda2.png")
    plt.close()

plot_th = np.zeros((len(LAMBDA3), LAMBDA3[0].shape[0]))
for i in range(len(LAMBDA3)):
    plot_th[i,:] = LAMBDA3[i].reshape(1,LAMBDA3[0].shape[0])
x_time = np.arange(0,len(LAMBDA3),1)
fig2, ax2 = plt.subplots(1,1,figsize=(20,8))
axins = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.01, 0., 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0)
ax2.set_xlabel('Epoch/10', fontsize=30)
ax2.set_ylabel('Location', fontsize=30)
ax2.tick_params(axis='y', labelsize=25)
ax2.tick_params(axis='x', labelsize=25)
ax2.set_title('LAMBDA3', fontsize=30, pad=20)
im = ax2.imshow(plot_th.T, cmap='jet', interpolation='nearest', aspect='auto')
ax2.autoscale(False)
cbar = fig2.colorbar(im, cax=axins)
cbar.ax.tick_params(labelsize=20)
# fig2.tight_layout()  # otherwise the right y-label is slightly clipped
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/Lambda3.png")
    plt.close()

plot_th = np.zeros((len(LAMBDA4), LAMBDA4[0].shape[0]))
for i in range(len(LAMBDA4)):
    plot_th[i,:] = LAMBDA4[i].reshape(1,LAMBDA4[0].shape[0])
x_time = np.arange(0,len(LAMBDA4),1)
fig2, ax2 = plt.subplots(1,1,figsize=(20,8))
axins = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 50%
                    loc='lower left',
                    bbox_to_anchor=(1.01, 0., 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0)
ax2.set_xlabel('Epoch/10', fontsize=30)
ax2.set_ylabel('Location', fontsize=30)
ax2.tick_params(axis='y', labelsize=25)
ax2.tick_params(axis='x', labelsize=25)
ax2.set_title('LAMBDA4', fontsize=30, pad=20)
im = ax2.imshow(plot_th.T, cmap='jet', interpolation='nearest', aspect='auto')
ax2.autoscale(False)
cbar = fig2.colorbar(im, cax=axins)
cbar.ax.tick_params(labelsize=20)
# fig2.tight_layout()  # otherwise the right y-label is slightly clipped
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/LAMBDA4.png")
    plt.close()

# unknown parameter plot
plt.figure()
plt.plot(xx, A_EGRMAX,label="Pred")
plt.axhline(y=A_egrmax_gt.detach().cpu().numpy(),color='r',linestyle='--', label='GT')
plt.legend()
plt.title("A_EGRMAX Prediction")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/A_EGRMAX_Pred.png")


plt.figure()
plt.plot(xx, C_OMEGA_PSI_1,label="Pred")
plt.axhline(y=c_wpsiVec_tensor[0].detach().cpu().numpy(),color='r',linestyle='--', label='GT')
plt.title("C_OMEGA_PSI_1 Prediction")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/C_OMEGA_PSI_1_Pred.png")

plt.figure()
plt.plot(xx, C_OMEGA_PSI_2,label="Pred")
plt.axhline(y=c_wpsiVec_tensor[1].detach().cpu().numpy(),color='r',linestyle='--', label='GT')
plt.title("C_OMEGA_PSI_2 Prediction")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/C_OMEGA_PSI_2_Pred.png")

plt.figure()
plt.plot(xx, C_OMEGA_PSI_3,label="Pred")
plt.axhline(y=c_wpsiVec_tensor[2].detach().cpu().numpy(),color='r',linestyle='--', label='GT')
plt.title("C_OMEGA_PSI_3 Prediction")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/C_OMEGA_PSI_3_Pred.png")

plt.figure()
plt.plot(xx, C_OMEGA_PHI_1,label="Pred")
plt.axhline(y=c_wphiVec_tensor[0].detach().cpu().numpy(),color='r',linestyle='--', label='GT')
plt.title("C_OMEGA_PHI_1 Prediction")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/C_OMEGA_PHI_1_Pred.png")

plt.figure()
plt.plot(xx, C_OMEGA_PHI_2,label="Pred")
plt.axhline(y=c_wphiVec_tensor[1].detach().cpu().numpy(),color='r',linestyle='--', label='GT')
plt.title("C_OMEGA_PHI_2 Prediction")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/C_OMEGA_PHI_2_Pred.png")

plt.figure()
plt.plot(xx, C_OMEGA_PHI_3,label="Pred")
plt.axhline(y=c_wphiVec_tensor[2].detach().cpu().numpy(),color='r',linestyle='--', label='GT')
plt.title("C_OMEGA_PHI_3 Prediction")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/C_OMEGA_PHI_3_Pred.png")


## save model
torch.save(pim_pem_nn.state_dict() , 'model_task2_3/pim_pem_nn.pth')
torch.save(omega_t_nn.state_dict() , 'model_task2_3/omega_t_nn.pth')
torch.save(uegr_1_2_nn.state_dict(), 'model_task2_3/uegr_1_2_nn.pth')
torch.save(u_vgt_nn.state_dict()   , 'model_task2_3/u_vgt_nn.pth')
torch.save(T_1_nn.state_dict()     , 'model_task2_3/T_1_nn.pth')
torch.save(x_r_nn.state_dict()     , 'model_task2_3/x_r_nn.pth')

## test model on train set
# p_im_pred_phy_norm = normalize(p_im_pred_phy)
# p_im_tensor_norm   = normalize(p_im_tensor)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), p_im_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), p_im_tensor.detach().cpu().numpy(), '--',label="GT")
plt.title("p_im")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/p_im.png")

# p_em_pred_phy_norm = normalize(p_em_pred_phy)
# p_em_tensor_norm   = normalize(p_em_tensor)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), p_em_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), p_em_tensor.detach().cpu().numpy(), '--',label="GT")
plt.title("p_em")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/p_em.png")

# uegr_1_bar_pred_phy_norm = normalize(uegr_1_bar_pred_phy)
# uegr_1_bar_norm   = normalize(uegr_1_bar)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), uegr_1_bar_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), uegr_1_bar.detach().cpu().numpy(), '--',label="GT")
plt.title("u_egr_bar_1")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/u_egr_bar_1.png")

# uegr_2_bar_pred_phy_norm = normalize(uegr_2_bar_pred_phy)
# uegr_2_bar_norm   = normalize(uegr_2_bar)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), uegr_2_bar_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), uegr_2_bar.detach().cpu().numpy(), '--',label="GT")
plt.title("u_egr_bar_2")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/u_egr_bar_2.png")

# u_vgt_bar_pred_phy_norm = normalize(u_vgt_bar_pred_phy)
# u_vgt_bar_norm   = normalize(u_vgt_bar)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), u_vgt_bar_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), u_vgt_bar.detach().cpu().numpy(), '--',label="GT")
plt.title("u_vgt_bar")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/u_vgt_bar.png")

# omega_t_pred_phy_norm = normalize(omega_t_pred_phy)
# omega_t_norm   = normalize(omega_t)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), omega_t_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), omega_t.detach().cpu().numpy(), '--',label="GT")
plt.title("omega_t")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/omega_t.png")

# T_1_pred_phy_norm = normalize(T_1_pred_phy)
# T_1_norm   = normalize(T_1)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), T_1_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), T_1.detach().cpu().numpy(), '--',label="GT")
plt.title("T_1")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/T_1.png")

# x_r_pred_phy_norm = normalize(x_r_pred_phy)
# x_r_norm   = normalize(x_r)
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), x_r_pred_phy.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), x_r.detach().cpu().numpy(), '--',label="GT")
plt.title("x_r")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task2_3/x_r.png")
