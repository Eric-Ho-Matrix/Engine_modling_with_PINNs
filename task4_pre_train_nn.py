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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    

def normalize(x_tensor):
    x_min = torch.min(x_tensor)
    x_max = torch.max(x_tensor)
    x_tensor_ret = (x_tensor - x_min) / ((x_max - x_min)+1e-12)
    return x_tensor_ret

################ End of Definition ##################

# Data Extraction
model_parameter = loadmat('WahlstromErikssonTCDI_EGR_VGT/parameterData.mat')['model']
workspace_sim = loadmat('WahlstromErikssonTCDI_EGR_VGT/Workspace.mat')

p_amb = model_parameter["p_amb"][0][0][0]
T_amb = model_parameter["p_amb"][0][0][0]
t      = workspace_sim['simTime']
PHI_C_GT = workspace_sim['check_phi_c']
p_im = workspace_sim['simp_im']
omega_t = workspace_sim['sim_omega_t']

# Compute the predicted temperature distribution
p_amb_tensor      = torch.from_numpy(p_amb).float().to(device)
t_tensor          = torch.from_numpy(t).float().to(device)
PHI_C_GT_tensor   = torch.from_numpy(PHI_C_GT).float().to(device)
p_im_tensor       = torch.from_numpy(p_im).float().to(device)
T_amb_tensor      = torch.from_numpy(T_amb).float().to(device)
omega_t_tensor    = torch.from_numpy(omega_t).float().to(device)

# define pim\pem network
input_dim = 1
hidden_layers = 8
num_neurons = 50
output_dim = 1
PHI_C_UPPER= PINN(input_dim, hidden_layers, num_neurons, output_dim).to(device)

# optimizer
optimizer = torch.optim.Adam([{'params': PHI_C_UPPER.parameters()}], lr=1e-3)
criterion = torch.nn.MSELoss()

pi_c = p_im_tensor / p_amb_tensor

Epoch = 50000
L_2_ERR = []
for epoch in range(Epoch):
    
    # predict data
    PHI_C_UPPER_pred= PHI_C_UPPER(t_tensor)

    loss = criterion(PHI_C_UPPER_pred, PHI_C_GT_tensor)
    L2_error = abs((PHI_C_UPPER_pred - PHI_C_GT_tensor)) / PHI_C_GT_tensor
    L_2_ERR.append(L2_error.detach().cpu().numpy())

    # back prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 999:
        print('Epoch {}, L2_Error: {:.8e}'.format(epoch+1, np.mean(L2_error.detach().cpu().numpy())))

save_flag = 1
# loss plot
xx = np.linspace(1,Epoch, Epoch)
plt.figure()
plt.plot(xx, L2_error)
plt.title("Total Loss")
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task4/Total_Loss.png")

## save model
torch.save(PHI_C_UPPER.state_dict() , 'model_task4/PHI_C.pth')

## test model on train set
plt.figure()
plt.plot(t_tensor.detach().cpu().numpy(), PHI_C_UPPER_pred.detach().cpu().numpy(), label="pred")
plt.plot(t_tensor.detach().cpu().numpy(), PHI_C_GT_tensor.detach().cpu().numpy(), '--',label="GT")
plt.title("PHI_C_UPPER")
plt.legend()
if save_flag == None:
    plt.show()
else:
    plt.savefig("pic_task4/PHI_C_UPPER.png")
    plt.close()