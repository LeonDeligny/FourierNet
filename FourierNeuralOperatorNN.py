import torch, os

import numpy as np

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# Properties of air at sea level and 293.15K
RHO = 1.184
NU = 1.56e-5
C = 346.1
P_ref = 1.013e5


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierLayer(torch.nn.Module):
    def __init__(self, modes, width):
        super(FourierLayer, self).__init__()
        self.modes = modes
        self.width = width
        self.scale = (2 * np.pi) ** 0.5
        self.weights = torch.nn.Parameter(torch.randn(self.width, self.width, 2))

    def forward(self, x):
        grid = self.get_grid(x.shape[-1])
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        for i in range(self.width):
            for j in range(self.width):
                k1, k2 = int(grid[i, j, 0] * self.scale), int(grid[i, j, 1] * self.scale)
                weight = self.weights[i, j]
                complex_weight = torch.complex(weight[0], weight[1])
                out_ft += complex_weight * x_ft.roll(shifts=(-k1, -k2), dims=(-2, -1))
        return torch.fft.irfft2(out_ft)

    def get_grid(self, size):
        grid = torch.stack(torch.meshgrid(torch.fft.fftfreq(size), torch.fft.fftfreq(size), indexing='ij'), dim=-1)
        return grid.to(device)

class FNO2d(torch.nn.Module):
    def __init__(self, modes, width):
        super(FNO2d, self).__init__()
        self.fourier_layer = FourierLayer(modes, width)
        self.fc1 = torch.nn.Linear(width, width)
        self.fc2 = torch.nn.Linear(width, width)

    def forward(self, x):
        x = self.fourier_layer(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x

class FourierNeuralOperatorNN(torch.nn.Module):
    def __init__(self, modes, width, df_train, df_aerofoil, mean_variance_dict, len_list, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3):
        super(FourierNeuralOperatorNN, self).__init__()
        self.x = torch.tensor(df_train['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.y = torch.tensor(df_train['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.sdf = torch.tensor(df_train['sdf'].astype(float).values).float().unsqueeze(1).to(device)
        self.x_n_a = torch.tensor(df_aerofoil['x_n'].astype(float).values).float().unsqueeze(1).to(device) # Normals to aerofoil
        self.y_n_a = torch.tensor(df_aerofoil['y_n'].astype(float).values).float().unsqueeze(1).to(device)

        self.u = torch.tensor(df_train['u'].astype(float).values).float().unsqueeze(1).to(device)
        self.v = torch.tensor(df_train['v'].astype(float).values).float().unsqueeze(1).to(device)
        self.nut = torch.tensor(df_train['nut'].astype(float).values).float().unsqueeze(1).to(device)
        self.p = torch.tensor(df_train['p'].astype(float).values).float().unsqueeze(1).to(device)

        self.u_inlet = torch.full((len(self.x), 1), fill_value=u_inlet).float().to(device)
        self.v_inlet = torch.full((len(self.x), 1), fill_value=v_inlet).float().to(device)
        self.gamma_1 = torch.full((len(self.x), 1), fill_value=gamma_1).float().to(device)
        self.gamma_2 = torch.full((len(self.x), 1), fill_value=gamma_2).float().to(device)
        self.gamma_3 = torch.full((len(self.x), 1), fill_value=gamma_3).float().to(device)

        self.mean_variance_dict = mean_variance_dict
        self.len_list = len_list

        self.u_net = FNO2d(modes, width)
        self.v_net = FNO2d(modes, width)
        self.p_net = FNO2d(modes, width)
        self.nut_net = FNO2d(modes, width)

        self.u_optimizer = optim.Adam(self.u_net.parameters(), lr=0.001)
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=0.001)
        self.p_optimizer = optim.Adam(self.p_net.parameters(), lr=0.001)
        self.nut_optimizer = optim.Adam(self.nut_net.parameters(), lr=0.001)

        self.u_scheduler = ExponentialLR(self.u_optimizer, gamma=0.95)
        self.v_scheduler = ExponentialLR(self.v_optimizer, gamma=0.95)
        self.p_scheduler = ExponentialLR(self.p_optimizer, gamma=0.95)
        self.nut_scheduler = ExponentialLR(self.nut_optimizer, gamma=0.95)

        self.loss_func = torch.nn.MSELoss()


    def fu_fv_ic_normalized_compute(self, mean_variance_dict, u, u_x, u_y, u_xx, u_yy, v, v_x, v_y, v_xx, v_yy, p_x, p_y, nut, nut_x, nut_y):
        f_u = (2 * (u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['u']['var'] * u_x) / mean_variance_dict['x']['var'] \
            + ((u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['v']['var'] * v_y) / mean_variance_dict['y']['var'] \
            + (mean_variance_dict['u']['var'] * u_y * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean'])) / mean_variance_dict['y']['var'] \
            + (mean_variance_dict['p']['var'] * p_x) / mean_variance_dict['x']['var'] \
            - (mean_variance_dict['nut']['var'] * nut_x * mean_variance_dict['u']['var'] * u_x) / (mean_variance_dict['x']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['u']['var'] ** 2) * u_xx) / (mean_variance_dict['x']['var'] ** 2) \
            - (mean_variance_dict['nut']['var'] * nut_y * mean_variance_dict['u']['var'] * u_y) / (mean_variance_dict['y']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['u']['var'] ** 2) * u_yy) / (mean_variance_dict['y']['var'] ** 2) 
        
        f_v = (2 * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean']) * mean_variance_dict['v']['var'] * v_x) / mean_variance_dict['y']['var'] \
            + ((u * mean_variance_dict['u']['var'] + mean_variance_dict['u']['mean']) * mean_variance_dict['v']['var'] * v_x) / mean_variance_dict['x']['var'] \
            + (mean_variance_dict['u']['var'] * u_x * (v * mean_variance_dict['v']['var'] + mean_variance_dict['v']['mean'])) / mean_variance_dict['x']['var'] \
            + (mean_variance_dict['p']['var'] * p_y) / mean_variance_dict['y']['var'] \
            - (mean_variance_dict['nut']['var'] * nut_x * mean_variance_dict['v']['var'] * v_x) / (mean_variance_dict['x']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['v']['var'] ** 2) * v_xx) / (mean_variance_dict['x']['var'] ** 2) \
            - (mean_variance_dict['nut']['var'] * nut_y * mean_variance_dict['v']['var'] * v_y) / (mean_variance_dict['y']['var'] ** 2) \
            - ((NU + (nut * mean_variance_dict['nut']['var'] + mean_variance_dict['nut']['mean'])) * (mean_variance_dict['v']['var'] ** 2) * v_yy) / (mean_variance_dict['y']['var'] ** 2) 
    
        ic = ((mean_variance_dict['u']['var'] / mean_variance_dict['x']['var']) * u_x) \
            + ((mean_variance_dict['v']['var'] / mean_variance_dict['y']['var']) * v_y) # Incompressibility condition

        return f_u, f_v, ic
    
    def bc_normalized_compute(self, mean_variance_dict, p_x_a, p_y_a, x_n_a, y_n_a):
        x_n_a_normalized = (x_n_a * mean_variance_dict['x_n']['var'] + mean_variance_dict['x_n']['mean'])
        y_n_a_normalized = (y_n_a * mean_variance_dict['y_n']['var'] + mean_variance_dict['y_n']['mean'])

        bc = mean_variance_dict['p']['var'] / torch.sqrt(x_n_a_normalized**2 + y_n_a_normalized**2) \
              * (x_n_a_normalized * p_x_a / mean_variance_dict['x']['var'] \
              + y_n_a_normalized * p_y_a / mean_variance_dict['y']['var']) \
                
        return bc
    
    def net_NS(self, mean_variance_dict, len_list, x_n_a, y_n_a, x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3):
        inputs = torch.cat([x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3], dim=1)

        u = self.u_net(inputs)
        v = self.v_net(inputs)
        p = self.p_net(inputs)
        nut = self.nut_net(inputs)

        u_f, u_a = u[:len_list[0], :], u[len_list[0]:len_list[0]+len_list[1],:]
        v_f, v_a = v[:len_list[0], :], v[len_list[0]:len_list[0]+len_list[1],:]
        p_f = p[:len_list[0], :]
        nut_f, nut_a = nut[:len_list[0], :], nut[len_list[0]:len_list[0]+len_list[1],:]

        u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        p_x_a = p_x[len_list[0]:len_list[0]+len_list[1],:] # Gradient of pressure on airfoil
        p_y_a = p_y[len_list[0]:len_list[0]+len_list[1],:]

        nut_x = grad(nut, x, grad_outputs=torch.ones_like(nut), create_graph=True)[0]
        nut_y = grad(nut, y, grad_outputs=torch.ones_like(nut), create_graph=True)[0]
        
        f_u, f_v, ic = self.fu_fv_ic_normalized_compute(mean_variance_dict, u, u_x, u_y, u_xx, u_yy, v, v_x, v_y, v_xx, v_yy, p_x, p_y, nut, nut_x, nut_y)
        bc = self.bc_normalized_compute(mean_variance_dict, p_x_a, p_y_a, x_n_a, y_n_a)

        return u, v, p, nut, u_f, v_f, p_f, nut_f, u_a, v_a, nut_a, f_u, f_v, ic, bc

    def forward(self, mean_variance_dict, len_list, x_n_a, y_n_a, x, y, u_inlet, v_inlet, sdf, gamma_1, gamma_2, gamma_3):
        u_pred, v_pred, p_pred, nut_pred, \
        u_f_pred, v_f_pred, p_f_pred, nut_f_pred, \
        u_a_pred, v_a_pred, nut_a_pred, \
        f_u_pred, f_v_pred, ic_pred, bc_pred = self.net_NS( 
                                                    mean_variance_dict, len_list,
                                                    x_n_a, y_n_a,
                                                    x, y, u_inlet, v_inlet, 
                                                    sdf, gamma_1, gamma_2, gamma_3
                                                        )
        
    
        f_u_loss, f_v_loss = self.loss_func(f_u_pred, torch.zeros_like(f_u_pred)), self.loss_func(f_v_pred, torch.zeros_like(f_v_pred))
        # rans_loss = f_u_loss + f_v_loss
        f_u_loss_norm, f_v_loss_norm = f_u_loss / (torch.abs(f_u_pred).mean() + 1e-8), f_v_loss / (torch.abs(f_v_pred).mean() + 1e-8)
        rans_loss_norm = f_u_loss_norm + f_v_loss_norm

        ic_loss = self.loss_func(ic_pred, torch.zeros_like(ic_pred))
        ic_loss_norm = ic_loss / (torch.abs(ic_pred).mean() + 1e-8)

        bc_loss = self.loss_func(bc_pred, torch.zeros_like(bc_pred))
        bc_loss_norm = bc_loss / (torch.abs(bc_loss).mean() + 1e-8)

        u_a_loss = self.loss_func(self.u[len_list[0]:len_list[0]+len_list[1],:], u_a_pred)
        u_f_loss = self.loss_func(self.u[:len_list[0], :], u_f_pred)
        v_a_loss = self.loss_func(self.v[len_list[0]:len_list[0]+len_list[1],:], v_a_pred)
        v_f_loss = self.loss_func(self.v[:len_list[0], :], v_f_pred)
        nut_f_loss = self.loss_func(self.nut[:len_list[0], :], nut_f_pred)
        nut_a_loss = self.loss_func(self.nut[len_list[0]:len_list[0]+len_list[1],:], nut_a_pred)
        p_f_loss = self.loss_func(self.p[:len_list[0], :], p_f_pred)
        
        u_train_loss = self.loss_func(self.u[len_list[0]:len_list[0]+len_list[1]+len_list[2],:], u_pred[len_list[0]:len_list[0]+len_list[1]+len_list[2],:]) 
        v_train_loss = self.loss_func(self.v[len_list[0]:len_list[0]+len_list[1]+len_list[2],:], v_pred[len_list[0]:len_list[0]+len_list[1]+len_list[2],:])
        p_train_loss = self.loss_func(self.p[len_list[0]:len_list[0]+len_list[1]+len_list[2],:], p_pred[len_list[0]:len_list[0]+len_list[1]+len_list[2],:]) 
        nut_train_loss = self.loss_func(self.nut[len_list[0]:len_list[0]+len_list[1]+len_list[2],:], nut_pred[len_list[0]:len_list[0]+len_list[1]+len_list[2],:])

        u_loss = u_train_loss + ic_loss_norm + rans_loss_norm # ic_loss + rans_loss # self.loss_func(self.u, u_pred) 
        v_loss = v_train_loss + ic_loss_norm + rans_loss_norm # ic_loss + rans_loss # self.loss_func(self.v, v_pred)
        p_loss = p_train_loss + bc_loss_norm + rans_loss_norm # rans_loss # self.loss_func(self.p, p_pred) 
        nut_loss = nut_train_loss + rans_loss_norm # rans_loss # self.loss_func(self.nut, nut_pred)

        return u_loss, v_loss, p_loss, nut_loss, u_train_loss, v_train_loss, p_train_loss, nut_train_loss, f_u_loss, f_v_loss, ic_loss, bc_loss

    def train(self, nIter, checkpoint_path='path_to_checkpoint.pth'):
        # Temporary storage for loss values for logging purposes
        self.temp_losses = {}
        self.display = {}

        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            
            self.u_net.load_state_dict(checkpoint['u_net_state_dict'])
            self.u_optimizer.load_state_dict(checkpoint['u_optimizer_state_dict'])
            self.u_scheduler.load_state_dict(checkpoint['u_scheduler_state_dict'])

            self.v_net.load_state_dict(checkpoint['v_net_state_dict'])
            self.v_optimizer.load_state_dict(checkpoint['v_optimizer_state_dict'])
            self.v_scheduler.load_state_dict(checkpoint['v_scheduler_state_dict'])

            self.p_net.load_state_dict(checkpoint['p_net_state_dict'])
            self.p_optimizer.load_state_dict(checkpoint['p_optimizer_state_dict'])
            self.p_scheduler.load_state_dict(checkpoint['p_scheduler_state_dict'])

            self.nut_net.load_state_dict(checkpoint['nut_net_state_dict'])
            self.nut_optimizer.load_state_dict(checkpoint['nut_optimizer_state_dict'])
            self.nut_scheduler.load_state_dict(checkpoint['nut_scheduler_state_dict'])

            # Restore the RNG state
            torch.set_rng_state(checkpoint['rng_state'])

            # If you're resuming training and want to start from the next iteration,
            # make sure to load the last iteration count and add one
            start_iteration = checkpoint.get('iterations', 0) + 1
            print(f"Resuming from iteration {start_iteration}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")
            start_iteration = 0

        def compute_losses():
            # Compute all losses
            losses = self.forward(self.mean_variance_dict, self.len_list,
                                self.x_n_a, self.y_n_a,
                                self.x, self.y, self.u_inlet, self.v_inlet, 
                                self.sdf, self.gamma_1, self.gamma_2, self.gamma_3
                                )

            # Unpack the losses and store them in a dictionary for easy access
            (u_loss, v_loss, p_loss, nut_loss, u_train_loss, v_train_loss, p_train_loss, nut_train_loss, f_u_loss, f_v_loss, ic_loss, bc_loss) = losses

            self.temp_losses = {'u_loss': u_loss, 'v_loss': v_loss, 'p_loss': p_loss, 'nut_loss': nut_loss}

            self.display = {
                            'u_loss': u_loss, 'v_loss': v_loss, 'p_loss': p_loss, 'nut_loss': nut_loss,
                            'u_train_loss': u_train_loss, 'v_train_loss': v_train_loss, 'p_train_loss': p_train_loss, 'nut_train_loss': nut_train_loss,
                            # 'u_f_loss': u_f_loss, 'v_f_loss': v_f_loss, 'p_f_loss': p_f_loss, 'nut_f_loss': nut_f_loss, 
                            # 'u_a_loss': u_a_loss, 'v_a_loss': v_a_loss, 'nut_a_loss': nut_a_loss, 
                            'f_u_loss': f_u_loss, 'f_v_loss': f_v_loss, 'ic_loss': ic_loss, 'bc_loss': bc_loss
                        }


        for it in range(start_iteration, nIter + start_iteration):
                
            compute_losses()

            self.u_optimizer.zero_grad()
            self.temp_losses['u_loss'].backward()
            self.u_optimizer.step()
            self.u_scheduler.step()

            self.v_optimizer.zero_grad()
            self.temp_losses['v_loss'].backward()
            self.v_optimizer.step()
            self.v_scheduler.step()

            self.p_optimizer.zero_grad()
            self.temp_losses['p_loss'].backward()
            self.p_optimizer.step()
            self.p_scheduler.step()

            self.nut_optimizer.zero_grad()
            self.temp_losses['nut_loss'].backward()
            self.nut_optimizer.step()
            self.nut_scheduler.step()

            if it % 2 == 0:
                print(f"Iteration: {it}")
            if it % 10 == 0:  # Print losses every 10 iterations
                for name, value in self.display.items():
                    print(f"{name}: {value.item()}")

                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                
                checkpoint = {
                    'u_net_state_dict': self.u_net.state_dict(),
                    'u_optimizer_state_dict': self.u_optimizer.state_dict(),
                    'u_scheduler_state_dict': self.u_scheduler.state_dict(),

                    'v_net_state_dict': self.v_net.state_dict(),
                    'v_optimizer_state_dict': self.v_optimizer.state_dict(),
                    'v_scheduler_state_dict': self.v_scheduler.state_dict(),

                    'p_net_state_dict': self.p_net.state_dict(),
                    'p_optimizer_state_dict': self.p_optimizer.state_dict(),
                    'p_scheduler_state_dict': self.p_scheduler.state_dict(),

                    'nut_net_state_dict': self.nut_net.state_dict(),
                    'nut_optimizer_state_dict': self.nut_optimizer.state_dict(),
                    'nut_scheduler_state_dict': self.nut_scheduler.state_dict(),

                    'iterations': it,

                    'rng_state': torch.get_rng_state(),
                }

                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to '{checkpoint_path}' at iteration {it}")



    def predict(self, df_test, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3):
        x_star = torch.tensor(df_test['x'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        y_star = torch.tensor(df_test['y'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)

        sdf_star = torch.tensor(df_test['sdf'].astype(float).values).float().unsqueeze(1).to(device)
                
        u_inlet_star = torch.full((len(x_star), 1), fill_value=u_inlet).float().to(device)
        v_inlet_star = torch.full((len(x_star), 1), fill_value=v_inlet).float().to(device)

        gamma_1_star = torch.full((len(x_star), 1), fill_value=gamma_1).float().to(device)
        gamma_2_star = torch.full((len(x_star), 1), fill_value=gamma_2).float().to(device)
        gamma_3_star = torch.full((len(x_star), 1), fill_value=gamma_3).float().to(device)

        inputs = torch.cat([x_star, y_star, u_inlet_star, v_inlet_star, sdf_star, gamma_1_star, gamma_2_star, gamma_3_star], dim=1)

        u_star = self.u_net(inputs)
        v_star = self.v_net(inputs)
        p_star = self.p_net(inputs)
        nut_star = self.nut_net(inputs)

        return u_star.cpu().detach().numpy(), v_star.cpu().detach().numpy(), p_star.cpu().detach().numpy(), nut_star.cpu().detach().numpy()