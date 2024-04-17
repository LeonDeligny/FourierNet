import numpy as np, pandas as pd
import torch, os, logging

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from preprocessing import load_dataset, normalize

# Set up Python logging
logging.basicConfig(level=logging.ERROR)

# Create a log directory with a timestamp to keep different runs separate
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

pd.set_option('display.precision', 20)

# torch.manual_seed(1234)
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":

    # Load Data (change path if needed)
    path = ["airFoil2D_SST_31.68_0.424_0.273_4.301_1.0_11.616"]
    train_data, len_list = load_dataset(path, 
                                       n_random_sampling = 50000
                                       )

    Uinf, alpha, gamma_1, gamma_2, gamma_3 = float(path[0].split('_')[2]), float(path[0].split('_')[3])*np.pi/180, float(path[0].split('_')[4]), float(path[0].split('_')[5]), float(path[0].split('_')[6])
    print(f"Uinf: {Uinf}, alpha: {alpha}")
    
    u_inlet, v_inlet = np.cos(alpha)*Uinf, np.sin(alpha)*Uinf
    
    df_train_input = pd.DataFrame(train_data[0].x_train, columns=["x", "y", "sdf", "x_n", "y_n"])
    df_train_target = pd.DataFrame(train_data[0].y_train, columns=["u", "v", "p", "nut"])
    df_train = pd.concat([df_train_input, df_train_target], axis=1) 

    df_train_n, _ = normalize(df_train)
    mean_variance_dict = {column: {"mean": 0.0, "var": 1.0} for i, column in enumerate(df_train.columns)}
    df_aerofoil = df_train.iloc[len_list[0]:len_list[0]+len_list[1],:]

    df_freestream_n = df_train_n.iloc[:len_list[0],:]
    df_aerofoil_n = df_train_n.iloc[len_list[0]:len_list[0]+len_list[1],:]
    df_box_n = df_train_n.iloc[len_list[0]:len_list[0]+len_list[1]+len_list[2],:]
        
    print("Datasets Loaded.")

    from FourierNeuralOperatorNN import FourierNeuralOperatorNN
    from plot import plot_predictions_vs_test, plot_test

    # Train the model
    model = FourierNeuralOperatorNN(df_train, df_aerofoil, mean_variance_dict, len_list, u_inlet, v_inlet, gamma_1, gamma_2, gamma_3)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using devide : {device}")

    model.to(device)

    print(f"Started Training.")
    model.train(11)
    print(f"Finished Training.")
