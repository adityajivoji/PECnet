import torch

import sys
from torch.utils.data import DataLoader
import argparse
import copy
sys.path.append("../utils/")
import matplotlib.pyplot as plt
import numpy as np
from models import *
from social_utils import *
import yaml
import matplotlib.pyplot as plt
import numpy as np
# InteractiveShell.ast_node_interactivity = "all"
# from tqdm import tqdm
# import plotly.express as px
import argparse
# from sklearn.model_selection import train_test_split
import os
import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
from supermarket.dataset.dataloader import Supermarket
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import os

def plot_trajectory(loc, loc_end, loc_pred_head, epoch, save_dir=None):
    
	plt.figure()  # Create a new figure for each plot
    # print(type(loc), type(loc_end), type(loc_pred_head))
	loc_pred = loc_pred_head
	# Extract x and y coordinates from loc, loc_end, and loc_pred
	x_loc, y_loc = zip(*loc)
	x_loc_end, y_loc_end = zip(*loc_end)
	x_loc_pred, y_loc_pred = zip(*loc_pred)

	# Generate more points to create smooth curves
	t_loc = np.arange(len(loc))
	t_loc_end = np.arange(len(loc_end))
	t_loc_pred = np.arange(len(loc_pred))

	t_loc_new = np.linspace(t_loc.min(), t_loc.max(), 300)
	t_loc_end_new = np.linspace(t_loc_end.min(), t_loc_end.max(), 300)
	t_loc_pred_new = np.linspace(t_loc_pred.min(), t_loc_pred.max(), 300)

	spl_loc = make_interp_spline(t_loc, np.column_stack((x_loc, y_loc)), k=3)
	spl_loc_end = make_interp_spline(t_loc_end, np.column_stack((x_loc_end, y_loc_end)), k=3)
	spl_loc_pred = make_interp_spline(t_loc_pred, np.column_stack((x_loc_pred, y_loc_pred)), k=3)

	loc_smooth = spl_loc(t_loc_new)
	loc_end_smooth = spl_loc_end(t_loc_end_new)
	loc_pred_smooth = spl_loc_pred(t_loc_pred_new)

	x_loc_smooth, y_loc_smooth = loc_smooth[:, 0], loc_smooth[:, 1]
	x_loc_end_smooth, y_loc_end_smooth = loc_end_smooth[:, 0], loc_end_smooth[:, 1]
	x_loc_pred_smooth, y_loc_pred_smooth = loc_pred_smooth[:, 0], loc_pred_smooth[:, 1]

	# Plot the trajectories
	plt.plot(x_loc_smooth, y_loc_smooth, color='blue', label='loc')
	plt.plot(x_loc_end_smooth, y_loc_end_smooth, color='green', label='loc_end')
	plt.plot(x_loc_pred_smooth, y_loc_pred_smooth, color='orange', label='loc_pred')

	# Scatter plot the original coordinates
	plt.scatter(x_loc, y_loc, color='blue')
	plt.scatter(x_loc_end, y_loc_end, color='green')
	plt.scatter(x_loc_pred, y_loc_pred, color='orange')

	# Connect the last point of loc to the first point of loc_end and loc_pred
	plt.plot([x_loc[-1], x_loc_end[0]], [y_loc[-1], y_loc_end[0]], color='blue')
	plt.plot([x_loc[-1], x_loc_pred[0]], [y_loc[-1], y_loc_pred[0]], color='blue')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Smooth Trajectory')
	plt.grid(True)
	plt.legend()

	# Save the plot if save_dir is provided
	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f'trajectory_plot_{epoch}_sm.png')
		plt.savefig(save_path)
	plt.close()


print("total GPU", torch.cuda.device_count())
parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=16)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="supermarket.pt")
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)

def test(test_dataset, model, best_of_n = 1):

	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	test_loss = 0

	with torch.no_grad():
		_index = np.random.randint(0, 20)
		for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
			traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
			x = traj[:, :hyper_params["past_length"], :]
			y = traj[:, hyper_params["past_length"]:, :]
			y = y.cpu().numpy()
			# reshape the data
			past = x
			x = x.contiguous().view(-1, x.shape[1]*x.shape[2])
			x = x.to(device)

			future = y[:, :-1, :]
			dest = y[:, -1, :]
			all_l2_errors_dest = []
			all_guesses = []
			for index in range(best_of_n):

				dest_recon = model.forward(x, initial_pos, device=device)
				dest_recon = dest_recon.cpu().numpy()
				all_guesses.append(dest_recon)

				l2error_sample = np.linalg.norm(dest_recon - dest, axis = 1)
				all_l2_errors_dest.append(l2error_sample)

			all_l2_errors_dest = np.array(all_l2_errors_dest)
			all_guesses = np.array(all_guesses)
			# average error
			l2error_avg_dest = np.mean(all_l2_errors_dest)

			# choosing the best guess
			indices = np.argmin(all_l2_errors_dest, axis = 0)

			best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]

			# taking the minimum error out of all guess
			l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))

			# back to torch land
			best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

			# using the best guess for interpolation
			interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
			interpolated_future = interpolated_future.cpu().numpy()
			best_guess_dest = best_guess_dest.cpu().numpy()

			# final overall prediction
			predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
			predicted_future = np.reshape(predicted_future, (-1, hyper_params["future_length"], 2))
			if i == _index:
				index=0
				plot_trajectory(past[index].cpu().numpy(), y[index], predicted_future[index], epoch=0, save_dir='./')
				break
		



def main():
	N = args.num_trajectories #number of generated trajectories
	model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
	model = model.double().to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	test_dataset = Supermarket("german_4", hyper_params["past_length"], hyper_params["future_length"], "test")

	# for traj in test_dataset.trajectory_batches:
	# 	traj -= traj[:, :1, :]
	# 	traj *= hyper_params["data_scale"]

	#average ade/fde for k=20 (to account for variance in sampling)
	num_samples = 150
	average_ade, average_fde = 0, 0
	test(test_dataset, model, best_of_n = N)
	# 	average_ade += test_loss
	# 	average_fde += final_point_loss_best

	# print()
	# print("Average ADE:", average_ade/num_samples)
	# print("Average FDE:", average_fde/num_samples)

main()
