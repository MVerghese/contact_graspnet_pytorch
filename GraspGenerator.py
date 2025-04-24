import glob
import os
import torch
import numpy as np
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils

from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image
from contact_graspnet_pytorch.checkpoints import CheckpointIO
# from utils import *
class Generator:
	def __init__(self,checkpoint_path="/home/mverghese/RobotControl/contact_graspnet_pytorch/checkpoints/contact_graspnet"):
		global_config = config_utils.load_config(checkpoint_path, batch_size=1, arg_configs=[])
		# Build the model
		self.grasp_estimator = GraspEstimator(global_config)

		# Load the weights
		model_checkpoint_dir = os.path.join(checkpoint_path, 'checkpoints')
		checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=self.grasp_estimator.model)
		try:
			load_dict = checkpoint_io.load('model.pt')
		except FileExistsError:
			print('No model checkpoint found')
			load_dict = {}
	
	def inference(self, pc):
		pred_grasps_cam, scores, contact_pts, gripper_openings = self.grasp_estimator.predict_scene_grasps(pc, 
																					   pc_segments={}, 
																					   local_regions=False, 
																					   filter_grasps=False, 
																					   forward_passes=3)
		# print(np.shape(pred_grasps_cam[-1]))
		pred_grasps_cam = pred_grasps_cam[-1]
		# print(np.shape(pred_grasps_cam))
		for i in range(len(pred_grasps_cam)):
			pred_grasps_cam[i][:3,1] = np.cross(pred_grasps_cam[i][:3,2],pred_grasps_cam[i][:3,0])
			


		return pred_grasps_cam, scores[-1], contact_pts[-1], gripper_openings[-1]

	def masked_inference(self,full_pc, object_pc):
		grasps, scores, contact_pts, gripper_openings = self.inference(full_pc)
		print(np.shape(grasps))
		print(np.shape(contact_pts))
		print(np.shape(gripper_openings))
		# Filter grasps based on the object point cloud
		for i in range(len(grasps)):
			if np.all(np.linalg.norm(object_pc - contact_pts[i,:], axis=1) > 0.02):
				scores[i] = 0.
		return grasps, scores, contact_pts, gripper_openings
		
		
	
	def visualize(self, pc, pred_grasps_cam, scores, pc_colors=None):
		pred_grasps_cam = {-1: pred_grasps_cam}
		scores = {-1: scores}

		visualize_grasps(pc, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
	

def main():
	generator = Generator()
	pc = np.load("point_cloud.npy")
	pc_colors = np.load("colors.npy")
	pred_grasps_cam, scores, contact_pts, gripper_openings = generator.inference(pc)
	for grasp in pred_grasps_cam:
		R = grasp[:3,:3]
		# R[:,1] = np.cross(R[:,0],R[:,2])
		print("Magniudes: ",np.linalg.norm(R[:,0]), np.linalg.norm(R[:,1]), np.linalg.norm(R[:,2]))
		print("Cross product diff", np.linalg.norm(np.cross(R[:,0],R[:,2]) - R[:,1]))
		print("Dot Products", np.dot(R[:,0],R[:,1]), np.dot(R[:,0],R[:,2]), np.dot(R[:,1],R[:,2]))
	generator.visualize(pc, pred_grasps_cam, scores, pc_colors)
	print(gripper_openings)

if __name__ == "__main__":
	main()