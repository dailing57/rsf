import numpy as np
import torch
import torch.optim as optim
import yaml
from vis_o3d import vis_boxes_from_params
import rsf_utils
from rsf_loss import RSFLossv2, RSFLossCycle
from inference import flow_inference
from pytorch3d.structures import Pointclouds, list_to_padded
from pytorch3d.ops import estimate_pointcloud_normals, iterative_closest_point
from pytorch3d import transforms
import argparse
import open3d as o3d


class SF_Optimizer:
    def __init__(self, anchors, config, pc1, pc2, pc1_normals, pc2_normals, R_ego, t_ego, init_perbox=None, init_global=None, use_gt_ego=False, icp_init=False):
        self.anchors = anchors
        self.num_boxes = anchors.shape[0]
        self.config = config
        self.batch_size = len(pc1)

        pc1_opt, pc2_opt = [torch.clone(p) for p in pc1], [torch.clone(p) for p in pc2]
        pc1_normals_opt, pc2_normals_opt = [torch.clone(p) for p in pc1_normals], [torch.clone(p) for p in pc2_normals]

        self.pc1, self.pc2 = Pointclouds(pc1).to(device='cuda'), Pointclouds(pc2).to(device='cuda')
        self.pc1_normals, self.pc2_normals = list_to_padded(pc1_normals).to(device='cuda'), list_to_padded(pc2_normals).to(device='cuda')
        self.pc1_opt, self.pc2_opt = Pointclouds(pc1_opt).to(device='cuda'), Pointclouds(pc2_opt).to(device='cuda')
        self.pc1_normals_opt, self.pc2_normals_opt = list_to_padded(pc1_normals_opt).to(device='cuda'), list_to_padded(pc2_normals_opt).to(device='cuda')

        self.gt_R_ego, self.gt_t_ego = torch.stack(R_ego).transpose(-1, -2).to('cuda'), torch.stack(t_ego).to('cuda')
        self.gt_ego_transform = rsf_utils.get_rigid_transform(self.gt_R_ego, self.gt_t_ego)
        self.predicted_flow, self.segmentation, self.motion_parameters = None, None, None
        self.updated = True

        if init_perbox is None:
            if config['cycle']:
                perbox_params = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.1, 0, 0, .9, 0, 0, 0, 1.1, 0, 0, .9, 0, 0]), (self.batch_size, self.num_boxes, 1))
            else:
                perbox_params = np.tile(np.array([0,0,0,0,0,0,0,0,0,1.1,0,0,.9,0,0]), (self.batch_size, self.num_boxes, 1))
            self.perbox_params = torch.tensor(perbox_params, requires_grad=True, device='cuda', dtype=torch.float32)
        else:
            self.perbox_params = init_perbox
        if init_global is None:
            if use_gt_ego:
                self.global_params = torch.cat([torch.stack(R_ego).transpose(-1, -2).to('cuda').reshape(len(R_ego), -1), torch.stack(t_ego).to('cuda')], dim=-1)
            elif icp_init:
                icp_output = iterative_closest_point(self.pc1_opt, self.pc2_opt)
                R_icp, t_icp, scale_icp = icp_output[3]
                self.global_params = torch.tensor(np.concatenate([R_icp.detach().cpu().numpy().reshape(R_icp.shape[0], -1),
                                        t_icp.detach().cpu().numpy()], axis=-1), requires_grad=True, device='cuda', dtype=torch.float32)
            else:
                self.global_params = torch.tensor([[1.1,0,0,0,1,0,0,0,.9,0,0,0]]*self.batch_size, requires_grad=True, device='cuda', dtype=torch.float32)
        else:
            self.global_params = init_global
        if use_gt_ego:
            self.opt = optim.Adam([self.perbox_params], lr=config['lr'])
        else:
            self.opt = optim.Adam([self.global_params, self.perbox_params], lr = config['lr'])
        if config['cycle']:
            self.loss_function = RSFLossCycle(anchors, config)
        else:
            self.loss_function = RSFLossv2(anchors, config)

    def optimize(self, epochs):
        for j in range(epochs):
            if j % 10 == 0:
                self.viscur()
            self.opt.zero_grad()
            loss = self.loss_function(self.pc1_opt, self.pc2_opt, self.pc1_normals_opt, self.pc2_normals_opt, self.global_params, self.perbox_params)
            if self.config['print_loss']:
                print(loss['total_loss'].item())
            loss['total_loss'].backward()
            self.opt.step()
        self.updated = True

    def predict(self):
        if self.updated:
            output_flow, output_seg, output_params = [], [], []
            with torch.no_grad():
                for vis_idx in range(self.batch_size):
                    predicted_flow, segmentation, motion_parameters = flow_inference(self.pc1.points_list()[vis_idx], self.global_params[vis_idx],
                                                                                    self.perbox_params[vis_idx], self.anchors, self.config, cc=False, cycle=self.config['cycle'])
                    output_flow.append(predicted_flow)
                    output_seg.append(segmentation)
                    output_params.append(motion_parameters)
            self.predicted_flow, self.segmentation, self.motion_parameters = output_flow, output_seg, output_params
            self.updated = False
        return self.predicted_flow, self.segmentation, self.motion_parameters

    def viscur(self):
        pc1_eval, pc2_eval = self.pc1.points_list()[0], self.pc2.points_list()[0]
        ego_transform = rsf_utils.global_params2Rt(self.global_params)
        boxes, box_transform = rsf_utils.perbox_params2boxesRt(self.perbox_params, self.anchors)
        box_transform = transforms.Transform3d(
            matrix=ego_transform.get_matrix().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)
        ego_transform = ego_transform[0]
        boxes = boxes[:self.num_boxes]
        box_transform = box_transform[:self.num_boxes]
        vis_boxes_from_params(pc1_eval, pc2_eval, ego_transform, boxes, box_transform)  

    def visualize(self):
        with torch.no_grad():
            batch_prediction = self.predict()
            for vis_idx, prediction in enumerate(zip(*batch_prediction)):
                pc1_eval, pc2_eval = self.pc1.points_list()[vis_idx], self.pc2.points_list()[vis_idx]
                ego_transform = rsf_utils.global_params2Rt(self.global_params)
                boxes, box_transform = rsf_utils.perbox_params2boxesRt(self.perbox_params, self.anchors)
                box_transform = transforms.Transform3d(
                    matrix=ego_transform.get_matrix().repeat_interleave(self.num_boxes, dim=0)).compose(box_transform)
                ego_transform = ego_transform[vis_idx]
                boxes = boxes[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]
                box_transform = box_transform[vis_idx * self.num_boxes:(vis_idx + 1) * self.num_boxes]

                print('Optimized boxes')
                vis_boxes_from_params(pc1_eval, pc2_eval, ego_transform, boxes, box_transform)
                _, _, pred_motion_params = self.predict()
                print('Inferred boxes')
                vis_boxes_from_params(pc1_eval, pc2_eval, 
                                      pred_motion_params[vis_idx]['ego_transform'], 
                                      pred_motion_params[vis_idx]['boxes'],
                                      pred_motion_params[vis_idx]['box_transform'])



def optimize(cfg):
    hyperparameters = cfg['hyperparameters']

    # anchors
    z_center = -1
    box_scale = hyperparameters['box_scale'] #1.25
    anchor_width = 1.6*box_scale
    anchor_length = 3.9*box_scale
    anchor_height = 1.5*box_scale

    anchor_x = torch.arange(-34, 34, 4, dtype=torch.float32)
    anchor_y = torch.arange(-34, 34, 6, dtype=torch.float32)
    anchors_xy = torch.stack(torch.meshgrid(anchor_x, anchor_y, indexing='ij'), dim=-1)
    offsets = torch.tensor([[0, 3], [0, 0]]).repeat(anchors_xy.shape[0] // 2, 1)
    if anchors_xy.shape[0] % 2 != 0:
        offsets = torch.cat((offsets, torch.tensor([[0, 3]])), dim=0)
    anchors_xy += offsets.unsqueeze(1)
    anchors_xy = anchors_xy.view(-1, 2)
    anchors_xy -= torch.mean(anchors_xy, dim=0, keepdim=True)
    anchors = torch.cat((anchors_xy, torch.stack([torch.tensor([z_center, anchor_width, anchor_length, anchor_height, 0])] * anchors_xy.shape[0], dim=0)), dim=1)
    anchors = anchors.float().to(device='cuda')

    # data
    def str_to_trans(poses):
        return np.vstack((np.fromstring(poses, dtype=float, sep=' ').reshape(3, 4), [0, 0, 0, 1]))
    
    def load_poses(pose_path):
        poses = []
        with open(pose_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                poses.append(str_to_trans(line))
        return np.array(poses)

    def load_calib(calib_path):
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    return str_to_trans(line.replace('Tr:', ''))

    def load_vertex(scan_path):
        current_vertex = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
        current_vertex[:,3] = np.ones(current_vertex.shape[0])
        return current_vertex

    pc_path = cfg['data']['pc_path']
    poses_path = cfg['data']['poses_path']
    calib_path = cfg['data']['calib_path']

    poses = load_poses(poses_path)
    calib = load_calib(calib_path)

    pc1, pc2 = load_vertex(pc_path[0])[:, :3], load_vertex(pc_path[1])[:, :3]
    pcd1, pcd2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd1.points, pcd2.points = o3d.utility.Vector3dVector(pc1), o3d.utility.Vector3dVector(pc2)
    pcd1.estimate_normals(), pcd2.estimate_normals()
    pc1_normals, pc2_normals = np.array(pcd1.normals), np.array(pcd2.normals)

    if cfg['data']['remove_ground']:
        is_not_ground_s = (pc1[:, 2] > -1.4)
        is_not_ground_t = (pc2[:, 2] > -1.4)
        pc1 = pc1[is_not_ground_s]
        pc1_normals = pc1_normals[is_not_ground_s]
        pc2 = pc2[is_not_ground_t]
        pc2_normals = pc2_normals[is_not_ground_t]

    if cfg['data']['filter_normals']:
        horizontal_normals_s = np.abs(pc1_normals[:, -1]) < .85
        horizontal_normals_t = np.abs(pc2_normals[:, -1]) < .85
        pc1 = pc1[horizontal_normals_s]
        pc1_normals = pc1_normals[horizontal_normals_s]
        pc2 = pc2[horizontal_normals_t]
        pc2_normals = pc2_normals[horizontal_normals_t]

    if cfg['data']['only_near_points']:
        is_near_s = (np.amax(np.abs(pc1), axis=1) < 35)
        is_near_t = (np.amax(np.abs(pc2), axis=1) < 35)
        pc1 = pc1[is_near_s]
        pc1_normals = pc1_normals[is_near_s]
        pc2 = pc2[is_near_t]
        pc2_normals = pc2_normals[is_near_t]

    pc1_normals, pc2_normals = torch.from_numpy(pc1_normals).float().unsqueeze(0), torch.from_numpy(pc2_normals).float().unsqueeze(0)
    pc1, pc2 = torch.from_numpy(pc1).float().unsqueeze(0), torch.from_numpy(pc2).float().unsqueeze(0)
    trans = torch.from_numpy(np.linalg.inv(poses[cfg['data']['p1_id']] @ calib) @ poses[cfg['data']['p2_id']] @ calib).float()
    R_ego, t_ego = [trans[:3, :3]], [trans[:3, 3]]

    # optimize
    optimizer = SF_Optimizer(anchors, hyperparameters, pc1, pc2, pc1_normals, pc2_normals, R_ego, t_ego)
    optimizer.optimize(hyperparameters['epochs'])
    optimizer.visualize()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--error_filename', type=str, default='errors_file')
    parser.add_argument('--cfg', type=str, default='configs/njust_cfg.yaml')
    args = parser.parse_args()

    with open(args.cfg) as file:
        cfg = yaml.safe_load(file)

    optimize(cfg)
