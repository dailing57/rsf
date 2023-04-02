import open3d as o3d
import numpy as np
import rsf_utils
import torch
import matplotlib.pyplot as plt

np.random.seed(57)
color_map = np.random.rand(1000, 3) * 0.75 + 0.25
cmap = plt.colormaps.get_cmap('rainbow')
def get_color(c):
    return cmap(c)

def vis_flow(pc1, pc2=None, sf=None, sf_color = None, color=None, seg=False,
             color_max = None, flow_max = None, savefig = False, filename = 'flow_vis.png', view = None):
    
    # create an empty visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # draw a coordinate system
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
    vis.add_geometry(coord)

    # get the coordinates of the first point cloud
    x1 = pc1[:, 0]  # x position of point
    y1 = pc1[:, 1]  # y position of point
    z1 = pc1[:, 2]  # z position of point
    
    # handle segmentation
    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc1))
    if seg and sf is not None:
        labels = rsf_utils.flow_segmentation(pc1, sf)
        colors = np.zeros((len(pc1), 3))
        for i in range(len(pc1)):
            colors[i] = color_map[int(labels[i])]
        pcd1.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd1)
        vis.run()
        vis.destroy_window()
        return
    
    # handle coloring
    if color is not None:
        colors = []
        for c in color:
            colors.append(np.array(get_color(c)[:3]))
        pcd1.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    else:
        pcd1.paint_uniform_color([1,0,0])
    vis.add_geometry(pcd1)

    # draw the second point cloud
    if pc2 is not None:
        pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc2))
        pcd2.paint_uniform_color([0,0,1])
        vis.add_geometry(pcd2)

    # draw the flow vectors
    if sf is not None:
        flowx = sf[:, 0]
        flowy = sf[:, 1]
        flowz = sf[:, 2]
        if sf_color is not None:
            colors = np.zeros((len(sf_color), 3))
            for i in range(len(sf_color)):
                colors[i] = o3d.utility.color_map[int(sf_color[i])%len(o3d.utility.color_map)]
            colors = o3d.utility.Vector3dVector(colors)
        else:
            colors = None
        pt, st = [], []
        for i in range(len(x1)):
            pt.append([x1[i], y1[i], z1[i]])
            pt.append([x1[i]+flowx[i], y1[i]+flowy[i], z1[i]+flowz[i]])
            st.append([2 * i, 2 * i + 1])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(pt))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(st))
        vis.add_geometry(line_set)
    vis.run()
    vis.destroy_window()

def vis_boxes_from_params(pc1, pc2, ego_transform, boxes, box_transform):
    pc1_ego = ego_transform.transform_points(pc1)
    if boxes is None:
        vis_flow(pc1_ego.detach().cpu().numpy(), pc2.detach().cpu().numpy())
        return
    print('Confidences: ')
    print(boxes[:, 0].detach().cpu().numpy())
    ego_transform_r = ego_transform.stack(*([ego_transform]*(boxes.shape[0]-1)))
    transformed_boxes = rsf_utils.transform_boxes(boxes, ego_transform_r)
    vis_transform = ego_transform_r.inverse().compose(box_transform)
    vis_boxes(pc1_ego.detach().cpu().numpy(), transformed_boxes.detach().cpu().numpy(), vis_transform, pc2.detach().cpu().numpy())

def vis_boxes(points, boxes, transform=None, pc2=None, savefig=False, filename='box_vis.png', view=None, color=None):
    num_boxes = len(boxes)
    params = boxes.shape[1]
    if params != 7 and params != 8:
        print("invalid number of box parameters")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    vis.add_geometry(coord)

    for n, box in enumerate(boxes):
        if params == 7:
            x, y, z, width, length, height, heading = box
        elif params == 8:
            c, x, y, z, width, length, height, heading = box

        center = np.array([x, y, z])
        forward = np.array([-np.sin(heading), np.cos(heading), 0]) * length / 2
        right = np.array([np.cos(heading), np.sin(heading), 0]) * width / 2
        up = np.array([0, 0, height / 2])
        p1 = center+forward+right+up
        p2 = center+forward+right-up
        p3 = center+forward-right+up
        p4 = center+forward-right-up
        p5 = center-forward+right+up
        p6 = center-forward+right-up
        p7 = center-forward-right+up
        p8 = center-forward-right-up
        corners = np.vstack((p1,p2,p3,p4,p5,p6,p7,p8))
        if color is None:
            use_color = (n+2)/(num_boxes+1)
        else:
            use_color = color[n]
        if params == 7:
            draw_box(corners, vis, use_color, use_color)
        elif params == 8:
            draw_box(corners, vis, c, c)

        if transform is not None:
            new_corners = transform[n].transform_points(torch.tensor(corners, device='cuda', dtype=torch.float32)).detach().cpu().numpy()
            draw_box(new_corners, vis, c, c)
    if pc2 is not None:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        pcd2.paint_uniform_color([0,0,1])
        vis.add_geometry(pcd2)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)
    pcd1.paint_uniform_color([1,0,0])
    vis.add_geometry(pcd1)
    vis.run()
    vis.destroy_window()

def draw_box(corners, vis, color, heading_color):
    center = np.mean(corners, axis=0)
    heading_line = np.vstack((center, np.mean(corners[:4], axis=0)))
    heading_line_points = o3d.utility.Vector3dVector(heading_line)
    heading_line_colors = np.array([get_color(heading_color)[:3]])
    heading_line_colors = o3d.utility.Vector3dVector(heading_line_colors)
    heading_line_plot = o3d.geometry.LineSet(points=heading_line_points, lines=o3d.utility.Vector2iVector([[0, 1]]))
    heading_line_plot.colors = heading_line_colors
    vis.add_geometry(heading_line_plot)
    p1, p2, p3, p4, p5, p6, p7, p8 = corners
    box = np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])
    box_colors = np.tile(np.array(get_color(color)[:3]), (12, 1))
    box_colors = o3d.utility.Vector3dVector(box_colors)
    box_points = o3d.utility.Vector3dVector(box)
    box_plot = o3d.geometry.LineSet(points=box_points, lines=o3d.utility.Vector2iVector([[0, 1], [0, 2], [3, 1], [3, 2], [4, 5], [4, 6], [7, 5], [7, 6], [0, 4], [1, 5], [3, 7], [2, 6]]))
    box_plot.colors = box_colors
    vis.add_geometry(box_plot)