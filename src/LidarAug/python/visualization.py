import numpy as np
import open3d
import matplotlib.pyplot as plt
from . import utils


class PCViewer:

    def __init__(self,
                 point_cloud=None,
                 gt_boxes=None,
                 pred_boxes=None,
                 point_features=4,
                 normalize_intensities=False,
                 calib=None):
        if isinstance(point_cloud, str):
            self.point_cloud = utils.load_point_cloud_from_file(
                point_cloud,
                point_features=point_features,
                normalize_intensities=normalize_intensities)
        else:
            self.point_cloud = point_cloud

        if isinstance(gt_boxes, str) and calib is not None:
            self.set_gt_boxes_from_label(label_file=gt_boxes, calib_file=calib)
        else:
            self.gt_boxes = gt_boxes

        self.pred_boxes = pred_boxes

    def set_point_cloud(self, point_cloud):
        self.point_cloud = point_cloud

    def set_gt_boxes(self, gt_boxes):
        self.gt_boxes = gt_boxes

    def set_pred_boxes(self, pred_boxes, pred_scores=None, min_score=0.5):
        to_delete = []
        if pred_scores is not None:
            for i, score in enumerate(pred_scores):
                if score < min_score:
                    to_delete.append(i)
        self.pred_boxes = np.delete(pred_boxes, to_delete, axis=0)

    def set_gt_boxes_from_label(self, label_file, calib_file):
        if isinstance(calib_file, str):
            calib = utils.Calibration(calib_file)
        else:
            calib = calib_file
        self.gt_boxes = utils.gt_boxes_from_label(label_file, calib)

    def draw(self, normalize_intensities=False):
        points = self.point_cloud[:, :3]
        intensities = 1 - (self.point_cloud[:, 3:].reshape(-1) /
                           (255 if normalize_intensities else 1))
        cm = plt.get_cmap('jet')
        colors = np.array(cm(intensities))[:, :3]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(colors)
        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([20 / 255, 20 / 255, 20 / 255])
        opt.point_size = 0.5

        lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7],
                 [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

        if self.gt_boxes is not None:
            corner_boxes = utils.boxes_to_corners_3d(self.gt_boxes)
            colors = [[0, 1, 0] for _ in range(len(lines))]
            for gt_box, corner_box in zip(self.gt_boxes, corner_boxes):
                if gt_box[0] < -900:
                    continue
                line_set = open3d.geometry.LineSet()
                line_set.points = open3d.utility.Vector3dVector(corner_box)
                line_set.lines = open3d.utility.Vector2iVector(lines)
                line_set.colors = open3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)

        if self.pred_boxes is not None:
            corner_boxes = self.pred_boxes  #utils.boxes_to_corners_3d(self.pred_boxes) # todo remove
            colors = [[1, 0, 0] for _ in range(len(lines))]
            #colors[1] = [0, 0, 1]
            for corner_box in corner_boxes:
                line_set = open3d.geometry.LineSet()
                line_set.points = open3d.utility.Vector3dVector(corner_box)
                line_set.lines = open3d.utility.Vector2iVector(lines)
                line_set.colors = open3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)

        def reset_view(vis):
            ctr = vis.get_view_control()
            ctr.set_front(np.array([-1, 0, 0.4]))
            ctr.set_up(np.array([0, 0, 1]))
            ctr.set_lookat(np.array([0, 0, 0]))
            ctr.set_zoom(0.05)
            return False

        def rotate_view_r(vis):
            ctr = vis.get_view_control()
            ctr.rotate(30, 0)
            ctr.set_up(np.array([0, 0, 1]))
            return False

        def rotate_view_l(vis):
            ctr = vis.get_view_control()
            ctr.rotate(-30, 0)
            ctr.set_up(np.array([0, 0, 1]))
            return False

        def translate_view_up(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0, 0, 2)
            ctr.set_up(np.array([0, 0, 1]))
            return False

        def translate_view_down(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0, 0, -2)
            ctr.set_up(np.array([0, 0, 1]))
            return False

        reset_view(vis)
        vis.register_key_callback(ord("R"), reset_view)
        vis.register_key_callback(ord("A"), rotate_view_r)
        vis.register_key_callback(ord("D"), rotate_view_l)
        vis.register_key_callback(ord("W"), translate_view_up)
        vis.register_key_callback(ord("S"), translate_view_down)

        vis.run()
        vis.destroy_window()
