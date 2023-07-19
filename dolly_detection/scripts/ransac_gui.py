#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSlider, QLabel, QWidget
from PyQt5.QtCore import Qt

import numpy as np
from sklearn import linear_model

import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class RANSAC_subscriber():
    def __init__(self):
        rospy.init_node("ransac_surface_detection", anonymous=True)
        self.subscription = rospy.Subscriber('/diffbot/scan', LaserScan, self.RANSAC)
        self.marker_publisher = rospy.Publisher("line_markers", Marker, queue_size=1)
        self.rate = 50
        rospy.Rate(self.rate)

        self.max_range = 0.5
        self.min_range = 0.003
        self.min_samples = 18
        self.residual_threshold = 0.044
        self.max_fails = 2
        self.max_cluster_dist = 0.041
        self.stop_n_inliers = 36

    def RANSAC(self, msg):
        variables = [
       "max_range",
       "min_range",
       "min_samples",
       "residual_threshold",
       "max_fails",
       "max_cluster_dist",
       "stop_n_inliers"
        ]

        for variable in variables:
            value = getattr(self, variable)
            print(f"{variable}: {value}")

        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_inc = msg.angle_increment
        ranges = np.array(msg.ranges)

        if len(ranges) == 0:
            print("No Points from LaserScan")
        angle_arr = np.arange(angle_min, angle_max+(0.1*angle_inc), angle_inc)

        def y_dist(angle, dist):
            return np.sin(angle) * dist

        def x_dist(angle, dist):
            return np.cos(angle) * dist

        positions = np.array([np.array([x_dist(a, d), y_dist(a, d)]) for (a, d) in zip(angle_arr, ranges)])
        positions = positions[np.linalg.norm(positions, axis=1) < self.max_range]
        positions = positions[np.linalg.norm(positions, axis=1) > self.min_range] # Sort wheel points away

        if len(positions) == 0:
            print("No Points Detected")

        # Split the dataset
        clusters = []
        cluster_start = 0
        for i in range(positions.shape[0]):
            if i == positions.shape[0] - 1:
                clusters.append(positions[cluster_start:])
                break
            if np.linalg.norm(np.absolute(positions[i]) - np.absolute(positions[i + 1])) > self.max_cluster_dist:
                clusters.append(positions[cluster_start:i])
                cluster_start = i
        if clusters == []:
            clusters = np.array([positions])
        else:
            clusters = np.array(clusters, dtype=object)

        # RANSAC
        fit_sets = []
        fit_models = []
        for points in clusters:
            while np.array(points).shape[0] > self.min_samples:
                fails = 0
                try:
                    rs = linear_model.RANSACRegressor(min_samples=self.min_samples,
                                                      residual_threshold=self.residual_threshold,
                                                      stop_n_inliers=self.stop_n_inliers)
                    rs.fit(np.expand_dims(points[:, 0], axis=1), points[:, 1])
                    inlier_mask = rs.inlier_mask_
                    inlier_points = points[np.array(inlier_mask)]
                    min_x = np.min(inlier_points[:, 0], axis=0)
                    max_x = np.max(inlier_points[:, 0], axis=0)
                    start = np.array([min_x, rs.predict([[min_x]])[0]])
                    end = np.array([max_x, rs.predict([[max_x]])[0]])
                    fit_sets.append(inlier_points)
                    fit_models.append(np.array([start, end]))
                    points = points[~np.array(inlier_mask)]
                except:
                    fails += 1
                    if fails >= self.max_fails:
                        break

        def nearest_point_on_line(line_start, line_end, point=np.array((0, 0))):
            line_start -= point
            line_end -= point
            a_to_p = -line_start
            a_to_b = line_end - line_start
            sq_mag_a_to_b = a_to_b[0] ** 2 + a_to_b[1] ** 2
            if sq_mag_a_to_b == 0:
                return np.array([0, 0])
            dot_product = a_to_p[0] * a_to_b[0] + a_to_p[1] * a_to_b[1]
            dist_a_to_c = dot_product / sq_mag_a_to_b
            c = np.array([start[0] + a_to_b[0] * dist_a_to_c, start[1] + a_to_b[1] * dist_a_to_c])
            return c + point

        def is_point_between(s, e, p):
            if (np.linalg.norm(e - p) > np.linalg.norm(e - s) or np.linalg.norm(p - s) > np.linalg.norm(e - s)):
                return False
            else:
                return True

        min_dist = np.inf
        min_dist_point = np.array([0, 0])
        for model in fit_models:
            # Find nearest point on the line, relative to the robot
            point = nearest_point_on_line(model[0], model[1])

            # Get the distance to the point
            dist = np.sqrt(point[0] ** 2 + point[1] ** 2)

            # Check if the point is on the line segment
            if not is_point_between(model[0], model[-1], point):
                dist_start = np.sqrt(model[0, 0] ** 2 + model[0, 1] ** 2)
                dist_end = np.sqrt(model[1, 0] ** 2 + model[1, 1] ** 2)
                dist = np.min([dist_end, dist_start])

            # If the distance is smaller, update the minimum distance and corresponding point
            if dist < min_dist:
                min_dist = dist
                min_dist_point = point

        def angle_to_point(point):
            if point[0] == 0:
                if point[1] > 0:
                    return np.pi / 2
                else:
                    if point[1] < 0:
                        return -np.pi / 2
                    else:
                        return 0
            if point[0] < 0:
                a = np.pi + np.arctan(point[1] / point[0])
                if a > np.pi:
                    a = -2 * np.pi + a
                return a
            return np.arctan(point[1] / point[0])

        # Publish line markers for visualization in RViz
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01
        marker.color.r = 1.0
        marker.color.a = 1.0

        for model in fit_models:
            start = Point()
            start.x = model[0][0] * -1
            start.y = model[0][1] * -1
            start.z = 0.0
            end = Point()
            end.x = model[1][0] * -1
            end.y = model[1][1] * -1
            end.z = 0.0

            marker.points.append(start)
            marker.points.append(end)

        self.marker_publisher.publish(marker)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RANSAC Configuration")
        self.setGeometry(100, 100, 300, 200)

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        max_range_label = QLabel("Max Range")
        min_range_label = QLabel("Min Range")
        min_samples_label = QLabel("Min Inliers")
        residual_threshold_label = QLabel("Residual Threshold")
        max_fails_label = QLabel("Max Fails")
        max_cluster_dist_label = QLabel("Max Cluster Dist")
        stop_n_inliers_label = QLabel("Stop N Inliers")

        self.max_range_slider = QSlider(Qt.Horizontal)
        self.min_range_slider = QSlider(Qt.Horizontal)
        self.min_samples_slider = QSlider(Qt.Horizontal)
        self.residual_threshold_slider = QSlider(Qt.Horizontal)
        self.max_fails_slider = QSlider(Qt.Horizontal)
        self.max_cluster_dist_slider = QSlider(Qt.Horizontal)
        self.stop_n_inliers_slider = QSlider(Qt.Horizontal)

        self.max_range_slider.setRange(0, 1000)
        self.min_range_slider.setRange(0, 1000)
        self.min_samples_slider.setRange(0, 1000)
        self.residual_threshold_slider.setRange(0, 1000)
        self.max_fails_slider.setRange(0, 1000)
        self.max_cluster_dist_slider.setRange(0, 1000)
        self.stop_n_inliers_slider.setRange(0, 1000)

        self.max_range_slider.setValue(50)
        self.min_range_slider.setValue(20)
        self.min_samples_slider.setValue(10)
        self.residual_threshold_slider.setValue(1)
        self.max_fails_slider.setValue(2)
        self.max_cluster_dist_slider.setValue(2)
        self.stop_n_inliers_slider.setValue(50)

        self.max_range_slider.valueChanged.connect(self.update_max_range)
        self.min_range_slider.valueChanged.connect(self.update_min_range)
        self.min_samples_slider.valueChanged.connect(self.update_min_samples)
        self.residual_threshold_slider.valueChanged.connect(self.update_residual_threshold)
        self.max_fails_slider.valueChanged.connect(self.update_max_fails)
        self.max_cluster_dist_slider.valueChanged.connect(self.update_max_cluster_dist)
        self.stop_n_inliers_slider.valueChanged.connect(self.update_stop_n_inliers)

        layout.addWidget(max_range_label)
        layout.addWidget(self.max_range_slider)
        layout.addWidget(min_range_label)
        layout.addWidget(self.min_range_slider)
        layout.addWidget(min_samples_label)
        layout.addWidget(self.min_samples_slider)
        layout.addWidget(residual_threshold_label)
        layout.addWidget(self.residual_threshold_slider)
        layout.addWidget(max_fails_label)
        layout.addWidget(self.max_fails_slider)
        layout.addWidget(max_cluster_dist_label)
        layout.addWidget(self.max_cluster_dist_slider)
        layout.addWidget(stop_n_inliers_label)
        layout.addWidget(self.stop_n_inliers_slider)

        self.ransac = RANSAC_subscriber()

    def update_max_range(self, value):
        self.ransac.max_range = value / 1000.0

    def update_min_range(self, value):
        self.ransac.min_range = value / 1000.0

    def update_min_samples(self, value):
        self.ransac.min_samples = value

    def update_residual_threshold(self, value):
        self.ransac.residual_threshold = value / 1000.0

    def update_max_fails(self, value):
        self.ransac.max_fails = value

    def update_max_cluster_dist(self, value):
        self.ransac.max_cluster_dist = value / 1000.0

    def update_stop_n_inliers(self, value):
        self.ransac.stop_n_inliers = value

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
