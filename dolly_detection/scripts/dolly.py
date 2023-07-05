#!/usr/bin/env python3

import rospy
import tf2_ros
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, TransformStamped
import math
from sklearn.cluster import DBSCAN
import numpy as np
import tf

# Size of dolly
DOLLY_SIZE_X = 1.12895
DOLLY_SIZE_Y = 1.47598

class LegPointCluster:
    def __init__(self):
        self.points = []
    
    def add_point(self, point):
        self.points.append(point)
    
    def get_center_point(self):
        center_point = Point()
        num_points = len(self.points)
        if num_points > 0:
            sum_x = sum_y = 0.0
            for point in self.points:
                sum_x += point.x
                sum_y += point.y
            center_point.x = sum_x / num_points
            center_point.y = sum_y / num_points
        return center_point

def cartesian_conversion(scan_data):
    cartesian_points = []
    angle = scan_data.angle_min
    for range_value in scan_data.ranges:
        if not math.isnan(range_value) and range_value != 0.0 and range_value < scan_data.range_max:
            x = range_value * math.cos(angle)
            y = range_value * math.sin(angle)
            point = Point(x, y, 0.0)
            cartesian_points.append(point)
        angle += scan_data.angle_increment
    return cartesian_points

def cluster_points(points, eps, min_samples):
    # Convert to numpy array
    data = np.array([[point.x, point.y] for point in points])

    # Clustering with DBSCAN algorithm
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    labels = db.labels_

    # Clustering 
    clusters = []
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(num_clusters):
        cluster = LegPointCluster()
        cluster_points = [points[j] for j in range(len(points)) if labels[j] == i]
        for point in cluster_points:
            cluster.add_point(point)
        clusters.append(cluster)

    return clusters

def scan_callback(scan_data):
    cartesian_points = cartesian_conversion(scan_data)

    # Clustering parameters
    eps = 0.1  # Distance
    min_samples = 1  # Minimum samples

    clusters = cluster_points(cartesian_points, eps, min_samples)

    # Finding Clusters
    nearest_cluster = None
    second_nearest_cluster = None
    farthest_cluster = None
    min_distance = float("inf")
    second_min_distance = float("inf")
    max_distance = 0.0

    dolly_center = Point()

    if len(clusters) > 0:
        dolly_center.x = sum([cluster.get_center_point().x for cluster in clusters]) / len(clusters)
        dolly_center.y = sum([cluster.get_center_point().y for cluster in clusters]) / len(clusters)

    for cluster in clusters:
        center_point = cluster.get_center_point()
        distance = math.sqrt(center_point.x ** 2 + center_point.y ** 2)

        if distance < min_distance:
            farthest_cluster = second_nearest_cluster
            second_nearest_cluster = nearest_cluster
            nearest_cluster = cluster
            max_distance = second_min_distance
            second_min_distance = min_distance
            min_distance = distance
        elif distance < second_min_distance:
            farthest_cluster = second_nearest_cluster
            second_nearest_cluster = cluster
            max_distance = second_min_distance
            second_min_distance = distance
        elif distance > max_distance:
            farthest_cluster = cluster
            max_distance = distance

    if nearest_cluster is not None and second_nearest_cluster is not None and farthest_cluster is not None:
        # Get cluster center points
        x1, y1 = nearest_cluster.get_center_point().x, nearest_cluster.get_center_point().y
        x2, y2 = second_nearest_cluster.get_center_point().x, second_nearest_cluster.get_center_point().y
        x3, y3 = farthest_cluster.get_center_point().x, farthest_cluster.get_center_point().y

        # Center of the rectangle
        cx = (x1 + x3) / 2
        cy = (y1 + y3) / 2

        # Dimensions of the rectangle
        w = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        h = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

        # Corners of the rectangle
        x = cx - w / 2
        y = cy - h / 2

        # Dolly's rotation
        yaw = math.atan2(y2 - y1, x2 - x1)

        # TF
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "dolly"
        transform.transform.translation.x = -1 * dolly_center.x
        transform.transform.translation.y = -1 * dolly_center.y
        transform.transform.translation.z = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]

        # Publish TF
        tf_broadcaster = tf2_ros.TransformBroadcaster()
        tf_broadcaster.sendTransform(transform)

        # Print pose
        rospy.loginfo("Dolly Center: (%f, %f)", cx, cy)
        rospy.loginfo("Dolly Yaw: %f", yaw)

def main():
    rospy.init_node('dolly_point_filtering_node')
    rospy.Subscriber('/diffbot/scan', LaserScan, scan_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
