#!/usr/bin/env python3

import rospy
import tf2_ros
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, TransformStamped
import math
from sklearn.cluster import DBSCAN, KMeans
import numpy as np
import tf

# Size of dolly
DOLLY_SIZE_X = 1.12895
DOLLY_SIZE_Y = 1.47598
DOLLY_SIZE_HYPOTENUSE = (DOLLY_SIZE_X ** 2 + DOLLY_SIZE_Y ** 2) ** 0.5

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

# Convert laserscan data to cartesian data
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

# Calculates distance between two points, used in filtering
def calculate_distance(cluster1, cluster2):
    x1, y1 = cluster1.get_center_point().x, cluster1.get_center_point().y
    x2, y2 = cluster2.get_center_point().x, cluster2.get_center_point().y
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def cluster_points(points, eps, min_samples):
    # Convert to numpy array
    data = np.array([[point.x, point.y] for point in points])

    # Clustering with DBSCAN algorithm
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    labels = db.labels_

    # Clustering 
    unfiltered_clusters = []
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(num_clusters):
        cluster = LegPointCluster()
        cluster_points = [points[j] for j in range(len(points)) if labels[j] == i]
        for point in cluster_points:
            cluster.add_point(point)
        if len(cluster_points) <= 5:  # If there are no more than 5 instances in the cluster
            unfiltered_clusters.append(cluster)

   # Clustering filter -> Must have at least 3 clusters at 1.92 distance and clusters closer than 5 meters
    clusters = []
    for cluster in unfiltered_clusters:
        x1, y1 = cluster.get_center_point().x, cluster.get_center_point().y
        near_clusters = []
        for other_cluster in unfiltered_clusters:
            if other_cluster != cluster:
                x2, y2 = other_cluster.get_center_point().x, other_cluster.get_center_point().y
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance <= 1.92:
                    near_clusters.append(other_cluster)
        if len(near_clusters) >= 2 and cluster.get_center_point().x**2 + cluster.get_center_point().y**2 <= 25:  
            clusters.append(cluster)

    # Must other clusters at given sizes
    filtered_clusters = []

    dimension_offset = 0.2
    dimension_ranges = [(DOLLY_SIZE_X, dimension_offset), (DOLLY_SIZE_Y, dimension_offset), (DOLLY_SIZE_HYPOTENUSE, dimension_offset)]

    for cluster in clusters:
        x1, y1 = cluster.get_center_point().x, cluster.get_center_point().y
        valid_distance_count = 0

        for other_cluster in clusters:
            if other_cluster != cluster:
                x2, y2 = other_cluster.get_center_point().x, other_cluster.get_center_point().y
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if any((dim - offset <= distance <= dim + offset) for dim, offset in dimension_ranges):
                    valid_distance_count += 1

        if valid_distance_count >= 3:
            filtered_clusters.append(cluster)

    return filtered_clusters


def scan_callback(scan_data):
    cartesian_points = cartesian_conversion(scan_data)

    # DBSCAN Clustering hyperparameters
    eps = 0.4  # Distance (m)
    min_samples = 1  # Minimum samples

    filtered_clusters = cluster_points(cartesian_points, eps, min_samples)
    clusters = filtered_clusters

    # Check if the number of clusters is divisible by 4
    num_clusters = len(clusters)
    if num_clusters % 4 != 0:
        rospy.logwarn("Number of clusters is not divisible by 4.") #FixMe
        return

    dolly_count = num_clusters // 4
    
    # Apply k-means algorithm to group clusters
    kmeans_data = np.array([[cluster.get_center_point().x, cluster.get_center_point().y] for cluster in clusters])
    kmeans = KMeans(n_clusters=dolly_count, random_state=0).fit(kmeans_data)
    # labels = kmeans.labels_.tolist()

    sorted_clusters = [[] for _ in range(dolly_count)]
    for j in range(dolly_count*4):
        sorted_clusters[kmeans.labels_[j]].append(clusters[j])
    
    dolly_transforms = []
    cluster_transforms = []
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    for i in range(dolly_count):

        dolly_center = Point()

        dolly_center.x = kmeans.cluster_centers_[i][0] * -1
        dolly_center.y = kmeans.cluster_centers_[i][1] * -1

        nearest_cluster = None
        second_nearest_cluster = None
        min_distance = float("inf")
        second_min_distance = float("inf")
        max_distance = 0.0

        distance = math.sqrt(kmeans.cluster_centers_[i][0] ** 2 + kmeans.cluster_centers_[i][1] ** 2)

        for j in range(3): #FixMe
            
            if distance < min_distance:
                second_nearest_cluster = nearest_cluster
                nearest_cluster = sorted_clusters[i][j]
                max_distance = second_min_distance
                second_min_distance = min_distance
                min_distance = distance
            elif distance < second_min_distance:
                second_nearest_cluster = sorted_clusters[i][j]
                max_distance = second_min_distance
                second_min_distance = distance
            elif distance > max_distance:
                max_distance = distance
            
        # Center points of clusters
        x1, y1 = nearest_cluster.get_center_point().x, nearest_cluster.get_center_point().y
        x2, y2 = second_nearest_cluster.get_center_point().x, second_nearest_cluster.get_center_point().y
        
        dolly_yaw = math.atan2(y2 - y1, x2 - x1)

        # Dolly TF
        dolly_transform = TransformStamped()
        dolly_transform.header.stamp = rospy.Time.now()
        dolly_transform.header.frame_id = "base_link"
        dolly_transform.child_frame_id = f"dolly_{i}"
        dolly_transform.transform.translation.x = dolly_center.x 
        dolly_transform.transform.translation.y = dolly_center.y   
        dolly_transform.transform.translation.z = 0.0
        quaternion = tf.transformations.quaternion_from_euler(0, 0, dolly_yaw)
        dolly_transform.transform.rotation.x = quaternion[0]
        dolly_transform.transform.rotation.y = quaternion[1]
        dolly_transform.transform.rotation.z = quaternion[2]
        dolly_transform.transform.rotation.w = quaternion[3]
        dolly_transforms.append(dolly_transform)
        
        rospy.loginfo(f"Center of dolly{i} ({dolly_center.x}, {dolly_center.y})")
        
        #Cluster TF
        for j, cluster in enumerate(sorted_clusters[i]):
            cluster_center = cluster.get_center_point()
            cluster_transform = TransformStamped()
            cluster_transform.header.stamp = rospy.Time.now()
            cluster_transform.header.frame_id = "base_link"
            cluster_transform.child_frame_id = f"cluster_{i*4+j}"
            cluster_transform.transform.translation.x = cluster_center.x * -1 
            cluster_transform.transform.translation.y = cluster_center.y * -1
            cluster_transform.transform.translation.z = 0.0
            cluster_transform.transform.rotation.x = 0.0
            cluster_transform.transform.rotation.y = 0.0
            cluster_transform.transform.rotation.z = 0.0
            cluster_transform.transform.rotation.w = 1.0
            cluster_transforms.append(cluster_transform)

    tf_broadcaster.sendTransform(dolly_transforms)
    tf_broadcaster.sendTransform(cluster_transforms)
    rospy.loginfo("Number of Dolly Groups: %d", dolly_count)

def main():
    rospy.init_node('dolly_pose_estimation')
    rospy.Subscriber('/diffbot/scan', LaserScan, scan_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
