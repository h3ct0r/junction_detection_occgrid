#!/usr/bin/env python
""" Junction Manager"""

from copy import copy
import rospy
from threading import Lock
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
import time
import collections


class JunctionManager(object):
    def __init__(self):

        self.lock = Lock()
        self.map_deque = collections.deque(maxlen=1)  # size of maps to join, default to 1 (not joining anything)
        self.latest_joined_map = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Parameters
        self.robot_name = rospy.get_namespace().split("/")[1]
        self.robot_type = self.robot_name[:-1]

        # Subscribers
        self.latest_costmap = None
        self._costmap_sub_topic_name = '/spot1/move_base/global_costmap/costmap_throttle'
        self.costmap_sub = rospy.Subscriber(
            self._costmap_sub_topic_name,
            OccupancyGrid,
            self.handle_costmap_cb
        )

    def run(self):
        """ main entry point """

        rate = rospy.Rate(5)

        while not self.is_initialized():
            rospy.logwarn("Waiting for initialization...")
            rate.sleep()

        while not rospy.is_shutdown():
            self.join_maps()

            if self.latest_joined_map is None:
                rate.sleep()
                continue

            self.junction_detection_opencv()
            rate.sleep()

    def is_initialized(self):
        """ check for initial data needed for this node """

        try:
            rospy.wait_for_message(self._costmap_sub_topic_name, OccupancyGrid, timeout=5)
        except rospy.ROSException as rex:
            rospy.logwarn(rex)
            return False

        return True

    @staticmethod
    def map_to_img(occ_grid):
        """ convert nav_msgs/OccupancyGrid to OpenCV mat 
            small noise in the occ grid is removed by 
            thresholding on the occupancy probability (> 50%)
        """

        data = occ_grid.data
        w = occ_grid.info.width
        h = occ_grid.info.height

        img = np.zeros((h, w, 1), np.uint8)
        img += 255  # start with a white canvas instead of a black one

        # occupied cells (0 - 100 prob range)
        # free cells (0)
        # unknown -1
        for i in range(0, h):
            for j in range(0, w):
                if data[i * w + j] >= 50:
                    img[i, j] = 0
                elif 0 < data[i * w + j] < 50:
                    img[i, j] = 255
                elif data[i * w + j] == -1:
                    img[i, j] = 205

        # crop borders if performing map stitching
        # img = img[20:380, 20:380]
        return img

    @staticmethod
    def remove_isolated_pixels(img, is_debug=False):
        """ remove isolated components 
            using a 8 point conectivity
            small areas that are less than the standard deviation / 4.0
            off all areas are removed TODO: check this heuristic?
        """

        # 8 points conectivity (up, down, left, right and diagonals)
        # 4 points conectivity (only up, down, left, right)
        connectivity = 8
        output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

        num_labels = output[0]
        labels = output[1]
        stats = output[2]

        new_image = img.copy()

        areas = [stats[label, cv2.CC_STAT_AREA] for label in range(num_labels)]
        std_area = np.std(areas) / 4.0

        if is_debug:
            print("areas:", areas, "std:", std_area)

        for label in range(num_labels):
            # remove pixels from smaller connected components
            # smaller than the std dev of the total areas
            area = stats[label, cv2.CC_STAT_AREA]
            if area < std_area:
                new_image[labels == label] = 0

                if is_debug:
                    cv2.imshow("new_image", new_image)
                    time.sleep(0.2)

        return new_image

    @staticmethod
    def get_skeleton_intersection(skeleton):
        """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.
        https://stackoverflow.com/questions/41705405/finding-intersections-of-a-skeletonised-image-in-python-opencv

        Keyword arguments:
        skeleton -- the skeletonised image to detect the intersections of

        Returns:
        List of 2-tuples (x,y) containing the intersection coordinates
        """

        def neighbours(x, y, image):
            """ Return 8-neighbours of image point P1(x,y), in a clockwise order """
            img = image
            x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1;
            return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1],
                    img[x_1][y_1]]

            # A biiiiiig list of valid intersections             2 3 4

        # These are in the format shown to the right         1 C 5
        #                                                    8 7 6
        validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                             [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                             [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                             [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                             [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                             [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                             [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                             [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                             [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                             [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                             [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                             [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                             [1, 0, 1, 1, 0, 1, 0, 0]]

        image = skeleton.copy()
        image = image / 255
        intersections = list()
        for x in range(1, len(image) - 1):
            for y in range(1, len(image[x]) - 1):
                # If we have a white pixel
                if image[x][y] == 1:
                    neigh = neighbours(x, y, image)
                    valid = True
                    if neigh in validIntersection:
                        intersections.append((y, x))

        # Filter intersections to make sure we don't count them twice or ones that are very close together
        for point1 in intersections:
            for point2 in intersections:
                if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10 ** 2) and (point1 != point2):
                    intersections.remove(point2)

        # Remove duplicates
        intersections = list(set(intersections))
        return intersections

    @staticmethod
    def merge_imgs(img1, img2):
        """ merge two map images in a particularly fast way (for python anyways)
            we only consider white and black areas for merging, unkown areas are not
            merged
        """

        from itertools import product

        h = img1.shape[0]
        w = img1.shape[1]

        h2 = img2.shape[0]
        w2 = img2.shape[1]

        for pos in product(range(h), range(w)):
            if pos[0] > h2 - 1 or pos[1] > w2 - 1:
                continue

            pixel = img2.item(pos)
            if pixel != 205:
                img1.itemset(pos, pixel)

        return img1

    @staticmethod
    def count_branches(centroid, skeletonized_img, radius=50, is_debug=False):
        """ Get number of branches given a skeletonized image and a branch point
            we use a radius to detect collisions between the circle and the branches
        """

        n_branches = 0
        h = skeletonized_img.shape[0]
        w = skeletonized_img.shape[1]
        color_debug = cv2.cvtColor(skeletonized_img, cv2.COLOR_GRAY2RGB)

        pts = cv2.ellipse2Poly(
            (int(centroid[0]), int(centroid[1])),
            (radius, radius),
            0, 0, 360, 1)

        if is_debug:
            for p in pts:
                if p[1] > h - 1 or p[0] > w - 1:
                    continue

                color_debug[int(p[1]), int(p[0])] = [0, 0, 255]

        non_zero = False
        for p in pts:
            if p[0] > h - 1 or p[1] > w - 1:
                continue

            pixel = abs(skeletonized_img.item(p[1], p[0]))
            if not non_zero and pixel > 0.0:
                non_zero = True
                n_branches += 1
            elif non_zero and pixel < 1.0:
                non_zero = False

            if is_debug:
                if pixel > 0.0:
                    color_debug[int(p[1]), int(p[0])] = [255, 0, 0]

        if is_debug:
            cv2.imshow('color_debug', color_debug)
            cv2.waitKey(1)

        return n_branches

    def junction_detection_opencv(self):
        """ use opencv to filter the occupancy grid and extract 
            a workable skeleton of the traversable area
            intersections are detected by a library of possible intersections
        """
        costmap_mat = self.latest_joined_map
        cv2.imshow("costmap_mat", costmap_mat)

        _, occ_area = cv2.threshold(costmap_mat, 0, 100, cv2.THRESH_BINARY_INV)
        _, free_area = cv2.threshold(costmap_mat, 250, 255, cv2.THRESH_BINARY)

        # inverted obstacle mask, obstacles are converted from black to white
        black_pixels = np.where(
            (occ_area[:] > 0)
        )
        occ_area[black_pixels] = 255

        # cv2.imshow("occ_area", occ_area)
        # cv2.imshow("free_area", free_area)

        for i in xrange(5):
            occ_area = cv2.medianBlur(occ_area, 7);
        # cv2.imshow("occ_area2", occ_area)

        for i in xrange(5):
            free_area = cv2.medianBlur(free_area, 7);
        # cv2.imshow("free_area2", free_area)

        kernel = np.ones((3, 3), np.uint8)
        dilation_occ = cv2.morphologyEx(occ_area, cv2.MORPH_DILATE, kernel, iterations=10)
        # cv2.imshow("dilation_occ", dilation_occ)

        dilation_free = cv2.dilate(free_area, kernel, iterations=5)
        # cv2.imshow("dilation_free", dilation_free)

        # remove the inflated obstacles to the free navigable area
        diff_im = cv2.subtract(dilation_free, dilation_occ)
        # cv2.imshow("diff_im", diff_im)

        filtered_diff = JunctionManager.remove_isolated_pixels(diff_im, is_debug=False)
        cv2.imshow("filtered_diff", filtered_diff)

        dist_result = cv2.distanceTransform(filtered_diff, distanceType=cv2.DIST_L2,
                                            maskSize=5, dstType=cv2.CV_8U)
        dist_normalized = cv2.normalize(dist_result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U)
        cv2.imshow("dist_normalized", dist_normalized)

        # threshold the normalized distance map to remove outliers and low intensity regions
        _, dist_no_small_branches = cv2.threshold(dist_normalized, 65, 255, cv2.THRESH_BINARY)
        # cv2.imshow("dist_no_small_branches", dist_no_small_branches)

        dist_biggest_component = JunctionManager.remove_isolated_pixels(dist_no_small_branches, is_debug=False)
        # cv2.imshow("dist_biggest_component", dist_biggest_component)

        dist_filtered = cv2.bitwise_and(dist_normalized, dist_biggest_component)
        # cv2.imshow("dist_filtered", dist_filtered)

        thinned_dist = cv2.ximgproc.thinning(dist_biggest_component)
        #cv2.imshow("thinned_dist", thinned_dist)

        roi_img = thinned_dist.copy()
        roi_mask = np.zeros_like(roi_img)
        roi_radius = 140
        roi_mask = cv2.circle(roi_mask,
                              (int(roi_img.shape[0] / 2.0), int(roi_img.shape[1] / 2.0)),
                              roi_radius,
                              (255, 255, 255),
                              -1)
        roi_img = cv2.bitwise_and(roi_img, roi_mask)
        cv2.imshow("roi_img", roi_img)

        # estimate corners and intersections
        corners = JunctionManager.get_skeleton_intersection(roi_img)
        color = cv2.cvtColor(thinned_dist, cv2.COLOR_GRAY2RGB)  # color image for debugging purposes

        # dilate skeleton to improve the detection of branches
        roi_dilated = cv2.dilate(roi_img, np.ones((3, 3), np.uint8), iterations=1)
        #cv2.imshow("roi_dilated", roi_dilated)

        super_branch_found = False
        for idx in range(len(corners)):
            corner = corners[idx]

            # simple heuristic to detect super branches (a true intersection)
            n_branches = JunctionManager.count_branches(corner, roi_dilated)

            if n_branches >= 4.0:
                super_branch_found = True
                branch_label = 'id {}: {} (SUPER)'.format(idx, n_branches)
            else:
                branch_label = 'id {}: {}'.format(idx, n_branches)

            cv2.putText(color, branch_label,
                        (int(corner[0]), int(corner[1])),
                        self.font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            color = cv2.circle(color, (int(corner[0]), int(corner[1])), 3, [0, 0, 255], -1)
            color = cv2.circle(color, (int(corner[0]), int(corner[1])), 60, [0, 0, 255], 1)

        if super_branch_found:
            rospy.loginfo("Super branch found!")

        cv2.imshow("color", color)
        cv2.waitKey(1)

    def handle_costmap_cb(self, msg):
        """ receive the occupancy grid map and register it """
        self.latest_costmap = msg
        self.map_deque.append(self.latest_costmap)

    def join_maps(self, is_debug=False):
        """ join/stich multiple occupancy grid maps using the relative 
            transformation between them given by the map.info.origin location
        """

        local_map = self.map_deque[0]
        resolution = local_map.info.resolution
        last_origin_x = local_map.info.origin.position.x / float(resolution)
        last_origin_y = local_map.info.origin.position.y / float(resolution)

        self.latest_joined_map = self.map_to_img(local_map)

        accumulated_diff_x = 0
        accumulated_diff_y = 0

        for cur_map in list(self.map_deque)[1:]:
            cur_origin_x = cur_map.info.origin.position.x / float(resolution)
            cur_origin_y = cur_map.info.origin.position.y / float(resolution)

            diff_x = cur_origin_x - last_origin_x
            diff_y = cur_origin_y - last_origin_y

            top = 0 if diff_y < 0 else abs(diff_y)
            bottom = 0 if diff_y > 0 else abs(diff_y)
            left = 0 if diff_x > 0 else abs(diff_x)
            right = 0 if diff_x < 0 else abs(diff_x)

            if is_debug:
                print("top:{} bottom:{} left:{} right:{}".format(top, bottom, left, right))

            self.latest_joined_map = cv2.copyMakeBorder(self.latest_joined_map,
                                                        int(round(top)), int(round(bottom)), int(round(left)),
                                                        int(round(right)),
                                                        cv2.BORDER_CONSTANT, value=205)

            top = 0 if accumulated_diff_y < 0 else abs(accumulated_diff_y)
            bottom = 0 if accumulated_diff_y > 0 else abs(accumulated_diff_y)
            left = 0 if accumulated_diff_x > 0 else abs(accumulated_diff_x)
            right = 0 if accumulated_diff_x < 0 else abs(accumulated_diff_x)
            # print("top:{} bottom:{} left:{} right:{}".format(top, bottom, left, right))

            cur_img = self.map_to_img(cur_map)
            cur_img = cv2.copyMakeBorder(cur_img,
                                         int(round(top)), int(round(bottom)), int(round(left)), int(round(right)),
                                         cv2.BORDER_CONSTANT, value=205)

            accumulated_diff_x += diff_x
            accumulated_diff_y += diff_y

            if is_debug:
                print("joined:({}, {}); accumulated:({}, {});".format(diff_x, diff_y, accumulated_diff_x,
                                                                      accumulated_diff_y))

            M = np.float32([
                [1, 0, accumulated_diff_x],
                [0, 1, accumulated_diff_y]
            ])
            cur_img = cv2.warpAffine(cur_img, M, (cur_img.shape[1], cur_img.shape[0]), borderValue=205)
            self.latest_joined_map = JunctionManager.merge_imgs(self.latest_joined_map, cur_img)

            last_origin_x = cur_origin_x
            last_origin_y = cur_origin_y

            if is_debug:
                cv2.imshow("cur_img", cur_img)
                # cv2.imshow("latest_joined_map2", self.latest_joined_map)
                cv2.waitKey(1)


def main():
    rospy.init_node('junction_manager')
    manager = JunctionManager()
    manager.run()


if __name__ == '__main__':
    main()
