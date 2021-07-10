#!/usr/bin/env python
""" Junction Manager"""

from copy import copy
import rospy
from threading import Lock
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
import time


class JunctionManager(object):
    def __init__(self):

        self.lock = Lock()
        self.registered_stairs_nodes = []

        # Parameters
        self.robot_name = rospy.get_namespace().split("/")[1]
        self.robot_type = self.robot_name[:-1]
        
        # Subscribers
        self.latest_costmap = None
        self._costmap_sub_topic_name = '/spot1/move_base/global_costmap/costmap'
        self.costmap_sub = rospy.Subscriber(
            self._costmap_sub_topic_name,
            OccupancyGrid,
            self.handle_costmap_cb
        )

    def run(self):
        """ main entry point """

        rate = rospy.Rate(5)

        while not self.is_initialized():
            rospy.logwarn("Waiting for initialization")
            rate.sleep()

        while not rospy.is_shutdown():
            self.juntion_detection_opencv()
            rate.sleep()

    def is_initialized(self):
        """ check for initial data needed for this node """

        try:
    	   rospy.wait_for_message(self._costmap_sub_topic_name, OccupancyGrid, timeout=5)
        except rospy.ROSException as rex:
            return False

        return True

    def map_to_img(self, occ_grid):
        """ convert nav_msgs/OccupancyGrid to OpenCV mat """

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

        return img

    def remove_isolated_pixels(self, img, debug=False):
        """ remove isolated components """
        connectivity = 8  # 8 points conectivity (up, down, left, right and diagonals)

        output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

        num_labels = output[0]
        labels = output[1]
        stats = output[2]

        new_image = img.copy()
 
        areas = [stats[label, cv2.CC_STAT_AREA] for label in range(num_labels)]
        std_area = np.std(areas) / 4.0

        if debug:
            print("areas:", areas, "std:", std_area)

        for label in range(num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            # remove pixels from smaller connected components 
            # smaller than the std dev of the total areas
            if stats[label, cv2.CC_STAT_AREA] < std_area:
                new_image[labels == label] = 0
                if debug:
                    cv2.imshow("new_image", new_image)
                    time.sleep(0.2)

        return new_image

    def skeletonize(self, img):
        """ OpenCV function to return a skeletonized version of img, a Mat object """
        # hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
        # https://stackoverflow.com/questions/42845747/optimized-skeleton-function-for-opencv-with-python

        img = img.copy() # don't clobber original
        skel = img.copy()

        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]

            if cv2.countNonZero(img) == 0:
                break

        return skel

    def juntion_detection_opencv(self):
        _latest_costmap = self.latest_costmap

        costmap_mat = self.map_to_img(_latest_costmap)

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
        #cv2.imshow("occ_area2", occ_area)

        for i in xrange(5):
            free_area = cv2.medianBlur(free_area, 7);
        #cv2.imshow("free_area2", free_area)

        kernel = np.ones((3, 3), np.uint8)
        dilation_occ = cv2.morphologyEx(occ_area, cv2.MORPH_DILATE, kernel, iterations=10)
        #cv2.imshow("dilation_occ", dilation_occ)

        dilation_free = cv2.dilate(free_area, kernel, iterations=5)
        # cv2.imshow("dilation_free", dilation_free)

        diff_im = cv2.subtract(dilation_free, dilation_occ)
        #cv2.imshow("diff_im", diff_im)

        filtered_diff = self.remove_isolated_pixels(diff_im, debug=False)
        #cv2.imshow("filtered_diff", filtered_diff)
        
        # skeleton = self.skeletonize(filtered_diff)
        # cv2.imshow("skeleton", skeleton)
        
        # thinned = cv2.ximgproc.thinning(filtered_diff)  
        # cv2.imshow("thinned", thinned)

        #thinned2 = cv2.ximgproc.thinning(filtered_diff, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        #cv2.imshow("thinned2", thinned2)

        dist_result = cv2.distanceTransform(filtered_diff, distanceType=cv2.DIST_L2, maskSize=5, dstType=cv2.CV_8U)
        #cv2.imshow("dist_result", dist_result)
        
        dist_normalized = cv2.normalize(dist_result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #cv2.imshow("dist_normalized", dist_normalized)

        _, dist_no_small_branches = cv2.threshold(dist_normalized, 60, 255, cv2.THRESH_BINARY)
        #cv2.imshow("dist_no_small_branches", dist_no_small_branches)

        dist_biggest_component = self.remove_isolated_pixels(dist_no_small_branches, debug=True)
        #cv2.imshow("dist_biggest_component", dist_biggest_component)

        # dist_filtered = cv2.bitwise_and(dist_normalized, dist_biggest_component)
        # cv2.imshow("dist_filtered", dist_filtered)

        thinned_dist = cv2.ximgproc.thinning(dist_biggest_component)
        cv2.imshow("thinned_dist", thinned_dist)
        
        cv2.waitKey(1)

    def handle_costmap_cb(self, msg):
        self.latest_costmap = msg


def main():
    rospy.init_node('junction_manager')
    manager = JunctionManager()
    manager.run()

if __name__ == '__main__':
    main()
