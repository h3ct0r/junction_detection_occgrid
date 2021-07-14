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
import cython
from timeit import default_timer as timer
import weave


class JunctionManager(object):
    def __init__(self):

        self.lock = Lock()
        self.registered_stairs_nodes = []
        self.map_deque = collections.deque(maxlen=1)
        self.latest_joined_map = None

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
            rospy.logwarn("Waiting for initialization")
            rate.sleep()

        while not rospy.is_shutdown():
            self.join_maps()

            if self.latest_joined_map is None:
                rate.sleep()
                continue

            # cv2.imshow("latest_joined_map", self.latest_joined_map)
            # cv2.waitKey(1)

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

        # crop borders
        img = img[20:380, 20:380]
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

    def _thinningIteration(self, im, iter):
        I, M = im, np.zeros(im.shape, np.uint8)
        expr = """
        for (int i = 1; i < NI[0]-1; i++) {
            for (int j = 1; j < NI[1]-1; j++) {
                int p2 = I2(i-1, j);
                int p3 = I2(i-1, j+1);
                int p4 = I2(i, j+1);
                int p5 = I2(i+1, j+1);
                int p6 = I2(i+1, j);
                int p7 = I2(i+1, j-1);
                int p8 = I2(i, j-1);
                int p9 = I2(i-1, j-1);
                int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
                if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
                    M2(i,j) = 1;
                }
            }
        } 
        """

        weave.inline(expr, ["I", "iter", "M"])
        return (I & ~M)


    def thinning(self, src):
        dst = src.copy() / 255
        prev = np.zeros(src.shape[:2], np.uint8)
        diff = None

        while True:
            dst = self._thinningIteration(dst, 0)
            dst = self._thinningIteration(dst, 1)
            diff = np.absolute(dst - prev)
            prev = dst.copy()
            if np.sum(diff) == 0:
                break

        return dst * 255

    def neighbours(self, x,y,image):
        """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
        return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]   

    def getSkeletonIntersection(self, skeleton):
        """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

        Keyword arguments:
        skeleton -- the skeletonised image to detect the intersections of

        Returns: 
        List of 2-tuples (x,y) containing the intersection coordinates
        """
        # A biiiiiig list of valid intersections             2 3 4
        # These are in the format shown to the right         1 C 5
        #                                                    8 7 6 
        validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                             [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                             [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                             [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                             [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                             [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                             [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                             [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                             [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                             [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                             [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                             [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                             [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                             [1,0,1,1,0,1,0,0]];
        image = skeleton.copy();
        image = image/255;
        intersections = list();
        for x in range(1,len(image)-1):
            for y in range(1,len(image[x])-1):
                # If we have a white pixel
                if image[x][y] == 1:
                    neigh = self.neighbours(x,y,image);
                    valid = True;
                    if neigh in validIntersection:
                        intersections.append((y,x));
        # Filter intersections to make sure we don't count them twice or ones that are very close together
        for point1 in intersections:
            for point2 in intersections:
                if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                    intersections.remove(point2);
        # Remove duplicates
        intersections = list(set(intersections));
        return intersections;

    def juntion_detection_opencv(self):
        #_latest_costmap = self.latest_costmap
        #costmap_mat = self.map_to_img(_latest_costmap)

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
        cv2.imshow("filtered_diff", filtered_diff)
        
        # skeleton = self.skeletonize(filtered_diff)
        # cv2.imshow("skeleton", skeleton)
        
        # thinned = cv2.ximgproc.thinning(filtered_diff)  
        # cv2.imshow("thinned", thinned)

        #thinned2 = cv2.ximgproc.thinning(filtered_diff, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        #cv2.imshow("thinned2", thinned2)

        dist_result = cv2.distanceTransform(filtered_diff, distanceType=cv2.DIST_L2, 
            maskSize=5, dstType=cv2.CV_8U) 
        max_dist = dist_result.max()
        # print("max_dist:", max_dist)
        # cv2.imshow("dist_result", dist_result)
        
        dist_normalized = cv2.normalize(dist_result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("dist_normalized", dist_normalized)

        _, dist_no_small_branches = cv2.threshold(dist_normalized, 65, 255, cv2.THRESH_BINARY)
        # cv2.imshow("dist_no_small_branches", dist_no_small_branches)

        dist_biggest_component = self.remove_isolated_pixels(dist_no_small_branches, debug=False)
        #cv2.imshow("dist_biggest_component", dist_biggest_component)

        dist_filtered = cv2.bitwise_and(dist_normalized, dist_biggest_component)
        # cv2.imshow("dist_filtered", dist_filtered)

        thinned_dist = cv2.ximgproc.thinning(dist_biggest_component)
        cv2.imshow("thinned_dist", thinned_dist)

        # bw2 = self.thinning(dist_biggest_component)
        # cv2.imshow("bw2", bw2)

        corners = self.getSkeletonIntersection(thinned_dist)
        color = cv2.cvtColor(thinned_dist, cv2.COLOR_GRAY2RGB)
        for idx in range(len(corners)):
            corner = corners[idx]
            #color[int(corner[1]), int(corner[0])] = [0, 0, 255]
            print("corner:", idx)
            #n_branches = check_branches(corner, gray)
            #cv2.putText(color, '{}:{}'.format(idx, n_branches), (int(corner[0]), int(corner[1])), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            color = cv2.ellipse(color, (int(corner[0]), int(corner[1])), (50, 50), 0, 0, 360, [0, 100, 255])
            color = cv2.circle(color, (int(corner[0]), int(corner[1])), 3, [0, 0, 255], -1)
            color = cv2.circle(color, (int(corner[0]), int(corner[1])), 60, [0, 0, 255], 1)

        cv2.imshow("color", color)

        #dst = cv2.cornerHarris(thinned_dist, 5, 5, 0.04)
        #cv2.imshow("latest_joined_map", self.latest_joined_map)
        cv2.waitKey(1)

    def handle_costmap_cb(self, msg):
        self.latest_costmap = msg
        self.map_deque.append(self.latest_costmap)
        print("len(map_deque):", len(self.map_deque))

    def merge_imgs(self, img1, img2):
        from itertools import product

        #start = timer()
        #if img1.shape != img2.shape:
        #    rospy.logwarn("img1.shape ({}) != img2.shape ({})".format(img1.shape, img2.shape))

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
        
        #end = timer()
        #print("merge_imgs {}s".format(end - start))
        return img1

    # @cython.boundscheck(False)
    # cpdef unsigned char[:, :] merge_imgs(self, unsigned char [:, :] img1, unsigned char [:, :] img2):
    #     # set the variable extension types
    #     cdef int w, h

    #     # grab the image dimensions
    #     h = img1.shape[0]
    #     w = img2.shape[1]

    #     # loop over the image
    #     for y in range(0, h):
    #         for x in range(0, w):
    #             # threshold the pixel
    #             if img2[y, x] != 205:
    #                 img1[y, x] = img2[y, x]

    #     # return the thresholded image
    #     return img1

    def join_maps(self):
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

            # top, bottom, left, right
            top = 0 if diff_y < 0 else abs(diff_y)
            bottom = 0 if diff_y > 0 else abs(diff_y)
            left = 0 if diff_x > 0 else abs(diff_x)
            right = 0 if diff_x < 0 else abs(diff_x)
            print("top:{} bottom:{} left:{} right:{}".format(top, bottom, left, right))

            self.latest_joined_map = cv2.copyMakeBorder(self.latest_joined_map, 
                int(round(top)), int(round(bottom)), int(round(left)), int(round(right)),
                cv2.BORDER_CONSTANT, value=205)

            top = 0 if accumulated_diff_y < 0 else abs(accumulated_diff_y)
            bottom = 0 if accumulated_diff_y > 0 else abs(accumulated_diff_y)
            left = 0 if accumulated_diff_x > 0 else abs(accumulated_diff_x)
            right = 0 if accumulated_diff_x < 0 else abs(accumulated_diff_x)
            print("top:{} bottom:{} left:{} right:{}".format(top, bottom, left, right))

            cur_img = self.map_to_img(cur_map)
            cur_img = cv2.copyMakeBorder(cur_img, 
                int(round(top)), int(round(bottom)), int(round(left)), int(round(right)),
                cv2.BORDER_CONSTANT, value=205)

            accumulated_diff_x += diff_x
            accumulated_diff_y += diff_y

            print("joined:({}, {}); accumulated:({}, {});".format(diff_x, diff_y, accumulated_diff_x, accumulated_diff_y))

            M = np.float32([
                [1, 0, accumulated_diff_x],
                [0, 1, accumulated_diff_y]
            ])
            cur_img = cv2.warpAffine(cur_img, M, (cur_img.shape[1], cur_img.shape[0]), borderValue=205)

            #print("img1", self.latest_joined_map.shape, "img2", cur_img.shape)

            cv2.imshow("cur_img", cur_img)
            #cv2.imshow("latest_joined_map2", self.latest_joined_map)
            cv2.waitKey(1)

            self.latest_joined_map = self.merge_imgs(self.latest_joined_map, cur_img)

            #self.latest_joined_map = cv2.addWeighted(self.latest_joined_map, 0.5, cur_img, 0.5, -1)

            last_origin_x = cur_origin_x
            last_origin_y = cur_origin_y

        # add weighted
        #print("\n\n")
        pass

    # def join_maps(self):
    #     local_map = self.map_deque[0]
    #     resolution = local_map.info.resolution
    #     w = local_map.info.width
    #     h = local_map.info.height
    #     last_origin_x = int(local_map.info.origin.position.x / float(resolution))
    #     last_origin_y = int(local_map.info.origin.position.y / float(resolution))

    #     self.latest_joined_map = self.map_to_img(local_map)

    #     accumulated_diff_x = 0
    #     accumulated_diff_y = 0

    #     for m in list(self.map_deque)[1:]:
    #         cur_origin_x = int(m.info.origin.position.x / float(resolution))
    #         cur_origin_y = int(m.info.origin.position.y / float(resolution))

    #         diff_x = cur_origin_x - last_origin_x
    #         diff_y = cur_origin_y - last_origin_y

    #         # top, bottom, left, right
    #         joined_x = 0
    #         if diff_x > 0:
    #             joined_x = diff_x

    #         joined_y = 0
    #         if diff_y > 0:
    #             joined_y = diff_y

    #         accumulated_diff_x += diff_x
    #         accumulated_diff_y += diff_y

    #         self.latest_joined_map = cv2.copyMakeBorder(self.latest_joined_map, 
    #             abs(joined_y), abs(joined_x), abs(joined_y), abs(joined_x),
    #             cv2.BORDER_CONSTANT, value=205)

    #         cur_img = self.map_to_img(m)
    #         cur_img = cv2.copyMakeBorder(cur_img, 
    #             abs(accumulated_diff_y), abs(accumulated_diff_x), abs(accumulated_diff_y), abs(accumulated_diff_x),
    #             cv2.BORDER_CONSTANT, value=205)

    #         print("joined:{},{}; accumulated:{},{};".format(joined_x, joined_x, accumulated_diff_x, accumulated_diff_y))

    #         M = np.float32([
    #             [1, 0, accumulated_diff_x],
    #             [0, 1, accumulated_diff_y]
    #         ])
    #         cur_img = cv2.warpAffine(cur_img, M, (cur_img.shape[1], cur_img.shape[0]), borderValue=205)

    #         #print("img1", self.latest_joined_map.shape, "img2", cur_img.shape)

    #         cv2.imshow("cur_img", cur_img)
    #         #cv2.imshow("latest_joined_map2", self.latest_joined_map)
    #         cv2.waitKey(1)

    #         self.latest_joined_map = self.merge_imgs(self.latest_joined_map, cur_img)

    #         #self.latest_joined_map = cv2.addWeighted(self.latest_joined_map, 0.5, cur_img, 0.5, -1)

    #         last_origin_x = cur_origin_x
    #         last_origin_y = cur_origin_y

    #     # add weighted
    #     #print("\n\n")
    #     pass

    # def join_maps(self):
    #     local_map = self.map_deque[-1]
    #     resolution = local_map.info.resolution
    #     w = local_map.info.width
    #     h = local_map.info.height
    #     last_origin_x = int(local_map.info.origin.position.x / float(resolution))
    #     last_origin_y = int(local_map.info.origin.position.y / float(resolution))
    #     #print(last_origin_x, last_origin_y)

    #     self.latest_joined_map = self.map_to_img(local_map)

    #     accumulated_diff_x = 0
    #     accumulated_diff_y = 0

    #     for m in reversed(list(self.map_deque)[:-1]):
    #         cur_origin_x = int(m.info.origin.position.x / float(resolution))
    #         cur_origin_y = int(m.info.origin.position.y / float(resolution))

    #         diff_x = cur_origin_x - last_origin_x
    #         diff_y = cur_origin_y - last_origin_y
    #         accumulated_diff_x += diff_x
    #         accumulated_diff_y += diff_y

    #         # top, bottom, left, right
    #         self.latest_joined_map = cv2.copyMakeBorder(self.latest_joined_map, 
    #             abs(diff_y), abs(diff_x), abs(diff_y), abs(diff_x),
    #             cv2.BORDER_CONSTANT, value=205)

    #         cur_img = self.map_to_img(m)
    #         cur_img = cv2.copyMakeBorder(cur_img, 
    #             abs(accumulated_diff_y), abs(accumulated_diff_x), abs(accumulated_diff_y), abs(accumulated_diff_x),
    #             cv2.BORDER_CONSTANT, value=205)

    #         print(diff_x, diff_y)

    #         M = np.float32([
    #             [1, 0, diff_x],
    #             [0, 1, diff_y]
    #         ])
    #         cur_img = cv2.warpAffine(cur_img, M, (cur_img.shape[1], cur_img.shape[0]), borderValue=205)

    #         print("img1", self.latest_joined_map.shape, "img2", cur_img.shape)

    #         self.latest_joined_map = cv2.addWeighted(self.latest_joined_map, 0.5, cur_img, 0.5, -1)

    #     # add weighted

    #     print("\n\n")
    #     pass


def main():
    rospy.init_node('junction_manager')
    manager = JunctionManager()
    manager.run()

if __name__ == '__main__':
    main()


# rosrun topic_tools throttle messages /spot1/move_base/global_costmap/costmap 1.0
# rostopic hz /spot1/move_base/global_costmap/costmap