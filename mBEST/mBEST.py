import numpy as np
# import matplotlib.pyplot as plt
import cv2
from time import time
from itertools import combinations
from skimage.draw import line
from sklearn.cluster import DBSCAN
import mBEST.utils as ut
import mBEST.skeletonize as sk


class mBEST:
    def __init__(self, seg_func, epsilon=40, delta=25, colors=None):
        self.seg_func = seg_func
        self.epsilon = epsilon
        self.delta = delta

        self.end_point_kernel = np.array(([1, 1, 1], [1, 10, 1], [1, 1, 1]), dtype=np.uint8)

        self.intersection_clusterer = DBSCAN(eps=self.epsilon, min_samples=1)
        self.adjacent_pixel_clusterer = DBSCAN(eps=3, min_samples=1)

        self.colors = colors
        if colors is None:
            self.colors = np.array(
                [
                    [0, 255, 0], 
                    [0, 0, 255], 
                    [255, 0, 0],
                    [0, 255, 255], 
                    [255, 255, 0], 
                    [255, 0, 255]
                ],
                np.uint8
            ).tolist()

    def _detect_keypoints(self, skeleton_img):
        padded_img = np.zeros((skeleton_img.shape[0]+2, skeleton_img.shape[1]+2), dtype=np.uint8)
        padded_img[1:-1, 1:-1] = skeleton_img
        res = cv2.filter2D(src=padded_img.astype(np.uint8), ddepth=-1, kernel=self.end_point_kernel)
        ends = np.argwhere(res == 11) - 1
        intersections = np.argwhere(res > 12) - 1
        return ends, intersections

    def _prune_split_ends(self, skeleton, ends, intersections):
        my_skeleton = skeleton.copy()
        inter_indices = [True]*len(intersections)#[True for _ in intersections]
        inter_index_dict = {"{},{}".format(row, col): i for i, (row, col) in enumerate(intersections)}
        valid_ends = []
        ends = list(ends)  # so that we can append new ends

        for i, e in enumerate(ends):
            curr_pixel = e.copy()
            path = [curr_pixel]
            prune = False
            found_nothing = False

            
            while True:
                row,col = curr_pixel
                my_skeleton[row, col] = 0
                neighbors = np.argwhere(my_skeleton[row-1:row+2,col-1:col+2])
                # Keep following segment
                if neighbors.shape[0] == 1:
                    curr_pixel += neighbors.flatten()-1
                    path.append(curr_pixel)
                # We've reached an end
                elif neighbors.shape[0] == 0:
                    found_nothing = True
                    break
                # Found an intersection
                else:
                    # Remove intersection pixels from list
                    ind = "{},{}".format(curr_pixel[0], curr_pixel[1])
                    inter_indices[inter_index_dict[ind]] = False

                    for j in range(-2, 3):
                        for k in range(-2, 3):
                            ind = "{},{}".format(j+curr_pixel[0], k+curr_pixel[1])
                            if ind in inter_index_dict:
                                inter_indices[inter_index_dict[ind]] = False

                    prune = True
                    break

                # This is most likely a valid segment
                if len(path) > self.delta:
                    break

            if found_nothing: 
                continue
            path = np.asarray(path)

            # Prune noisy segment from skeleton.
            if prune:
                skeleton[path] = 0
                row, col = path[-1]
                skeleton[row-2:row+3, col-2:col+3] = 0
                vecs = ut.get_boundary_pixels(skeleton[row-4:row+5, col-4:col+5])

                # Reconnect the segments together after pruning a branch
                if len(vecs) == 2:
                    vecs[:, 0] += row - 3
                    vecs[:, 1] += col - 3
                    reconnected_line = line(vecs[0][0], vecs[0][1], vecs[1][0], vecs[1][1])
                    skeleton[reconnected_line[0], reconnected_line[1]] = 1

                # We created a new end so add it.
                elif len(vecs) == 1:
                    vecs = vecs.squeeze()
                    vecs[0] += row - 3
                    vecs[1] += col - 3
                    ends.append(vecs)

                # Reflect the changes on our skeleton copy.
                my_skeleton[row-2:row+3, col-2:col+3] = skeleton[row-2:row+3, col-2:col+3]

            else:
                valid_ends.append(i)

        ends = np.asarray(ends)
        return ends[valid_ends], intersections[inter_indices]

    def _cluster_intersections(self, intersections):
        # Clustering intersections consists of two phases
        # 1st phase: cluster adjacent intersection pixels into clusters
        # 2nd phase: cluster intersection clusters together
        temp_intersections = []
        new_intersections = []

        # 1st phase
        self.adjacent_pixel_clusterer.fit(intersections)
        for i in np.unique(self.adjacent_pixel_clusterer.labels_):
            temp_intersections.append(
                np.round(np.mean(intersections[self.adjacent_pixel_clusterer.labels_ == i], axis=0)).astype(np.uint16))

        temp_intersections = np.asarray(temp_intersections)

        # 2nd phase
        self.intersection_clusterer.fit(temp_intersections)
        for i in np.unique(self.intersection_clusterer.labels_):
            new_intersections.append(
                np.round(np.mean(temp_intersections[self.intersection_clusterer.labels_ == i], axis=0)).astype(np.uint16))

        return np.asarray(new_intersections, dtype=np.int16)

    @staticmethod
    def _compute_minimal_bending_energy_paths(ends, inter):
        indices = [i for i in range(len(ends))]
        all_pairs = list(combinations(indices, 2))

        # This should preferably be the case for every intersection, but could not be because of noise.
        if len(ends) == 4:
            possible_path_pairs = set()
            already_added = set()
            n = sum(indices)
            for c1 in all_pairs:
                if c1 in already_added: continue
                for c2 in all_pairs:
                    if c1 == c2 or sum(c1) + sum(c2) != n: continue
                    already_added.add(c1)
                    already_added.add(c2)
                    possible_path_pairs.add((c1, c2))
                    continue

            minimum_total_curvature = np.inf
            for (a1, a2), (b1, b2) in possible_path_pairs:
                total_curvature = ut.compute_cumulative_curvature(ends[a1], ends[a2],
                                                                  ends[b1], ends[b2], inter)

                if total_curvature < minimum_total_curvature:
                    minimum_total_curvature = total_curvature
                    indices[a1] = a2
                    indices[a2] = a1
                    indices[b1] = b2
                    indices[b2] = b1

        elif len(ends) == 3:
            minimum_curvature = np.inf
            total = 3  # 0 + 1 + 2
            for v1, v2 in all_pairs:
                curvature = ut.compute_curvature(ends[v1], ends[v2], inter)

                if curvature < minimum_curvature:
                    minimum_curvature = curvature
                    indices[v1] = v2
                    indices[v2] = v1

                    # Make the last path end here
                    third_path = total - (v1 + v2)
                    indices[third_path] = None

        else:
            return False

        return indices

    def _generate_intersection_paths(self, skeleton, intersections,image):
        paths_to_ends = {}
        crossing_orders = {}

        for inter in intersections:
            x, y = inter
            best_paths = False
            k_size = int(self.epsilon * 0.4)
            three_way = False

            # Compute the best paths through the intersection, i.e. the one that minimizes total bending energy.
            while best_paths is False:
                skeleton[x-k_size:x+k_size+1, y-k_size:y+k_size+1] = 0
                ends = ut.get_boundary_pixels(skeleton[x-k_size-2:x+k_size+3, y-k_size-2:y+k_size+3])
                ends = ends.reshape((-1, 2))

                ends[:, 0] += x-k_size-1
                ends[:, 1] += y-k_size-1

                best_paths = self._compute_minimal_bending_energy_paths(ends, inter)
                k_size += 5

            generated_paths = [list(np.asarray(line(e[0], e[1], inter[0], inter[1])).T[:-1]) for e in ends]

            for i, (row1, col1) in enumerate(ends):
                if best_paths[i] is None:
                    three_way = True
                    continue
                row2, col2 = ends[best_paths[i]]
                # Construct a path that minimizes the total bending energy of the intersection.
                if i < best_paths[i]:
                    constructed_path = generated_paths[i] + [inter] + generated_paths[best_paths[i]][::-1]# + [[row2, col2]]
                    constructed_path = np.asarray(constructed_path, dtype=np.int16)
                # If we already constructed the reverse path, just flip and reuse.
                else:
                    constructed_path = np.flip(paths_to_ends["{},{}".format(row2, col2)], axis=0)
                    # constructed_path[:-1] = constructed_path[1:]
                    # constructed_path[-1] = [row2, col2]
                paths_to_ends["{},{}".format(row1, col1)] = constructed_path

            if three_way: continue

            # Determine crossing order
            possible_paths = [1, 2, 3]
            possible_paths.remove(best_paths[0])
            row11, col11 = ends[0]
            row12, col12 = ends[best_paths[0]]
            row21, col21 = ends[possible_paths[0]]
            row22, col22 = ends[possible_paths[1]]
            id11 = "{},{}".format(row11, col11)
            id12 = "{},{}".format(row12, col12)
            id21 = "{},{}".format(row21, col21)
            id22 = "{},{}".format(row22, col22)
            p1 = paths_to_ends[id11]
            p2 = paths_to_ends[id21]

            # Using blurred image is key to getting rid of influence from glare
            blurred_image = cv2.blur(image, (5, 5))
            std1 = blurred_image[p1].std(axis=0).sum()
            std2 = blurred_image[p2].std(axis=0).sum()

            if std1 > std2:
                crossing_orders[id11] = 0
                crossing_orders[id12] = 0
                crossing_orders[id21] = 1
                crossing_orders[id22] = 1
            else:
                crossing_orders[id11] = 1
                crossing_orders[id12] = 1
                crossing_orders[id21] = 0
                crossing_orders[id22] = 0

        return paths_to_ends, crossing_orders

    @staticmethod
    def _generate_paths(skeleton, ends, intersection_paths):
        ends = list(ends.astype(np.int16))
        paths = []
        path_id = 0
        intersection_path_id = {}

        while len(ends) != 0:
            curr_pixel = ends.pop()
            done = False
            path = [curr_pixel]

            visited = set()

            while not done:
                path += list(ut.traverse_skeleton(skeleton, curr_pixel))[1:]
                p_x, p_y = path[-1]
                id = "{},{}".format(p_x, p_y)

                # We found an intersection, let's add our precomputed path to it.
                if id in intersection_paths:
                    if id in visited:  # found a cycle
                        # paths.append(np.asarray(path) - 1)  # -1 for offset
                        paths.append(np.asarray(path[1:]))  # -1 for offset
                        break
                    visited.add(id)
                    path += list(intersection_paths[id][1:,:])
                    curr_pixel = np.array([path[-1][0], path[-1][1]], dtype=np.int16)
                    intersection_path_id[id] = path_id
                    continue
                # We've finished this path.
                else:
                    # paths.append(np.asarray(path)-1)  # -1 for offset
                    paths.append(np.asarray(path))  # -1 for offset
                    # Remove the end so that we don't traverse again in opposite direction.
                    ut.remove_from_array(ends, path[-1])
                    break

            path_id += 1

        return paths, intersection_path_id

    @staticmethod
    def _compute_radii(mask, paths):
        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        path_radii = np.round(dist_img).astype(np.int)
        path_radii_avgs = [int(np.round(dist_img[path[:, 0], path[:, 1]].mean())) for path in paths]
        return path_radii, path_radii_avgs

    def _plot_paths(self, image, paths, is_over, path_radii_data, intersection_color=None):
        path_img = np.zeros_like(image)

        path_radii, path_radii_avgs = path_radii_data
        
        end_lengths = self.epsilon
        end_buffer = 10 if end_lengths > 10 else int(end_lengths * 0.5)
        img_height, img_width = image.shape[1], image.shape[0]
        left_limit = end_lengths
        right_limit = img_width - int(end_lengths * 0.5)
        bottom_limit = end_lengths
        top_limit = img_height - int(end_lengths * 0.5)

        # Generate segmentation along the DLO path(s)
        for i, path in enumerate(paths):
            for row,col in path[:end_buffer]:
                cv2.circle(path_img, (col,row), path_radii[row,col], self.colors[i], -1)
            for row,col in path[end_buffer:end_lengths]:
                if col < left_limit or col > right_limit or row < bottom_limit or row > top_limit:
                    cv2.circle(path_img, (col,row), path_radii[row,col], self.colors[i], -1)
                else:
                    cv2.circle(path_img, (col,row), path_radii_avgs[i], self.colors[i], -1)
            for row,col in path[end_lengths:-end_lengths]:
                cv2.circle(path_img, (col,row), path_radii_avgs[i], self.colors[i], -1)
            for row,col in path[-end_lengths:-end_buffer]:
                if col < left_limit or col > right_limit or row < bottom_limit or row > top_limit:
                    cv2.circle(path_img, (col,row), path_radii[row,col], self.colors[i], -1)
                else:
                    cv2.circle(path_img, (col,row), path_radii_avgs[i], self.colors[i], -1)
            for row,col in path[-end_buffer:]:
                cv2.circle(path_img, (col,row), path_radii[row,col], self.colors[i], -1)

            for row,col in path[is_over[i],:]:
                color = self.colors[i] if intersection_color is None else intersection_color
                cv2.circle(path_img, (col-1, row-1), path_radii_avgs[i], color, -1)


        # Handle intersections with appropriate crossing order
        # 1==over, 0==under
        # for id, path_id in intersection_path_id.items():
        #     if id not in crossing_orders or crossing_orders[id] == 1: continue
        #     color = self.colors[path_id]
        #     for row,col in intersection_paths[id]:
        #         cv2.circle(path_img, (col-1, row-1), path_radii_avgs[path_id], (color), -1)
        # for id, path_id in intersection_path_id.items():
        #     if id not in crossing_orders or crossing_orders[id] == 0: continue
        #     color = self.colors[path_id] if intersection_color is None else intersection_color
        #     for row,col in intersection_paths[id]:
        #         cv2.circle(path_img, (col-1, row-1), path_radii_avgs[path_id], color, -1)


        return path_img

    def compute3d(self,paths,inter_paths,inter_path_id,crossing_orders):
        inter_paths_list = []
        _inter_paths = inter_paths.copy()
        for i in range(len(paths)):
            new_dict = {key : _inter_paths[key] for key in _inter_paths if inter_path_id.get(key,-1) == i and crossing_orders.get(key,False)}
            # for key in new_dict:
            #     del _inter_paths[key]
            inter_paths_list.append(new_dict)

        overs = []
        for i in range(len(paths)):
            over = np.zeros(paths[i].shape[0],dtype=bool)
            for key,value in inter_paths_list[i].items():
                idx_start = np.nonzero(np.all(paths[i] == value[0,:],axis=1))[0][0]
                # indices = np.argwhere((paths[i][:,None,[0,1]] == value[:,[0,1]]).all(-1))[:,0]
                over[idx_start:idx_start+value.shape[0]] = True
            overs.append(over)
        return overs




    def run(self, image, intersection_color=None, plot=False, save_fig=False, save_id=0):

        # Create the mask
        orig_mask = self.seg_func(image)

        # Create the skeleton pixels.
        img = np.zeros((orig_mask.shape[0]+2, orig_mask.shape[1]+2), dtype=np.uint8)
        img[1:-1, 1:-1] = orig_mask
        mask = img == 0
        img[mask] = 0
        img[~mask] = 1
        skeleton = sk.skeletonize(img)

        # Keypoint Detection
        ends, intersections = self._detect_keypoints(skeleton)

        # Prune noisy split ends.
        ends, intersections = self._prune_split_ends(skeleton, ends, intersections)

        intersection_paths = {}
        crossing_orders = {}
        if len(intersections > 0):
            intersections = self._cluster_intersections(intersections)

            intersection_paths, crossing_orders = self._generate_intersection_paths(skeleton, intersections,image)

        paths, intersection_path_id = self._generate_paths(skeleton, ends, intersection_paths)

        is_over = self.compute3d(paths,intersection_paths,intersection_path_id,crossing_orders)

        if plot > -1:
            path_radii = self._compute_radii(orig_mask, paths)
            path_img = self._plot_paths(image,paths, is_over, path_radii, intersection_color)
            # dual = np.zeros(image.shape * np.array([1,2,1]),dtype=np.uint8)
            # dual[:,:image.shape[1],:] = image
            # dual[:,image.shape[1]:,:] = path_img
            dual = np.hstack([image,path_img])
            # dual = cv2.cvtColor(dual,cv2.COLOR_RGB2BGR)
            cv2.imshow("results",dual)
            cv2.waitKey(plot)

        paths3d = []
        for path,over in zip(paths,is_over):
            # adding 3rd dimension and also flipping first two dims so they are pixel coords not matrix coords.
            paths3d.append(np.insert(path[:,::-1],2,over.astype(np.uint32),axis=1).T) 
        return paths3d
