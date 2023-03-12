import re
import os
import csv
import copy
import shutil
import subprocess
import numpy as np
import cv2 as cv
import networkx
import random
from networkx.algorithms.components.connected import connected_components


class ServerConfig:
    def __init__(self, low_res, high_res, low_qp, high_qp, bsize,
                 h_thres, l_thres, max_obj_size, min_obj_size,
                 tracker_length, boundary, intersection_threshold,
                 tracking_threshold, suppression_threshold, simulation,
                 rpn_enlarge_ratio, prune_score, objfilter_iou, size_obj):
        self.low_resolution = low_res
        self.high_resolution = high_res
        self.low_qp = low_qp
        self.high_qp = high_qp
        self.batch_size = bsize
        self.high_threshold = h_thres
        self.low_threshold = l_thres
        self.max_object_size = max_obj_size
        self.min_object_size = min_obj_size
        self.tracker_length = tracker_length
        self.boundary = boundary
        self.intersection_threshold = intersection_threshold
        self.simulation = simulation
        self.tracking_threshold = tracking_threshold
        self.suppression_threshold = suppression_threshold
        self.rpn_enlarge_ratio = rpn_enlarge_ratio
        self.prune_score = prune_score
        self.objfilter_iou = objfilter_iou
        self.size_obj = size_obj
 

class Region:
    def __init__(self, fid, x, y, w, h, conf, label, resolution, #fid, 0, 0, 1, 1, 1.0, 2
                 origin="generic"):
        self.fid = int(fid)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)
        self.label = label
        self.resolution = float(resolution)
        self.origin = origin

    @staticmethod
    def convert_from_server_response(r, res, phase):
        return Region(r[0], r[1], r[2], r[3], r[4], r[5], r[6], res, phase)

    def __str__(self):
        string_rep = (f"{self.fid}, {self.x:0.3f}, {self.y:0.3f}, "
                      f"{self.w:0.3f}, {self.h:0.3f}, {self.conf:0.3f}, "
                      f"{self.label}, {self.origin}")
        return string_rep

    # Judge whether two regions are same or not
    def is_same(self, region_to_check, threshold=0.5):
        # If the fids or labels are different
        # then not the same
        if (self.fid != region_to_check.fid or
                ((self.label != "-1" and region_to_check.label != "-1") and
                 (self.label != region_to_check.label))):
            return False

        # If the intersection to union area
        # ratio is greater than the threshold
        # then the regions are the same
        if calc_iou(self, region_to_check) > threshold:
            return True
        else:
            return False

    def enlarge(self, ratio):
        x_min = max(self.x - self.w * ratio, 0.0)
        y_min = max(self.y - self.h * ratio, 0.0)
        x_max = min(self.x + self.w * (1 + ratio), 1.0)
        y_max = min(self.y + self.h * (1 + ratio), 1.0)
        self.x = x_min
        self.y = y_min
        self.w = x_max - x_min
        self.h = y_max - y_min

    def copy(self):
        return Region(self.fid, self.x, self.y, self.w, self.h, self.conf,
                      self.label, self.resolution, self.origin)


class Results:
    def __init__(self):
        self.regions = []
        self.regions_dict = {}

    def __len__(self):
        return len(self.regions)

    def results_high_len(self, threshold):
        count = 0
        for r in self.regions:
            if r.conf > threshold:
                count += 1
        return count

    def is_dup(self, result_to_add, threshold=0.5):
        # return the regions with IOU greater than threshold
        # and maximum confidence
        if result_to_add.fid not in self.regions_dict:
            return None

        max_conf = -1
        max_conf_result = None
        for existing_result in self.regions_dict[result_to_add.fid]:
            if existing_result.is_same(result_to_add, threshold):
                if existing_result.conf > max_conf:
                    max_conf = existing_result.conf
                    max_conf_result = existing_result
        return max_conf_result

    def combine_results(self, additional_results, threshold=0.5):
        for result_to_add in additional_results.regions:
            self.add_single_result(result_to_add, threshold)

    def add_single_result(self, region_to_add, threshold=0.5):
        if threshold == 1:
            self.append(region_to_add)
            return
        dup_region = self.is_dup(region_to_add, threshold)
        if (not dup_region or
                ("tracking" in region_to_add.origin and
                 "tracking" in dup_region.origin)):
            self.regions.append(region_to_add)
            if region_to_add.fid not in self.regions_dict:
                self.regions_dict[region_to_add.fid] = []
            self.regions_dict[region_to_add.fid].append(region_to_add)
        else:
            final_object = None
            if dup_region.origin == region_to_add.origin:
                final_object = max([region_to_add, dup_region],
                                   key=lambda r: r.conf)
            elif ("low" in dup_region.origin and
                  "high" in region_to_add.origin):
                final_object = region_to_add
            elif ("high" in dup_region.origin and
                  "low" in region_to_add.origin):
                final_object = dup_region
            dup_region.x = final_object.x
            dup_region.y = final_object.y
            dup_region.w = final_object.w
            dup_region.h = final_object.h
            dup_region.conf = final_object.conf
            dup_region.origin = final_object.origin

    def suppress(self, threshold=0.5):
        new_regions_list = []
        while len(self.regions) > 0:
            max_conf_obj = max(self.regions, key=lambda e: e.conf)
            new_regions_list.append(max_conf_obj)
            self.remove(max_conf_obj)
            objs_to_remove = []
            for r in self.regions:
                if r.fid != max_conf_obj.fid:
                    continue
                if calc_iou(r, max_conf_obj) > threshold:
                    objs_to_remove.append(r)
            for r in objs_to_remove:
                self.remove(r)
        new_regions_list.sort(key=lambda e: e.fid)
        for r in new_regions_list:
            self.append(r)

    def append(self, region_to_add):
        # update region (a list)
        self.regions.append(region_to_add)
        # update regions_dict (a dictionary)
        if region_to_add.fid not in self.regions_dict:
            self.regions_dict[region_to_add.fid] = []
        self.regions_dict[region_to_add.fid].append(region_to_add)

    def remove(self, region_to_remove):
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)
        self.regions.remove(region_to_remove)
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)

    def fill_gaps(self, number_of_frames):
        if len(self.regions) == 0:
            return
        results_to_add = Results()
        max_resolution = max([e.resolution for e in self.regions])
        fids_in_results = [e.fid for e in self.regions]
        # print(f"fids_in_results: {fids_in_results}")
        for i in range(number_of_frames):
            if i not in fids_in_results:
                # print(f"frame-{i} not in fid, add now")
                results_to_add.regions.append(Region(i, 0, 0, 0, 0,
                                                     0.1, "no obj",
                                                     max_resolution))
        self.combine_results(results_to_add)
        self.regions.sort(key=lambda r: r.fid)


    def write_results_txt(self, fname, start_time):
        results_file = open(fname, "a")
        for region in self.regions:
            # prepare the string to write
            str_to_write = (f"{region.fid+30*start_time},{region.x},{region.y},"
                            f"{region.w},{region.h},"
                            f"{region.label},{region.conf},"
                            f"{region.resolution},{region.origin}\n")
            results_file.write(str_to_write)
        results_file.close()

    def write_results_csv(self, fname, start_time):
        results_files = open(fname, "a")
        csv_writer = csv.writer(results_files)
        for region in self.regions:
            row = [region.fid+30*start_time, region.x, region.y,
                   region.w, region.h,
                   region.label, region.conf,
                   region.resolution, region.origin]
            csv_writer.writerow(row)
        results_files.close()

    def write(self, fname, start_time):
        if re.match(r"\w+[.]csv\Z", fname):
            self.write_results_csv(fname,start_time)
        else:
            self.write_results_txt(fname,start_time)


def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def filter_bbox_group(bb1, bb2, iou_threshold):
    if calc_iou(bb1, bb2) > iou_threshold and bb1.label == bb2.label:
        return True
    else:
        return False


def overlap(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1.x, bb2.x)
    y_top = max(bb1.y, bb2.y)
    x_right = min(bb1.x+bb1.w, bb2.x+bb2.w)
    y_bottom = min(bb1.y+bb1.h, bb2.y+bb2.h)

    # no overlap
    if x_right < x_left or y_bottom < y_top:
        return False
    else:
        return True


def pairwise_overlap_indexing_list(single_result_frame, iou_threshold):
    pointwise = [[i] for i in range(len(single_result_frame))]
    pairwise = [[i, j] for i, x in enumerate(single_result_frame)
                for j, y in enumerate(single_result_frame)
                if i != j if filter_bbox_group(x, y, iou_threshold)]
    return pointwise + pairwise


def simple_merge(single_result_frame, index_to_merge):
    # directly using the largest box
    bbox_large = []
    for i in index_to_merge:
        i2np = np.array([j for j in i])
        left = min(np.array(single_result_frame)[i2np], key=lambda x: x.x)
        top = min(np.array(single_result_frame)[i2np], key=lambda x: x.y)
        right = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.x + x.w)
        bottom = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.y + x.h)

        fid, x, y, w, h, conf, label, resolution, origin = (
            left.fid, left.x, top.y, right.x + right.w - left.x,
            bottom.y + bottom.h - top.y, left.conf, left.label,
            left.resolution, left.origin)
        single_merged_region = Region(fid, x, y, w, h, conf,
                                      label, resolution, origin)
        bbox_large.append(single_merged_region)
    return bbox_large


def merge_boxes_in_results(results_dict, min_conf_threshold, iou_threshold):
    final_results = Results()

    # Clean dict to remove min_conf_threshold
        # final_results.regions_dict = {fid1: [region1, region2, ...], fid2: [region1, region2, ...], ...}
    for _, regions in results_dict.items():
        to_remove = []
        for r in regions:
            if r.conf < min_conf_threshold:
                to_remove.append(r)
        for r in to_remove:
            regions.remove(r)

    for fid, regions in results_dict.items():
        overlap_pairwise_list = pairwise_overlap_indexing_list(
            regions, iou_threshold)
        overlap_graph = to_graph(overlap_pairwise_list)
        grouped_bbox_idx = [c for c in sorted(
            connected_components(overlap_graph), key=len, reverse=True)]
        merged_regions = simple_merge(regions, grouped_bbox_idx)
        for r in merged_regions:
            final_results.append(r)
    return final_results


def read_results_csv_dict(fname):
    """Return a dictionary with fid mapped to an array
    that contains all Regions objects"""
    results_dict = {}

    rows = []
    with open(fname) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rows.append(row)

    for row in rows:
        fid = int(row[0])
        x, y, w, h = [float(e) for e in row[1:5]]
        conf = float(row[6])
        label = row[5]
        resolution = float(row[7])
        origin = float(row[8])

        region = Region(fid, x, y, w, h, conf, label, resolution, origin)

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(region)

    return results_dict


def read_results_txt_dict(fname):
    """Return a dictionary with fid mapped to
       and array that contains all SingleResult objects
       from that particular frame"""
    results_dict = {}

    with open(fname, "r") as f:
        lines = f.readlines()
        f.close()

    for line in lines:
        line = line.split(",")
        fid = int(line[0])
        x, y, w, h = [float(e) for e in line[1:5]]
        label = line[5]
        conf = float(line[6])
        resolution = float(line[7])
        origin = "generic"
        if len(line) > 8:
            origin = line[8].strip()
        single_result = Region(fid, x, y, w, h, conf, label,
                               resolution, origin.rstrip())

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(single_result)

    return results_dict


def read_results_dict(fname):
    # TODO: Need to implement a CSV function
    if re.match(r"\w+[.]csv\Z", fname):
        return read_results_csv_dict(fname)
    else:
        return read_results_txt_dict(fname)


def read_partial_results_csv_dict(fname, start_fid, length):
    """Return a dictionary with fid mapped to an array
    that contains all Regions objects"""
    results_dict = {}
    max_fid = 0
    end_fid = start_fid + length
    print(f"start:{start_fid} end:{end_fid}")

    rows = []
    with open(fname) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rows.append(row)

    for row in rows:
        fid = int(row[0])

        # check whether fid in specified range: [start_id, end_id)
        if (fid < start_fid) and (fid >= end_fid):
            continue

        # reset fid to (0~length)
        fid = fid % length
        if fid > max_fid:
            max_fid = fid

        x, y, w, h = [float(e) for e in row[1:5]]
        conf = float(row[6])
        label = row[5]
        resolution = float(row[7])
        origin = float(row[8])

        region = Region(fid, x, y, w, h, conf, label, resolution, origin)

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(region)

    return results_dict, max_fid+1


def read_partial_results_txt_dict(fname, start_fid, length):
    """Return a dictionary with fid mapped to
       and array that contains all SingleResult objects
       from that particular frame"""
    results_dict = {}
    end_fid = start_fid + length # length = 60
    fid_list = []
    # print(f"start-fid={start_fid}; end-fid={end_fid}")
    max_fid = 0

    with open(fname, "r") as f:
        lines = f.readlines()
        f.close()

    count = 0
    for line in lines:
        line = line.split(",")
        fid = int(line[0])

        # check whether fid in specified range: [start_id, end_id)
        if (fid < start_fid) or (fid >= end_fid):
            continue

        if fid not in fid_list:
            fid_list.append(fid)

        # reset fid to (0~length)
        fid = fid % length
        if fid > max_fid:
            max_fid = fid
        
        x, y, w, h = [float(e) for e in line[1:5]]
        label = line[5]
        conf = float(line[6])
        resolution = float(line[7])
        origin = "generic"
        if len(line) > 8:
            origin = line[8].strip()
        single_result = Region(fid, x, y, w, h, conf, label,
                               resolution, origin.rstrip())

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(single_result)
            count += 1
    
    # print(f"---------ADDED {count}results")
    print("Fids are:")
    print(fid_list)


    return results_dict, max_fid+1


def read_partial_results_dict(fname, start_fid, length):
    # TODO: Need to implement a CSV function
    if re.match(r"\w+[.]csv\Z", fname):
        return read_partial_results_csv_dict(fname, start_fid, length)
    else:
        return read_partial_results_txt_dict(fname, start_fid, length)


# Calculating IOU
def calc_intersection_area(a, b):
    # (x, y) is TOP LEFT point
    to = max(a.y, b.y)
    le = max(a.x, b.x)
    bo = min(a.y + a.h, b.y + b.h)
    ri = min(a.x + a.w, b.x + b.w)

    w = max(0, ri - le)
    h = max(0, bo - to)

    return w * h


def calc_area(a):
    w = max(0, a.w)
    h = max(0, a.h)

    return w * h


def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area
    return intersection_area / union_area


def get_interval_area(width, all_yes):
    area = 0
    for y1, y2 in all_yes:
        area += (y2 - y1) * width
    return area


def insert_range_y(all_yes, y1, y2):
    ranges_length = len(all_yes)
    idx = 0
    while idx < ranges_length:
        if not (y1 > all_yes[idx][1] or all_yes[idx][0] > y2):
            # Overlapping
            y1 = min(y1, all_yes[idx][0])
            y2 = max(y2, all_yes[idx][1])
            del all_yes[idx]
            ranges_length = len(all_yes)
        else:
            idx += 1

    all_yes.append((y1, y2))


def get_y_ranges(regions, j, x1, x2):
    all_yes = []
    while j < len(regions):
        if (x1 < (regions[j].x + regions[j].w) and
                x2 > regions[j].x):
            y1 = regions[j].y
            y2 = regions[j].y + regions[j].h
            insert_range_y(all_yes, y1, y2)
        j += 1
    return all_yes

 
def compute_area_of_frame(regions): # regions: a list of regions in a SINGLE frame
    
    regions.sort(key=lambda r: r.x + r.w)

    all_xes = []
    for r in regions:
        all_xes.append(r.x)
        all_xes.append(r.x + r.w)
    all_xes.sort()

    area = 0
    j = 0
    for i in range(len(all_xes) - 1):
        x1 = all_xes[i]
        x2 = all_xes[i + 1]

        if x1 < x2:
            while (regions[j].x + regions[j].w) < x1:
                j += 1
            all_yes = get_y_ranges(regions, j, x1, x2)
            area += get_interval_area(x2 - x1, all_yes)

    return area


def compute_area_of_regions(results):
    if len(results.regions) == 0:
        return 0

    min_frame = min([r.fid for r in results.regions])
    max_frame = max([r.fid for r in results.regions])

    # compute the area of a frame by summing up all the regions in this frame
    # then summing up areas of all frames to get total area
    total_area = 0
    for fid in range(min_frame, max_frame + 1):
        regions_for_frame = [r for r in results.regions if r.fid == fid]
        total_area += compute_area_of_frame(regions_for_frame)

    return total_area


# [ENCODING]: compress the images sequence into a video and return size of encoded video 
def compress_and_get_size(images_path, start_id, end_id, qp, fps,
                          enforce_iframes=False, resolution=None):
    number_of_frames = end_id - start_id
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    if resolution and enforce_iframes:
        scale = f"scale=trunc(iw*{resolution}/2)*2:trunc(ih*{resolution}/2)*2"
        if not qp: # qp is not specified
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%010d.png",
                                              "-vcodec", "libx264", 
                                              "-g", "15",
                                              "-keyint_min", "15",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
        else: # qp is specified
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%010d.png",
                                              "-vcodec", "libx264",
                                              "-g", "15",
                                              "-keyint_min", "15",
                                              "-qp", f"{qp}",
                                            #   "-r", f"{fps}",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
    else:
        encoding_result = subprocess.run(["ffmpeg", "-y",
                                          "-start_number", str(start_id),
                                          "-i", f"{images_path}/%010d.png",
                                          "-loglevel", "error",
                                          "-vcodec", "libx264",
                                          "-pix_fmt", "yuv420p", 
                                          "-crf", "23",
                                          encoded_vid_path],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         universal_newlines=True)

    size = 0
    if encoding_result.returncode != 0:
        # Encoding failed
        print("ENCODING FAILED")
        print(encoding_result.stdout)
        print(encoding_result.stderr)
        exit()
    else:
        size = os.path.getsize(encoded_vid_path)

    return size

# [DECODING]: convert video into images sequence
def extract_images_from_video(images_path, req_regions, fps):
    if not os.path.isdir(images_path):
        print("%s not a valid directory", images_path)
        return

    for fname in os.listdir(images_path):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))

    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    extracted_images_path = os.path.join(images_path, "%010d.png")

    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                      "-g", "8", 
                                      "-q:v", "2",
                                    #   "-r", f"{fps}",
                                      "-vsync", "0", 
                                      "-start_number", "0",
                                      extracted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()

    fnames = sorted(
        [os.path.join(images_path, name)
         for name in os.listdir(images_path) if "png" in name])
    fids = sorted(list(set([r.fid for r in req_regions.regions])))
    fids_mapping = zip(fids, fnames) # one-to-one mapping => fids_mapping = [(fid1, fname1), (fid2, fname2), ...]
    for fname in fnames:
        # Rename temporarily
        os.rename(fname, f"{fname}_temp")

    for fid, fname in fids_mapping:
        os.rename(os.path.join(f"{fname}_temp"),
                  os.path.join(images_path, f"{str(fid).zfill(10)}.png"))


# crop images and return the number of frames
# "vide_name" will be used as path to create a dir
def crop_images(results, vid_name, images_direc, resolution=None):
    cached_image = None
    # key: frame id, value: image
    cropped_images = {}

    # this FOR loop aims at cropping each frame
    for region in results.regions:

        # read the frame into cached_image (for later modification)
        if not (cached_image and
                cached_image[0] == region.fid):
            image_path = os.path.join(images_direc,
                                      f"{str(region.fid).zfill(10)}.png")
            # cached_image is a tuple: (image of a frame, its frame id) 
            cached_image = (region.fid, cv.imread(image_path))

        # ** Just move the complete image
        # If in mpeg mode, we dont need to crop the frame, so we directly use the original (complete) frame
        if region.x == 0 and region.y == 0 and region.w == 1 and region.h == 1:
            cropped_images[region.fid] = cached_image[1]
            continue
        
        # If in dds mode, we need to crop the frame
        # resolution = width * height
        width = cached_image[1].shape[1] 
        height = cached_image[1].shape[0] 
        # (x0, y0): ; (x1, y1): 
        x0 = int(region.x * width)
        y0 = int(region.y * height)
        x1 = int((region.w * width) + x0 - 1)
        y1 = int((region.h * height) + y0 - 1)

        # if not cropped, then set it all black
        if region.fid not in cropped_images: 
            cropped_images[region.fid] = np.zeros_like(cached_image[1])

        cropped_image = cropped_images[region.fid]
        cropped_image[y0:y1, x0:x1, :] = cached_image[1][y0:y1, x0:x1, :]
        cropped_images[region.fid] = cropped_image

    # resize the frame according to resolution scale factor, then save the new (smaller) frames
    os.makedirs(vid_name, exist_ok=True)
    frames_count = len(cropped_images)
    frames = sorted(cropped_images.items(), key=lambda e: e[0])
    for idx, (_, frame) in enumerate(frames):
        if resolution: # resolution is a float point (resize scale factor)
            w = int(frame.shape[1] * resolution)
            h = int(frame.shape[0] * resolution)
            im_to_write = cv.resize(frame, (w, h), fx=0, fy=0,
                                    interpolation=cv.INTER_CUBIC)
            frame = im_to_write
        cv.imwrite(os.path.join(vid_name, f"{str(idx).zfill(10)}.png"), frame,
                   [cv.IMWRITE_PNG_COMPRESSION, 0])

    return frames_count


def merge_images(cropped_images_direc, low_images_direc, req_regions):
    # MERGE: put high-res regions into low-res images
        # Merge Logic:
        # 1. Read high-res images
        # 2. Enlarge low-res images 
        # 3. Put high-res regions in place (for each frame)
    images = {}
    for fname in os.listdir(cropped_images_direc):
        if "png" not in fname:
            continue
        fid = int(fname.split(".")[0])


        # Read high resolution image
        high_image = cv.imread(os.path.join(cropped_images_direc, fname))
        width = high_image.shape[1]
        height = high_image.shape[0]

        # Read low resolution image
        low_image = cv.imread(os.path.join(low_images_direc, fname))
        # Enlarge low resolution image
        enlarged_image = cv.resize(low_image, (width, height), fx=0, fy=0,
                                   interpolation=cv.INTER_CUBIC)
        # Put high-res regions in place in low-res image
        for r in req_regions.regions:
            if fid != r.fid:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int((r.w * width) + x0 - 1)
            y1 = int((r.h * height) + y0 - 1)

            enlarged_image[y0:y1, x0:x1, :] = high_image[y0:y1, x0:x1, :]
        cv.imwrite(os.path.join(cropped_images_direc, fname), enlarged_image,
                   [cv.IMWRITE_PNG_COMPRESSION, 0])
        images[fid] = enlarged_image
    return images


def compute_regions_size(results, vid_name, images_direc, 
                         resolution, qp, fps, 
                         enforce_iframes, estimate_banwidth=True):
    if estimate_banwidth:
        # If not simulation, compress and encode images
        # and get size

        # compress images by resolution
        vid_name = f"{vid_name}-cropped"
        frames_count = crop_images(results, vid_name, images_direc,
                                   resolution)
        
        # encode compressed images into a video according to qp and fps
        size = compress_and_get_size(vid_name, 0, frames_count, qp=qp, fps=fps,
                                     enforce_iframes=enforce_iframes,
                                     resolution=1)
        pixel_size = compute_area_of_regions(results)
        return size, pixel_size

        # return 0,0
    else:
        size = compute_area_of_regions(results)

        return size


def cleanup(vid_name, debug_mode=False, start_id=None, end_id=None):
    if not os.path.isdir(vid_name + "-cropped"):
        return

    if not debug_mode:
        shutil.rmtree(vid_name + "-base-phase-cropped")
        shutil.rmtree(vid_name + "-cropped")
    else:
        if start_id is None or end_id is None:
            print("Need start_fid and end_fid for debugging mode")
            exit()
        os.makedirs("debugging", exist_ok=True)
        leaf_direc = vid_name.split("/")[-1] + "-cropped"
        shutil.move(vid_name + "-cropped", "debugging")
        shutil.move(os.path.join("debugging", leaf_direc),
                    os.path.join("debugging",
                                 f"{leaf_direc}-{start_id}-{end_id}"),
                    copy_function=os.rename)


def get_size_from_mpeg_results(results_log_path, images_path, resolution):
    with open(results_log_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if line.rstrip().lstrip() != ""]

    num_frames = len([x for x in os.listdir(images_path) if "png" in x])

    bandwidth = 0
    for idx, line in enumerate(lines):
        if f"RES {resolution}" in line:
            bandwidth = float(lines[idx + 2])
            break
    size = bandwidth * 1024.0 * (num_frames / 10.0)
    return size


def filter_results(bboxes, gt_flag, gt_confid_thresh, mpeg_confid_thresh,
                   max_area_thresh_gt, max_area_thresh_mpeg):
    relevant_classes = ["vehicle"]
    if gt_flag:
        confid_thresh = gt_confid_thresh
        max_area_thresh = max_area_thresh_gt

    else:
        confid_thresh = mpeg_confid_thresh
        max_area_thresh = max_area_thresh_mpeg

    result = []
    for b in bboxes:
        b = b.x, b.y, b.w, b.h, b.label, b.conf
        (x, y, w, h, label, confid) = b
        if (confid >= confid_thresh and w*h <= max_area_thresh and
                label in relevant_classes):
            result.append(b)
    return result


def iou(b1, b2):
    (x1, y1, w1, h1, label1, confid1) = b1
    (x2, y2, w2, h2, label2, confid2) = b2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x4 = min(x1+w1, x2+w2)
    y4 = min(y1+h1, y2+h2)
    if x3 > x4 or y3 > y4:
        return 0
    else:
        overlap = (x4-x3)*(y4-y3)
        return overlap/(w1*h1+w2*h2-overlap)


def get_fid_by_fname(fname):
    rows = []
    num_of_frames = 0

    with open(fname) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rows.append(row)

    for row in rows:
        if (int(row[0]) > num_of_frames):
            num_of_frames = int(row[0])
        
        elif (int(row[0]) < num_of_frames): 
            break

    return num_of_frames + 1


def get_fid_by_results(regions_dict):
    # print(type(regions_dict))
    return len(regions_dict.keys())


# modify the tested results according to the fps difference
# final_results.regions_dict {fid1: [region1, region2, ...], fid2: [region1, region2, ...], ...}
def modify_results(map_dd, num_frames_cur, num_frames_gt, fps_cur, fps_gt=30):
    if (num_frames_cur == num_frames_gt or num_frames_cur == 0):
        print("Num frame of FPS is 0, not modified, return directly")
        return map_dd
        
    else:
        divisible = [1,2,5,10,15]
        if (fps_cur in divisible):
            ratio = int(fps_gt / fps_cur)

            if ratio != 1:
                updated_results = Results()
                count = 0

                # generate copies of results of each current frame according to fps_ratio
                # num_frames_gt = num_frames_cur * (fps_gt / fps_cur)
                for fid in range(num_frames_cur):

                    for i in range(ratio):
                        regions = copy.deepcopy(map_dd[fid])

                        # update fid of the regions in cur frame
                        for r in regions:
                            r.fid = count
                            updated_results.regions.append(r)

                        updated_results.regions_dict[count] = regions
                    
                        count += 1
                        

                # for i in range(8):
                #     print("fid: ", updated_results.regions_dict[i][0].fid)
                #     print(updated_results.regions_dict[i][0])    

                # # write modified results into file for checking
                # fname = "results/test_modify_results"
                # results_files = open(fname, "w")
                # csv_writer = csv.writer(results_files)
                # for region in updated_results.regions:
                #     row = [region.fid, region.x, region.y,
                #         region.w, region.h,
                #         region.label, region.conf,
                #         region.resolution, region.origin]
                #     csv_writer.writerow(row)
                # results_files.close()

                return updated_results.regions_dict
                    
            else:
                return map_dd

        else:
            updated_results = Results()
            count = 0
            frame_diff = num_frames_gt - num_frames_cur
            fid_list = []

            while (len(fid_list) < frame_diff):
                index = random.randint(0, num_frames_cur-1)
                if (index not in fid_list): 
                    fid_list.append(index)

            # print("fid that needs to repeat are:")
            # print(fid_list)


            for fid in range(num_frames_cur):

                regions = copy.deepcopy(map_dd[fid])

                # update fid of the regions in cur frame
                for r in regions:
                    r.fid = count
                    updated_results.regions.append(r)

                updated_results.regions_dict[count] = regions

                count += 1

                # need to repeat
                if (fid in fid_list):
                    for r in regions:
                        r.fid = count
                        updated_results.regions.append(r)

                    updated_results.regions_dict[count] = regions

                    count += 1
    

            # for i in range(8):
            #     print("fid: ", updated_results.regions_dict[i][0].fid)
            #     print(updated_results.regions_dict[i][0])    

            # # write modified results into file for checking
            # fname = "results/test_modify_results"
            # results_files = open(fname, "w")
            # csv_writer = csv.writer(results_files)
            # for region in updated_results.regions:
            #     row = [region.fid, region.x, region.y,
            #         region.w, region.h,
            #         region.label, region.conf,
            #         region.resolution, region.origin]
            #     csv_writer.writerow(row)
            # results_files.close()
            # # print("count, ", count)

            return updated_results.regions_dict        


# generate images for video according to specified fps
def generate_images(video_name, images_path, fps):
    if not os.path.isdir(images_path):
        print("%s not a valid directory", images_path)
        return

    for fname in os.listdir(images_path):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))

    encoded_vid_path = os.path.join(images_path, f"{video_name}.mp4")
    extacted_images_path = os.path.join(images_path, "%010d.png")

    # print(images_path)
    # print(encoded_vid_path)
    # print(extacted_images_path)

    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                    #   "-g", "8", 
                                      "-q:v", "2",
                                      "-r", f"{fps}",
                                    #   "-vsync", "0", 
                                      "-start_number", "0",
                                      extacted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()
    # else:
    #     print("done. imgs in ", images_path)


# get the duration of a video (in seconds)
def get_duration(fname):
    fname += ".mp4"
    fname = os.path.join("../dataset/", fname)

    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", fname],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)

    return float(result.stdout)


def generate_images(video_name, fps):
    dataset_dir = "../dataset"
    # video_name = f"{video_name}_gt"
    images_path = os.path.join(dataset_dir, video_name, 'src')
    # print(images_path)

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for fname in os.listdir(images_path):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))

    encoded_vid_path = os.path.join(dataset_dir, f"{video_name}.mp4")
    extacted_images_path = os.path.join(images_path, "%010d.png")
    # print(encoded_vid_path)
    # print(extacted_images_path)


    #subprocess.run("conda activate tf")
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                    #   "-g", "8", 
                                      "-q:v", "2",
                                      "-r", f"{fps}",
                                    #   "-vsync", "0", 
                                      "-start_number", "0",
                                      extacted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()
    # else:
    #     num_of_frames = len([x for x in os.listdir(images_path) if "png" in x])
    #     print("fps:", fps, " num of frames: ", num_of_frames)


def enlarge_area(bbox, ratio):
    (x, y, w, h, label, confid) = bbox
    x_min = max(x - w * ratio, 0.0)
    y_min = max(y - h * ratio, 0.0)
    x_max = min(x + w * (1 + ratio), 1.0)
    y_max = min(y + h * (1 + ratio), 1.0)
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    return (x, y, w, h, label, confid)



def evaluate(max_fid, map_dd, map_gt, 
             gt_confid_thresh, mpeg_confid_thresh,
             max_area_thresh_gt, max_area_thresh_mpeg, 
             enlarge_ratio=0, iou_thresh=0.3, f1_thresh=0.5):

    if (not map_dd):
        return 0,0,0,0,0,0,0,0,0
    
    tp_list = []
    fp_list = []
    fn_list = []
    count_list = []
    f1_list = []
    obj_num_list = []
    before_dd_len = 0
    before_gt_len = 0
    after_dd_len = 0
    after_gt_len = 0

    # print("max fid = ", max_fid)

    for fid in range(max_fid):
        # print("fid: ",fid)
        bboxes_dd = map_dd[fid]
        bboxes_gt = map_gt[fid]
        before_dd_len += len(bboxes_dd)
        before_gt_len += len(bboxes_gt)

        # filter results and calculate number of objects
        bboxes_dd = filter_results(
            bboxes_dd, gt_flag=False, 
            gt_confid_thresh=gt_confid_thresh,
            mpeg_confid_thresh=mpeg_confid_thresh,
            max_area_thresh_gt=max_area_thresh_gt,
            max_area_thresh_mpeg=max_area_thresh_mpeg)
        bboxes_gt = filter_results(
            bboxes_gt, gt_flag=True, 
            gt_confid_thresh=gt_confid_thresh,
            mpeg_confid_thresh=mpeg_confid_thresh,
            max_area_thresh_gt=max_area_thresh_gt,
            max_area_thresh_mpeg=max_area_thresh_mpeg)
        
        # number of objects in each frame
        obj_num_list.append(len(bboxes_dd))

        after_dd_len += len(bboxes_dd)
        after_gt_len += len(bboxes_gt)

        # enlarging the ground truth bouding box
        if (enlarge_ratio != 0):
            bboxes_gt_temp = []
            for b_gt in bboxes_gt:
                b_gt = enlarge_area(b_gt, enlarge_ratio)
                bboxes_gt_temp.append(b_gt)
            bboxes_gt = bboxes_gt_temp


        # start to compare
        tp = 0
        fp = 0
        fn = 0
        count = 0
        
        for b_dd in bboxes_dd:
            found = False
            for b_gt in bboxes_gt:
                if iou(b_dd, b_gt) >= iou_thresh:
                    found = True
                    break
            if found:
                tp += 1
            else:
                fp += 1

        for b_gt in bboxes_gt:
            found = False
            for b_dd in bboxes_dd:
                if iou(b_dd, b_gt) >= iou_thresh:
                    found = True
                    break
            if not found:
                fn += 1
            else:
                count += 1
        
        # skip for empty frame
        if (tp == 0 and fp == 0 and fn ==0):
            continue

        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        count_list.append(count)
        f1 = round((2.0*tp/(2.0*tp+fp+fn)), 3)
        f1_list.append(f1)

    tp = sum(tp_list)
    fp = sum(fp_list)
    fn = sum(fn_list)
    count = sum(count_list)

    # print(f"before filter: dd-{before_dd_len} gt-{before_gt_len}")
    # print(f"after filter: dd-{after_dd_len} gt-{after_gt_len}")

    # F1_FRAC = fraction of frames larger than f1 score threshold
    total = len(f1_list)
    partial = len([x for x in f1_list if x >= f1_thresh])
    # print(f"------------ACCURACY partial:{partial} total:{total}")
    f1_frac = 0
    f1_all = 0
    if (total != 0):
        f1_frac = round(partial/total, 3)
    # print("total-", total, " good-", partial, " f1-", f1_frac)

    if (tp+fp+fn != 0):
        # F1_ALL = overall f1 score
        f1_all = round((2.0*tp/(2.0*tp+fp+fn)), 3)

    # avg number of objects
    avg_num_objs = 0
    if (len(obj_num_list) != 0):
        avg_num_objs = round(sum(obj_num_list) / len(obj_num_list))

    precision = 0
    recall = 0
    if (tp+fp != 0):
        precision = round(tp/(tp+fp), 3)

    if (tp+fn != 0):
        recall = round(tp/(tp+fn), 3)

    no_objs = False
    if (tp+fp+fn == 0):
        no_objs = True

    return (tp, fp, fn, count,
            precision, recall,
            f1_frac, 
            avg_num_objs, 
            no_objs)


def write_stats_txt(fname, vid_name, config, f1, stats,
                    bw, frames_count, time_count, mode):
    header = ("video-name,low-resolution,high-resolution,low_qp,high_qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,time,mode")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{time_count},{mode}")

    if not os.path.isfile(fname):
        str_to_write = f"{header}\n{stats}\n"
    else:
        str_to_write = f"{stats}\n"

    with open(fname, "a") as f:
        f.write(str_to_write)


def write_stats_csv(fname, vid_name, config, f1, stats, bw,
                    frames_count, time_count, mode):
    header = ("video-name,low-resolution,high-resolution,low-qp,high-qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,time,mode").split(",")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{time_count},{mode}").split(",")

    results_files = open(fname, "a")
    csv_writer = csv.writer(results_files)
    if not os.path.isfile(fname):
        # If file does not exist write the header row
        csv_writer.writerow(header)
    csv_writer.writerow(stats)
    results_files.close()


def write_stats(fname, vid_name, config, f1, stats, bw,
                frames_count, time_count, mode):
    if re.match(r"\w+[.]csv\Z", fname):
        write_stats_csv(fname, vid_name, config, f1, stats, bw,
                        frames_count, time_count, mode)
    else:
        write_stats_txt(fname, vid_name, config, f1, stats, bw,
                        frames_count, time_count, mode)


def visualize_regions(results, images_direc,
                      low_conf=0.0, high_conf=1.0,
                      label="debugging"):
    idx = 0
    fids = sorted(list(set([r.fid for r in results.regions])))
    while idx < len(fids):
        image_np = cv.imread(
            os.path.join(images_direc, f"{str(fids[idx]).zfill(10)}.png"))
        width = image_np.shape[1]
        height = image_np.shape[0]
        regions = [r for r in results.regions if r.fid == fids[idx]]
        for r in regions:
            if r.conf < low_conf or r.conf > high_conf:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int(r.w * width + x0)
            y1 = int(r.h * height + y0)
            cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv.putText(image_np, f"{fids[idx]}", (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv.imshow(label, image_np)
        key = cv.waitKey()
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("k"):
            idx -= 2

        idx += 1
    cv.destroyAllWindows()


def visualize_single_regions(region, images_direc, label="debugging"):
    image_path = os.path.join(images_direc, f"{str(region.fid).zfill(10)}.png")
    image_np = cv.imread(image_path)
    width = image_np.shape[1]
    height = image_np.shape[0]

    x0 = int(region.x * width)
    y0 = int(region.y * height)
    x1 = int((region.w * width) + x0)
    y1 = int((region.h * height) + y0)

    cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv.putText(image_np, f"{region.fid}, {region.label}, {region.conf:0.2f}, "
               f"{region.w * region.h}",
               (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv.imshow(label, image_np)
    cv.waitKey()
    cv.destroyAllWindows()
