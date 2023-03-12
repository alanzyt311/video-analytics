import logging
import os
import sys
import time
import shutil
import requests
import json
# from dds_utils import (Results, read_results_dict, read_partial_results_dict,
#                        cleanup, Region, modify_results,
#                        compute_regions_size, extract_images_from_video,
#                        merge_boxes_in_results, get_duration,
#                        get_fid_by_fname, get_fid_by_results, evaluate)
from dds_utils import (Results, read_results_dict, read_partial_results_dict,
                       cleanup, Region, modify_results,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results,
                       get_fid_by_fname, get_fid_by_results, evaluate)
import yaml

# from controller import a2c
from bw_estimator import estimator
from mvs import mvs
from utils import utils
from utils.utils import generate_images, get_folder_size
sys.path.append('../')
from motion import motion
import pandas as pd

Max_Latncy = 20
Max_Bandwith = 80
Max_Obj = 20

IOU_THRE = 0.8

class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""

    def __init__(self, hname, config, server_handle=None):
        if hname:
            self.hname = hname
            self.session = requests.Session()
        else:
            self.server = server_handle

        self.config = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)


        self.last_end_time = 0
        self.bw_list = []

        self.terminate = False


        # For normalization
        self.max_bw = 0


        self.logger.info(f"Client initialized")


    def update_config(self, new_config):
        self.config = new_config

    def update(self, res, fps, qp):
        self.config.low_resolution = res
        self.config.fps = fps
        self.config.low_qp = qp

    def normalize(self, lantency, bandwith, obj):
        nor_lantency = lantency / Max_Latncy
        nor_bandiwith = bandwith / Max_Bandwith
        nor_obj = obj / Max_Obj
        return nor_lantency, nor_bandiwith, nor_obj

    def analyze_video(self):

        latency_list = []
        f1_list = []
        bw_list = []
        avg_num_objs = 0
        est_bw = 0

        # Infinite loop, for real experiments
        # while (continued_streaming):

        # Finite loop, for testing
        # for k in range(0, 6, self.config.update_freq):

        k = self.last_end_time
        start_time = self.last_end_time
        if (start_time + self.config.update_freq < self.config.duration):
            end_time = start_time + self.config.update_freq
        else:
            end_time = self.config.duration
            self.terminate = True



        ################################################################
        ############################ CLIENT ############################
        ################################################################



        # self.logger.info(f"Config Updated to [{self.config.low_qp}QP] [{self.config.fps}fps] [{self.config.low_resolution}res]")
        self.logger.info(f"Config Updated !!!")

        self.logger.info(f"Inference on video start from {start_time}s to {end_time}s "
                         f" using [{self.config.low_resolution}res] [{self.config.fps}fps] [{self.config.low_qp}QP] ")



        # GENERATE IMAGES IN CURRENT TIME SLOT WITH CERTAIN CONFIG
        gt_sample_batch_size_list = [1,2,5,10,15,20,25,30]
        sample_level = 7
        gt_sample_batch_size = gt_sample_batch_size_list[sample_level]

        fps_images_path, gt_images_path, original_num_of_frames = generate_images(gt_sample_batch_size, self.config.raw_video_name, self.config.fps, self.config.low_qp,
                                                                                  self.config.low_resolution, start_time, self.config.update_freq)
        # print(f"Images FPS path after generation: {fps_images_path}")
        # print(f"Images GT path after generation: {gt_images_path}")
        # return 0


        # total size that transmit is the size of modified and sampled frames
        # fps_size = os.path.getsize(fps_images_path)
        # gt_size = os.path.getsize(gt_images_path)
        fps_size = get_folder_size(fps_images_path)
        gt_size = get_folder_size(gt_images_path)

        fps_size = fps_size / 1024 / 1024 #MB
        gt_size = gt_size  / 1024 / 1024 #MB
        # total_size = fps_size + gt_size
        total_size = fps_size + gt_size
        self.logger.info(f"[FPS SIZE] {fps_size}MB sent "
                         f"using [{self.config.low_resolution}res] [{self.config.fps}fps] [{self.config.low_qp}QP] ")
        self.logger.info(f"[GT SIZE] {gt_size}MB sent")
        # return 0


        ############### Task transfered to Server ################


        ################################################################
        ############################ SERVER ############################
        ################################################################



        ############### FPS Inference ################

        # preparation for inference
        number_of_frames_fps = len(
            [f for f in os.listdir(fps_images_path) if ".png" in f])

        final_results_fps = Results()
        final_rpn_results_fps = Results()

        # count time
        time_diff = 0
        self.logger.info("Start counting time")
        time_start = time.perf_counter()



        # perform inference for each batch on SERVER
        for i in range(0, number_of_frames_fps, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames_fps, i + self.config.batch_size)

            # sort the frames to make it ordered
            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            # initialized the req_regions(result)
            req_regions = Results()
            for fid in range(start_frame, end_frame): #fid = frame id
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.low_resolution))


            # detection results/rpn_results: result_fps
            #   A list of (class, confidence, (x, y, w, h)) tuples
            result_fps, rpn_result_fps = (
                self.server.perform_detection(
                    fps_images_path,
                    self.config.low_resolution, batch_fnames))

            self.logger.info(f"[FPS] Detection {len(result_fps)} regions for "
                             f"batch {start_frame} to {end_frame} in {fps_images_path}")
            # modify the regions_dict here
            final_results_fps.combine_results(
                result_fps, self.config.intersection_threshold)
            final_rpn_results_fps.combine_results(
                rpn_result_fps, self.config.intersection_threshold)





        # calculate time diff and sum up
        time_end = time.perf_counter()
        time_diff = round((time_end - time_start), 2)
        self.logger.info(f"Stop counting time, cost {time_diff}s")
        latency_list.append(time_diff)
        

        ############### GT Inference ################

        # # preparation for inference
        # number_of_frames_gt = len([f for f in os.listdir(gt_images_path) if ".png" in f])

        # final_results_gt = Results()
        # final_rpn_results_gt = Results()


        # # perform inference for each batch on SERVER
        # for i in range(0, number_of_frames_gt, self.config.batch_size):
        #     start_frame = i
        #     end_frame = min(number_of_frames_gt, i + self.config.batch_size)

        #     # sort the frames to make it ordered
        #     batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
        #                            for idx in range(start_frame, end_frame)])

        #     # initialized the req_regions(result)
        #     req_regions = Results()
        #     for fid in range(start_frame, end_frame): #fid = frame id
        #         req_regions.append(
        #             Region(fid, 0, 0, 1, 1, 1.0, 2,
        #                    self.config.low_resolution))


        #     # detection results/rpn_results:
        #     #   A list of (class, confidence, (x, y, w, h)) tuples
        #     results_gt, rpn_results_gt = (
        #         self.server.perform_detection(
        #             gt_images_path,
        #             self.config.low_resolution, batch_fnames))

        #     self.logger.info(f"[GT] Detection {len(results_gt)} regions for "
        #                      f"batch {start_frame} to {end_frame} in {gt_images_path}")
        #     # modify the regions_dict here
        #     final_results_gt.combine_results(
        #         results_gt, self.config.intersection_threshold)
        #     final_rpn_results_gt.combine_results(
        #         rpn_results_gt, self.config.intersection_threshold)




        # handling fps results
        final_results_fps = merge_boxes_in_results(
            final_results_fps.regions_dict, 0.3, 0.3)
        # print(f"======== FPS dict keys before len = {len(final_results_fps.regions_dict.keys())} =========")
        # print(sorted(final_results_fps.regions_dict.keys()))           


        # print("=======combine RPN for FPS=======")
        # Add RPN regions to final_results_fps
        final_results_fps.combine_results(
            final_rpn_results_fps, self.config.intersection_threshold)
        # print(f"======== FPS dict keys after RPN len = {len(final_results_fps.regions_dict.keys())}=========")
        # print(sorted(final_results_fps.regions_dict.keys()))

        # print(f"===========filling gaps for FPS... with {number_of_frames_fps} frames===========")
        final_results_fps.fill_gaps(number_of_frames_fps)
        # print(f"======== FPS dict keys after GAP len = {len(final_results_fps.regions_dict.keys())}=========")
        # print(sorted(final_results_fps.regions_dict.keys()))


        # # handling gt results
        # final_results_gt = merge_boxes_in_results(
        #     final_results_gt.regions_dict, 0.3, 0.3)
        # # print(f"======== GT dict keys before len = {len(final_results_gt.regions_dict.keys())}=========")
        # # print(sorted(final_results_gt.regions_dict.keys()))


        # # print("=======combine RPN for GT=======")
        # # Add RPN regions to final_results_gt
        # final_results_gt.combine_results(
        #     final_rpn_results_gt, self.config.intersection_threshold)
        # # print(f"======== GT dict keys after RPN len = {len(final_results_gt.regions_dict.keys())}=========")
        # # print(sorted(final_results_gt.regions_dict.keys()))


        # # print(f"===========filling gaps for GT... with {number_of_frames_gt} frames===========")
        # final_results_gt.fill_gaps(number_of_frames_gt)
        # # print(f"======== GT dict keys after len = {len(final_results_gt.regions_dict.keys())}=========")
        # # print(sorted(final_results_gt.regions_dict.keys()))



        # write results into file
        # final_results_fps.write(video_name)







        #######################
        ###### EVALUTION ######
        #######################
        self.logger.info("Start evaluation of last batch")

        # PART I: Accuracy and Number of objects
        self.logger.info("Accuracy measurement")
        f1 = 0



        # number_of_frames_gt = get_fid_by_fname(args.ground_truth)

        ############### GT result handling ################
        # read partial gt results with fid in [start_fid, end_fid),
        # and reset fid to (0~length)
        start_fid = k * self.config.fps_gt
        length = self.config.update_freq * self.config.fps_gt
        ground_truth_dict, _ = read_partial_results_dict(self.config.ground_truth, start_fid, length)
        self.logger.info("Reading ground truth results complete")
        number_of_frames_gt = get_fid_by_results(ground_truth_dict)
        # self.logger.info("Fids are: \n",ground_truth_dict.keys)
        # print("gt_num_of_frames: ", number_of_frames_gt)
        # ground_truth_dict.fill_gaps(number_of_frames_gt)
        # print("gt_num_of_frames: ", number_of_frames_gt)




        # print(f"Total target frames: {original_num_of_frames}")
        # enlarge the partial gt set to 30 frames
        # number_of_frames_gt = get_fid_by_results(final_results_gt.regions_dict) # partial GT results
        # print("Before modification partial gt_num_of_frames: ",number_of_frames_gt)

        # number_of_frames_gt = 30
        # ground_truth_dict = modify_results(final_results_gt.regions_dict,
        #                                    number_of_frames_gt, original_num_of_frames,
        #                                    gt_sample_batch_size, self.config.fps_gt)
        # self.logger.info("Reading ground truth results complete")
        # number_of_frames_gt = get_fid_by_results(ground_truth_dict)
        # print("After modification gt_num_of_frames: ", number_of_frames_gt)




        # modify current result according to fps difference
        number_of_frames_fps = len(final_results_fps.regions_dict.keys())
        # print("Before modification fps_num_of_frames: ",number_of_frames_fps)

        final_results_fps.regions_dict = modify_results(final_results_fps.regions_dict,
                                                        number_of_frames_fps, original_num_of_frames,
                                                        self.config.fps, self.config.fps_gt)

        number_of_frames_fps = get_fid_by_results(final_results_fps.regions_dict)
        # number_of_frames_fps = len(fps_dict.regions_dict.keys())

        # print("After modification fps_num_of_frames: ",number_of_frames_fps)

        if (number_of_frames_gt != number_of_frames_fps):
            print("GT & FPS results num NOT EQUAL")
            print(f"GT={number_of_frames_gt}    FPS={number_of_frames_fps}")
            # return 0

        # return 0,0,0,0,False


        # def evaluate(max_fid, map_dd, map_gt,
        #   gt_confid_thresh, mpeg_confid_thresh,
        #   max_area_thresh_gt, max_area_thresh_mpeg,
        #   enlarge_ratio=0, iou_thresh=0.3, f1_thresh=0.5):
        tp, fp, fn, count, precision, recall, f1, avg_num_objs, no_objs = evaluate(
            number_of_frames_gt, final_results_fps.regions_dict, ground_truth_dict,
            self.config.low_threshold, 0.5,
            0.5, 0.5,
            0, IOU_THRE, 0.7)

        f1_list.append(f1)
        self.logger.info(f"Got an f1 score of {f1} "
                         f"for this experiment with "
                         f"tp {tp} fp {fp} fn {fn} "
                         f"with total bandwidth {fps_size} MB")

        self.logger.info(f"Avg number of objects of this segment is {avg_num_objs}")


        # PART III: MOTION VECTOR
        # PART III: MOTION VECTOR
        if(motion):
            motion_fname = os.path.join("./motion",f"{self.config.raw_video_name}.csv")
            df = pd.read_csv(motion_fname)
            motion_list = df['motion']
            motion_degree = motion_list[start_time]
            # motion_degree = mvs.get_motion(self.config.raw_video_name, self.last_end_time, self.config.update_freq)
            self.logger.info(f"Extract mvs successfully {motion_degree}")

        # update the last end time for next round
        self.last_end_time += self.config.update_freq

        # we should return the state to
        # return f1, total_size, time_diff, avg_num_objs, motion_degree, no_objs, self.terminate
        return f1, fps_size, time_diff, avg_num_objs, motion_degree, self.terminate

        # return f1, total_size, time_diff, avg_num_objs, self.terminate