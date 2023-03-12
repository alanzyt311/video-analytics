import logging
import os
import sys
import time
import shutil
import requests
import json
from dds_utils import (Results, read_results_dict, read_partial_results_dict,
                       cleanup, Region, modify_results,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results, get_duration,
                       get_fid_by_fname, get_fid_by_results, evaluate)
import yaml
 
from controller import a2c
from bw_estimator import estimator
from utils import utils
from utils.utils import generate_images



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

        self.logger.info(f"Client initialized")


    def update_config(self, new_config):
        self.config = new_config


    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):
        
        # fpath = os.path.join("../dataset/", video_name)
        # get_duration(fpath)
        # return 1

        # print(self.config.tested_images_path)
        # print(self.config.gt_images_path)
        # return 0

        latency_list = []
        f1_list = []
        bw_list = []

        # Infinite loop, for real experiments
        # while (continued_streaming):

        # Finite loop, for testing
        for k in range(0, 6, self.config.update_freq):
            start_time = k

            ################################################################
            ############################ CLIENT ############################
            ################################################################


            if (len(f1_list) != 0):
                self.config = a2c.update_config(f1_list, bw_list, latency_list, self.config, avg_num_objs, est_bw)
                self.logger.info(f"Config Updated to [{self.config.low_qp}QP] [{self.config.fps}fps] [{self.config.low_resolution}res]")

            self.logger.info(f"Inference on video start from {start_time}s to {start_time+self.config.update_freq}s " 
                                f" using [{self.config.low_qp}QP] [{self.config.fps}fps] [{self.config.low_resolution}res]")



            # GENERATE IMAGES IN CURRENT TIME SLOT WITH CERTAIN CONFIG
            raw_images_path = generate_images(self.config.raw_video_name, self.config.fps, self.config.low_qp, 
                                                    self.config.low_resolution, start_time, self.config.update_freq, False)
            print(f"Images path after generation: {raw_images_path}")
            

            # total size that transmit is the size of modified and sampled frames
            total_size = os.path.getsize(raw_images_path)
            self.logger.info(f"{total_size / 1024}KB sent "
                                        f" using [{self.config.low_qp}QP] [{self.config.fps}fps] [{self.config.low_resolution}res]")
            # return 0


            # preparation for inference   
            number_of_frames = len(
                [f for f in os.listdir(raw_images_path) if ".png" in f])

            final_results = Results()
            final_rpn_results = Results()

            # count time
            time_diff = 0
            self.logger.info("Start counting time")
            time_start = time.perf_counter()

            ############### Task transfered to Server ################


            ################################################################
            ############################ SERVER ############################
            ################################################################


            # perform inference for each batch on SERVER
            for i in range(0, number_of_frames, self.config.batch_size):
                start_frame = i
                end_frame = min(number_of_frames, i + self.config.batch_size)

                # sort the frames to make it ordered
                batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                    for idx in range(start_frame, end_frame)])
                
                # initialized the req_regions(result)
                req_regions = Results()
                for fid in range(start_frame, end_frame): #fid = frame id
                    req_regions.append(
                        Region(fid, 0, 0, 1, 1, 1.0, 2,
                            self.config.low_resolution))

                # ###### Encoding ######
                # # batch_video_size is size of compressed video for this batch
                # # _ is the pixel_size (total area of regions in all frames)
                # batch_video_size, pixel_size = compute_regions_size(
                #     req_regions, f"{video_name}-base-phase", raw_images_path,
                #     self.config.low_resolution, self.config.low_qp, self.config.fps,
                #     enforce_iframes, True)
                # self.logger.info(f"{batch_video_size / 1024}KB sent "
                #                 f"in base phase using [{self.config.low_qp}QP] [{self.config.low_resolution}] [{self.config.fps}FPS]")
                # nf = len(
                #     [f for f in os.listdir(f"{video_name}-base-phase-cropped") if ".png" in f])

                # ############### Task transfered to Server ################

                # ###### Decoding ######
                # extract_images_from_video(f"{video_name}-base-phase-cropped",
                #                         req_regions, self.config.fps)
                # nf1 = len(
                #     [f for f in os.listdir(f"{video_name}-base-phase-cropped") if ".png" in f])
                # # print("before crop len=", nf, "after crop len=", nf1)



                # detection results/rpn_results: 
                #   A list of (class, confidence, (x, y, w, h)) tuples
                results, rpn_results = (
                    self.server.perform_detection(
                        raw_images_path,
                        self.config.low_resolution, batch_fnames))

                self.logger.info(f"Detection {len(results)} regions for "
                                f"batch {start_frame} to {end_frame} in {raw_images_path}")
                # modify the regions_dict here
                final_results.combine_results(
                    results, self.config.intersection_threshold)
                final_rpn_results.combine_results(
                    rpn_results, self.config.intersection_threshold)

                # Remove encoded video manually
                # shutil.rmtree(f"{video_name}-base-phase-cropped")
                # total_size += batch_video_size
            


            # calculate time diff and sum up
            time_end = time.perf_counter()
            time_diff = round((time_end - time_start), 2)
            self.logger.info(f"Stop counting time, cost {time_diff}s")
            latency_list.append(time_diff)


            # handling results
            final_results = merge_boxes_in_results(
                final_results.regions_dict, 0.3, 0.3)
            final_results.fill_gaps(number_of_frames)

            # Add RPN regions to final_results
            final_results.combine_results(
                final_rpn_results, self.config.intersection_threshold)

            # write results into file
            # final_results.write(video_name)


            #######################
            ###### EVALUTION ######
            #######################
            self.logger.info("Start evaluation of last batch")

            # PART I: Accuracy and Number of objects
            self.logger.info("Accuracy measurement")
            f1 = 0
            stats = (0, 0, 0)



            # number_of_frames_gt = get_fid_by_fname(args.ground_truth)
            

            # read partial gt results with fid in [start_fid, end_fid), 
            # and reset fid to (0~length)
            start_fid = k * self.config.fps_gt
            length = self.config.update_freq * self.config.fps_gt
            ground_truth_dict, _ = read_partial_results_dict(self.config.ground_truth, start_fid, length)
            self.logger.info("Reading ground truth results complete")
            number_of_frames_gt = get_fid_by_results(ground_truth_dict)
            # print("gt_num_of_frames: ", number_of_frames_gt)


            # modify current result according to fps difference
            number_of_frames_cur = get_fid_by_results(final_results.regions_dict)
            # print("Before modification cur_num_of_frames: ",number_of_frames_cur)

            final_results.regions_dict = modify_results(final_results.regions_dict, 
                                                        number_of_frames_cur, number_of_frames_gt,
                                                        self.config.fps, self.config.fps_gt)   

            number_of_frames_cur = get_fid_by_results(final_results.regions_dict)
            # print("After modification cur_num_of_frames: ",number_of_frames_cur)         



            # def evaluate(max_fid, map_dd, map_gt, 
            #   gt_confid_thresh, mpeg_confid_thresh, 
            #   max_area_thresh_gt, max_area_thresh_mpeg, 
            #   enlarge_ratio=0, iou_thresh=0.3, f1_thresh=0.5):
            tp, fp, fn, count, precision, recall, f1, avg_num_objs = evaluate(
                number_of_frames_gt, final_results.regions_dict, ground_truth_dict,
                self.config.low_threshold, 0.5, 
                0.5, 0.5, 
                0, 0.2, 0.7)

            f1_list.append(f1)
            stats = (tp, fp, fn)
            self.logger.info(f"Got an f1 score of {f1} "
                        f"for this experiment with "
                        f"tp {tp} fp {fp} fn {fn} "
                        f"with total bandwidth {total_size/1024} KB")

            self.logger.info(f"Avg number of objects of this segment is {avg_num_objs}")

            # PART II: Estimated bandwidth
            bw_list.append(total_size)
            est_bw = estimator.bw_estimate(bw_list)
            self.logger.info(f"Estimated bandwidth for next segment is {est_bw/1024}KB")

            # PART III: 




            # invoke agent to determine the config for next loop
            
            cum_accuracy = sum(f1_list) / len(f1_list)

            # state = [[],[]]


        return 0, 0
