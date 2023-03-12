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
 
from controller import agent
from bw_estimator import estimator
from utils import utils



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
        accuracy_list = []
        bw_list = []
        avg_num_objs = 0
        est_bw = 0

        # Infinite loop, for real experiments
        # while (continued_streaming):

        # Finite loop, for testing
        for k in range(0, 4, self.config.update_freq):


            ################################################################
            ############################ CLIENT ############################
            ################################################################


            # update config
            if (len(accuracy_list) != 0):
                self.config = agent.update_config(accuracy_list, bw_list, latency_list, self.config, avg_num_objs, est_bw)

                self.logger.info(f"Config Updated to [{self.config.low_qp}QP] [{self.config.fps}fps] [{self.config.low_resolution}res]")



            # Inference
            start_time = k

            self.logger.info(f"Inference on video start from {start_time}s to {start_time+self.config.update_freq}s " 
                                f" using [{self.config.low_qp}QP] [{self.config.fps}fps] [{self.config.low_resolution}res]")



            # GENERATE IMAGES IN CURRENT TIME SLOT WITH CERTAIN CONFIG
            raw_images_path = utils.generate_images(self.config.raw_video_name, self.config.fps, self.config.low_qp, 
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
            print("gt_num_of_frames: ", number_of_frames_gt)


            # modify current result according to fps difference
            number_of_frames_cur = get_fid_by_results(final_results.regions_dict)
            print("Before modification cur_num_of_frames: ",number_of_frames_cur)

            final_results.regions_dict = modify_results(final_results.regions_dict,
                                                        number_of_frames_cur, number_of_frames_gt,
                                                        self.config.fps, self.config.fps_gt)

            number_of_frames_cur = get_fid_by_results(final_results.regions_dict)
            print("After modification cur_num_of_frames: ",number_of_frames_cur)         



            # def evaluate(max_fid, map_dd, map_gt, 
            #   gt_confid_thresh, mpeg_confid_thresh, 
            #   max_area_thresh_gt, max_area_thresh_mpeg, 
            #   enlarge_ratio=0, iou_thresh=0.3, f1_thresh=0.5):
            tp, fp, fn, count, precision, recall, f1, avg_num_objs = evaluate(
                number_of_frames_gt, final_results.regions_dict, ground_truth_dict,
                self.config.low_threshold, 0.5, 
                0.5, 0.5, 
                0, 0.2, 0.7)

            accuracy_list.append(f1)
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
            
            cum_accuracy = sum(accuracy_list) / len(accuracy_list)

            # state = [[],[]]
            

        return 0, 0




    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])

        # read existing low-res results
            # low_results_path = f'results/{video_name}_mpeg_{low_res}_{low_qp}'
        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

            # Encode frames in batch and get size
            # Make temporary frames to downsize complete frames
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                             f"in base phase")
            total_size[0] += encoded_batch_video_size

            # Low resolution phase
                # aim: try to find all the regions that need to be queried and inferenced again
                # r1: positive detection in low-res results
                # req_regions: regions need to be query/request for higher res
            low_images_path = f"{video_name}-base-phase-cropped"
            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)

            # put low-res +ve detection into two result sets
            low_phase_results.combine_results(
                r1, self.config.intersection_threshold)
            final_results.combine_results(
                r1, self.config.intersection_threshold)

            # High resolution phase
            if len(req_regions) > 0:
                # Crop, compress and get size
                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")
                total_size[1] += regions_size

                # High resolution phase every three filter
                # r2: positive detection in high-res
                r2 = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions)
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")

                # put high-res +ve detection into two result sets
                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                final_results.combine_results(
                    r2, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        return final_results, total_size

    def init_server(self, nframes):
        self.config['nframes'] = nframes
        response = self.session.post(
            "http://" + self.hname + "/init", data=yaml.dump(self.config))
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            # Need to add exception handling
            exit()

    def get_first_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(
            vid_name + "-base-phase-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/low", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        rpn = Results()
        for region in response_json["req_regions"]:
            rpn.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))

        return results, rpn

    def get_second_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(vid_name + "-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/high", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.high_resolution, "high-res"))

        return results

    def analyze_video(
            self, vid_name, raw_images, config, enforce_iframes):
        final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))

        self.init_server(nframes)

        for i in range(0, nframes, self.config.batch_size):
            start_frame = i
            end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")

            # First iteration
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(
                    fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            low_phase_size += batch_video_size
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                             f"Using QP {self.config.low_qp} and "
                             f"Resolution {self.config.low_resolution}.")
            results, rpn_regions = self.get_first_phase_results(vid_name)
            final_results.combine_results(
                results, self.config.intersection_threshold)
            all_required_regions.combine_results(
                rpn_regions, self.config.intersection_threshold)

            # Second Iteration
            if len(rpn_regions) > 0:
                batch_video_size, _ = compute_regions_size(
                    rpn_regions, vid_name, raw_images,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                high_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                 f"phase. Using QP {self.config.high_qp} and "
                                 f"Resolution {self.config.high_resolution}.")
                results = self.get_second_phase_results(vid_name)
                final_results.combine_results(
                    results, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(vid_name, False, start_frame, end_frame)

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)
