import os
import re
import csv
import logging
import sys
sys.path.append('../')

from backend.server import Server
from frontend.client_read_gt import Client
from dds_utils import (Results, ServerConfig, read_results_dict, 
                       evaluate, write_stats, modify_results, 
                       get_fid_by_fname, get_fid_by_results,
                       get_duration)
from utils import utils
from dds_utils import (get_duration)
from controller.rewards import (calc_rewards_linear)
from controller.controller import (gen_new_config)
import time
from munch import *
import yaml
import multiprocessing as mp
import tensorflow.compat.v1 as tf
import numpy as np

# dirty fix




SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None


 

#===============================================================
"""

State space (A 9x8 matrix)

1. Bandwidth measurement of past k segments
2. Latency measurement of past k segments

3. Accuracy of the last segment

4. FPS of last segment
5. Resolution of last segment
6. QP of last segment

7. Number of objects in last frame

8. Estimated available bandwidth of next segment
9. Degree of motion in next frame

"""
S_INFO = 9 # Dimension
S_LEN = 8  # take how many frames in the past



# Action (Dim = 3 x 8)
# action = [res_level, fps_level, qp_level], each level range in [0,7]
A_INFO = 3
A_LEN = 8 # 8 levels each
# RES_LEVEL = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# FPS_LEVEL = [30, 25, 20, 15, 10, 5, 2, 1]
# QP_LEVEL = [20, 24, 28, 32, 36, 40, 44, 48]
RES_LEVEL = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FPS_LEVEL = [i for i in range(1, 31)]
QP_LEVEL = [i for i in range(20, 50)]


DEFAULT_RES = int(len(RES_LEVEL)/2)-1
DEFAULT_FPS = int(len(FPS_LEVEL)/2)
DEFAULT_QP = int(len(QP_LEVEL)/2)


# Reward
ACC_THRE = 0.5 # accuracy threshold
LAT_THRE = 0.1 # latency threshold
ACC_PENALTY = 0.1
ACC_REWARD = 0.1
LAT_PENALTY = 0.1
LAT_REWARD = 0.1
BW_PENALTY = 0.1
BW_REWARD = 0.1
ENE_PENALTY = 0.1
ENE_REWARD = 0.1



NUM_AGENTS = 1
SUMMARY_DIR = './results'
LOG_FILE = './results/log'


def load_configuration():
    """read configuration information from yaml file

    Returns:
        dict: information of the yaml file 
    """
    with open('configuration.yml', 'r') as config:
        config_info = yaml.load(config, Loader=yaml.FullLoader)
    return config_info


# Create config for each agent for a single video
def create_config(video_name):

    all_instances = load_configuration()

    single_instance = all_instances['default']
    # print(single_instance)
    # return 0

    # video_name = single_instance['video_name']
    data_dir = single_instance['data_dir']

    single_instance['raw_video_name'] = video_name
    single_instance['gt_video_name'] = f"{video_name}_gt"
    tested_images_dir = os.path.join(data_dir, video_name, 'src')
    gt_images_dir = os.path.join(data_dir, f"{video_name}_gt", 'src')

    mpeg_fps = single_instance['fps']
    mpeg_qp = single_instance['low_qp']
    mpeg_resolution = single_instance['low_resolution']
    result_file_name = f"{video_name}_mpeg_{mpeg_resolution}_{mpeg_qp}_{mpeg_fps}"

    # get video duration (in seconds)
    duration = round(get_duration(video_name))
    # duration = 2

    print(duration)

    single_instance['duration'] = duration
    single_instance['video_name'] = f'results/{result_file_name}'
    single_instance['ground_truth'] = f'results/{video_name}_gt'
    single_instance['tested_images_path'] = f'{tested_images_dir}'
    single_instance['gt_images_path'] = f'{gt_images_dir}'
    single_instance['outfile'] = 'stats'

    # print(single_instance)
    return single_instance





# About logging
def init_log_file(fname):
    # if not os.path.exists(fname):
    results_files = open(fname, "a",  newline='')
    csv_writer = csv.writer(results_files)
    header = ("timestamp,resolution,fps,qp,accuracy,available_bw,predict_bw,est_bw_config,video_size,latency,objnum").split(",")
    csv_writer.writerow(header)
    results_files.close()

# fname,rounds,res,fps,qp,accuracy,available_bw,est_bw_config,video_size,latency,num_objs
def write_log(fname,timestamp,resolution,fps,qp,accuracy,available_bw,predict_bw,est_bw_config,video_size,latency,objnums):
    stats = (f"{timestamp},{resolution},{fps},{qp},{accuracy},{available_bw},{predict_bw},{est_bw_config},{video_size},{latency},{objnums}").split(",")

    results_files = open(fname, "a", newline='')
    csv_writer = csv.writer(results_files)
    csv_writer.writerow(stats)
    results_files.close()



def scale_data(data):
    min_value = min(data)
    max_value = max(data)
    new_min = 0
    new_max = 15
    scaled_data = []
    for value in data:
        scaled_value = (value - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
        scaled_data.append(scaled_value)
    return scaled_data


def main(args):

    print(f"{args.raw_video_name}'s duration is {args.duration}")

    logging.basicConfig(
        format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
        level=args.verbosity.upper())

    logger = logging.getLogger("dds")
    logger.addHandler(logging.NullHandler())

    # Make simulation objects
    logger.info(f"Starting server with high threshold of "
                f"{args.high_threshold} low threshold of "
                f"{args.low_threshold} tracker length of "
                f"{args.tracker_length}")
    
    # return 0

 
    config = args

    # config = ServerConfig(
    #     args.resolutions[0], args.resolutions[1], args.qp[0], args.qp[1],
    #     args.batch_size, args.high_threshold, args.low_threshold,
    #     args.max_object_size, args.min_object_size, args.tracker_length,
    #     args.boundary, args.intersection_threshold, args.tracking_threshold,
    #     args.suppression_threshold, args.simulate, args.rpn_enlarge_ratio,
    #     args.prune_score, args.objfilter_iou, args.size_obj)

    server = None
    mode = None
    results, bw = None, None

    logger.info("Start counting time")
    time_start = time.time()


    logger.warning(f"Running in MPEG mode with resolution "
                    f"{args.low_resolution} on {args.video_name}")
    server = Server(config)

    logger.info("Starting client")
    client = Client(args.hname, config, server)

    # init log files
    fname = os.path.join("./logs",f"{client.config.raw_video_name}.csv")
    init_log_file(fname)

    # variables
    rounds = 0
    timestamp = 0
    video_loops = 0
    # last_latency = 0
    # last_accuracy = 0
    # last_bandwidth = 0

    # last_res = DEFAULT_RES
    # last_fps = DEFAULT_FPS
    # last_qp = DEFAULT_QP
    # res_lv = DEFAULT_RES
    # fps_lv = DEFAULT_FPS
    # qp_lv = DEFAULT_QP
    
    res_lv = RES_LEVEL.index(0.6)
    fps_lv = FPS_LEVEL.index(5)
    qp_lv = QP_LEVEL.index(40)

    # r_record = []

    est_bw_config = 0
    bw_fname = "../controller/1_smooth.log"
    
    with open(bw_fname, 'r') as file:
        actual_raw_bw_list = []
        predict_raw_bw_list = []
        for line in file:
            actual, predict = line.strip().split(',')
            actual_raw_bw_list.append(float(actual))
            predict_raw_bw_list.append(float(predict))

        # left, right = 230, 350
        left, right = 350, 400
        # left, right = 50, 100

        actual_bw_list = scale_data(actual_raw_bw_list[left:right])
        predict_bw_list = scale_data(predict_raw_bw_list[left:right])

        

        # If not terminate, keep streaming
        terminate = False
        while (not terminate):

            # Step1: Update config with last round action
            res = RES_LEVEL[res_lv]
            fps = FPS_LEVEL[fps_lv]
            qp = QP_LEVEL[qp_lv]

            # # store for logging
            # last_res = res
            # last_fps = fps
            # last_qp = qp

            client.update(res, fps, qp)
            logger.info(f"NEW ACTION: [{client.config.low_resolution}]RES [{client.config.fps}]FPS [{client.config.low_qp}]QP")


            # Step2: video analytics
            accuracy, video_size, latency, num_objs, terminate = client.analyze_video()
            # accuracy, video_size, latency, num_objs, terminate = 1.0, 2, 2, 2, False


            # When terminate, start from the beginning and continue to train
            if (terminate):
                video_loops += 1
                logger.info(f"Reset video {video_loops} times!")
                client.last_end_time = 0
                client.terminate = False
                terminate = False

            rounds += 1
            timestamp += latency

            actual_bw = actual_bw_list[rounds]
            predict_bw_cur = predict_bw_list[rounds]
            write_log(fname,rounds,res,fps,qp,accuracy,actual_bw,predict_bw_cur,est_bw_config,video_size,latency,num_objs)

            # # Step3: Calculate rewards
            # reward = calc_rewards_linear(last_accuracy, accuracy, last_latency, latency,
            #                             last_bandwidth, bandwidth)
            # r_record.append(reward)
            # cum_reward = np.sum(r_record) - r_record[0]

            # # update memory
            # last_accuracy = accuracy
            # last_bandwidth = bandwidth
            # last_latency = latency


            # Step4: Controller generate new configs, config is the idx in list not value
            predict_bw_next = predict_bw_list[rounds+1]
            res_lv, fps_lv, qp_lv, next_est_bw_config = gen_new_config(predict_bw_next)

            # print(res, fps, qp)
            est_bw_config = next_est_bw_config


            # write_log(fname,rounds,last_res,last_fps,last_qp,accuracy,
            #         bandwidth,latency,num_objs,reward,cum_reward)




    return 1



if __name__ == "__main__":

    # # load configuration dictonary from command line
    # # use munch to provide class-like accessment to python dictionary
    # args = munchify(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))
    # print("Only one resolution given, running MPEG emulation")

    # config = create_config("trafficcam_1")
    #video_list = ["test2"]
    #video_list = ["drivearound_4"]
    video_list = ["trafficcam_1"]



    for i in range(len(video_list)):

        config = create_config(video_list[i])

        args = munchify(config)
        # print(args)


        # Specify NN_MODEL if continue from last time
        main(args)








# def main():
#     # print(f"{args.raw_video_name}'s duration is {args.duration}")

#     logging.basicConfig(
#         format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
#         level="info")

#     logger = logging.getLogger("central agent")
#     logger.addHandler(logging.NullHandler())

#     # Make simulation objects
#     logger.info(f"Start Training")
    
#     # create result directory
#     if not os.path.exists(SUMMARY_DIR):
#         os.makedirs(SUMMARY_DIR)


#     # Step0: inter-process communication queues
#     net_params_queues = []
#     exp_queues = []
#     for i in range(NUM_AGENTS):
#         net_params_queues.append(mp.Queue(1))
#         exp_queues.append(mp.Queue(1))


#     # Step1: create a coordinator and multiple agent processes
#     # (note: threading is not desirable due to python GIL)
#     coordinator = mp.Process(target=central_agent,
#                              args=(net_params_queues, exp_queues))
#     coordinator.start()


#     # Step2: Create a config list
#     video_list = ["trafficcam_3"]
#     config_list = []
#     for i in range(NUM_AGENTS):
#         config_list.append(create_config(video_list[i]))


#     # Step3: Init the agents and start them
#     agents = []
#     for i in range(NUM_AGENTS):
#         agents.append(mp.Process(target=agent,
#                                  args=(i, config_list[i],
#                                  net_params_queues[i],
#                                  exp_queues[i])))
    
#     for i in range(NUM_AGENTS):
#         agents[i].start()


#     # Step4: wait unit training is done
#     coordinator.join()

#     return 1


# if __name__ == "__main__":

#     # # load configuration dictonary from command line
#     # # use munch to provide class-like accessment to python dictionary
#     # args = munchify(yaml.load(sys.argv[1], Loader=yaml.SafeLoader))
#     # print("Only one resolution given, running MPEG emulation")

#     # main(args)

#     main()
