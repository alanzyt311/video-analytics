import os
import re
import logging
import sys
import csv
import copy
sys.path.append('../')

from backend.server import Server
from frontend.client_read_gt import Client
# from controller import a2c, a3c, rewards
from dds_utils import (Results, ServerConfig, read_results_dict,
                       evaluate, write_stats, modify_results,
                       get_fid_by_fname, get_fid_by_results,
                       get_duration)
from controller import a3c
# from controller.a3c import (ActorNetwork, CriticNetwork)
from controller.rewards import (calc_rewards_linear)
import time
from munch import *
import yaml
import multiprocessing as mp
import tensorflow.compat.v1 as tf
import numpy as np
from utils.utils import (normalize, normalize_without_motion, normalize_sk)
from motion import motion
# dirty fix
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#===============================================================
""" 

State space (S) 7x8 Matrix: 

1.Bandwidth measurement of past k segments
2.Latency measurement of past k segments

3.Accuracy of the last segment

(Configuration of last segment)
4.FPS of last segment
5.Resolution of last segment
6.QP of last segment

7.Motion vector of next segment


"""
CUM_REWARD = 0
if(motion):
    S_INFO = 7 # Dimension
else:
    S_INFO = 6
S_LEN = 8  # take how many frames in the past
S_DIM = [S_INFO,S_LEN]

"""

Action space (A) 3x8 Matrix: 

1. Resolution
2. FPS
3. QP

action each level range in [0,7]
"""

A_INFO = 3
A_LEN = 8 # 8 levels each
A_DIM = [3,8]
O_DIM = [3,8]
RES_LEVEL = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
FPS_LEVEL = [30, 25, 20, 15, 10, 5, 2, 1]
QP_LEVEL = [20, 24, 28, 32, 36, 40, 44, 48]


# default config setting
# DEFAULT_RES = 4
# DEFAULT_FPS = 4
# DEFAULT_QP = 4
DEFAULT_RES = 4
DEFAULT_FPS = 4
DEFAULT_QP = 4
DEFAULT_ACC = 0
DEFAULT_BW = 0
DEFAULT_LAT = 0



NUM_AGENTS = 1
TRAIN_SEQ_LEN = 4  # take as a train batch
MODEL_SAVE_INTERVAL = 10
RANDOM_SEED = 42
RAND_RANGE = 1000
GRADIENT_BATCH_SIZE = 2
SUMMARY_DIR = './summary'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'




def load_configuration():
    # """read configuration information from yaml file
    #
    # Returns:
    # dict: information of the yaml file
    # """
    with open('configuration.yml', 'r') as config:
        config_info = yaml.load(config, Loader=yaml.FullLoader)

    return config_info


# Create config for each agent for a single video
def create_config(video_name):

    all_instances = load_configuration()

    single_instance = all_instances['default']
    # single_instance = copy.deepcopy(all_instances['default'])

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


    single_instance['duration'] = duration
    single_instance['video_name'] = f'results/{result_file_name}'
    single_instance['ground_truth'] = f'results/{video_name}_gt'
    single_instance['tested_images_path'] = f'{tested_images_dir}'
    single_instance['gt_images_path'] = f'{gt_images_dir}'
    single_instance['outfile'] = 'stats'

    return single_instance


# About logging
def init_log_file(fname):
    # if not os.path.exists(fname):
    results_files = open(fname, "a",  newline='')
    csv_writer = csv.writer(results_files)
    header = ("timestamp,resolution,fps,qp,accuracy,bandwidth,latency,objnum,reward,cumreward").split(",")
    csv_writer.writerow(header)
    results_files.close()

def write_log_without_motion(fname,timestamp,resolution,fps,qp,accuracy,bandwidth,latency,objnums,reward,cumreward):
    stats = (f"{timestamp},{resolution},{fps},{qp},{accuracy},{bandwidth},{latency},{objnums},{reward},{cumreward}").split(",")
    results_files = open(fname, "a", newline='')
    csv_writer = csv.writer(results_files)
    csv_writer.writerow(stats)
    results_files.close()



class cost_group:
    def __init__(self):
        self.res = 0
        self.fps = 0
        self.qp = 0
        self.cost = 0
class top_k:
    def __init__(self):
        self.res_list = []
        self.res_cost = []
        self.fps_list = []
        self.fps_cost = []
        self.qp_list = []
        self.qp_cost = []
        self.res = self.fps = self.qp = 0

    def sort(self):
        cost_list = []
        for i in range(len(self.res_list)):
            for j in range(len(self.fps_list)):
                for k in range(len(self.qp_list)):
                    group = cost_group()
                    group.cost = self.res_cost[i] + self.fps_cost[j] + self.qp_cost[k]
                    group.res = i
                    group.fps = j
                    group.qp = k
                    cost_list.append(group)
        cost_list.sort(key = lambda t:t.cost)
        return cost_list



def main(args):


    np.random.seed(RANDOM_SEED)

    ################### ENV INIT ###################
    logging.basicConfig(
        format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
        level=args.verbosity.upper())

    logger = logging.getLogger("RL")
    logger.addHandler(logging.NullHandler())

    # Make simulation objects
    logger.info(f"Starting server with high threshold of "
                f"{args.high_threshold} low threshold of "
                f"{args.low_threshold} tracker length of "
                f"{args.tracker_length}")

    config = args
    server = Server(config)


    logger.warning(f"Running in MPEG mode with resolution "
                   f"{args.low_resolution} on {args.video_name} with duration {args.duration}")

    logger.info("Starting client")
    client = Client(args.hname, config, server)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    ################### AGENT INIT ###################
    with tf.Session() as sess:
        #init_log_file(fname)

        # Init network
        sess.run(tf.global_variables_initializer())

        topk = top_k()
        epoch = 0
        timestamp = 0

        fname = os.path.join("./logs",f"{client.config.raw_video_name}_chameleon_window.csv")
        init_log_file(fname)

        rounds = 0
        last_latency = 0
        last_accuracy = 0
        last_bandwidth = 0
        video_loops = 0
        make_topk = 1
        cum_reward = 0
        level = 0
        flag = 1
        window = 4
        while (True ):
            #make top_k
            if(make_topk):
                print("*****************start to make top-k*****************")
                for i in range(8):
                    res = i
                    client.update(RES_LEVEL[res], FPS_LEVEL[0], QP_LEVEL[0])
                    accuracy, bandwidth, latency, num_objs, motion, terminate = client.analyze_video()
                    if(latency > 10): latency -= 10
                    if(accuracy >= 0.8):
                        topk.res_list.append(res)
                        topk.res_cost.append(bandwidth + latency*20)
                    else: break
                    client.last_end_time = 0

                for i in range(8):
                    fps = i
                    client.update(RES_LEVEL[0], FPS_LEVEL[fps], QP_LEVEL[0])
                    accuracy, bandwidth, latency, num_objs, motion, terminate = client.analyze_video()
                    if(latency > 10): latency -= 10
                    if(accuracy >= 0.8):
                        topk.fps_list.append(fps)
                        topk.fps_cost.append(bandwidth + latency*20)
                    else: break
                    client.last_end_time = 0

                for i in range(8):
                    qp = i
                    client.update(RES_LEVEL[0], FPS_LEVEL[0], QP_LEVEL[qp])
                    accuracy, bandwidth, latency, num_objs, motion, terminate = client.analyze_video()
                    if(latency > 10): latency -= 10
                    if(accuracy >= 0.8):
                        topk.qp_list.append(qp)
                        topk.qp_cost.append(bandwidth + latency*20)
                    else: break
                    client.last_end_time = 0

                make_topk = 0
                cfg_list = topk.sort()
                print("*****************make top-k finish*****************")
                for i in range(len(cfg_list)):
                    print(f"res = {RES_LEVEL[cfg_list[i].res]}, fps = {FPS_LEVEL[cfg_list[i].fps]}, qp = {QP_LEVEL[cfg_list[i].qp]}")
                    print(f"cost = {cfg_list[i].cost}")
                    if(i > 10): break


            res = cfg_list[level].res
            fps = cfg_list[level].fps
            qp = cfg_list[level].qp

            res = RES_LEVEL[res]
            fps = FPS_LEVEL[fps]
            qp = QP_LEVEL[qp]

            # store for logging
            last_res = res
            last_fps = fps
            last_qp = qp


            client.update(res, fps, qp)
            logger.info(f"NEW ACTION: [{client.config.low_resolution}]RES [{client.config.fps}]FPS [{client.config.low_qp}]QP")
            accuracy, bandwidth, latency, num_objs, motion, terminate = client.analyze_video()

            if(window == 4):
                window = 0
                if(accuracy < 0.7):
                    up = int((0.7-accuracy) * 10)
                    if(level + up < len(cfg_list)):
                        level += up
                        print(f"level + {up}")
                    else:
                        level = len(cfg_list) - 1
                else:
                    if(level != 0 ):
                        level -=1
                        print("level - 1")
            else:
                window += 1
            # When terminate, start from the beginning and continue to train
            if (terminate):
                video_loops += 1
                logger.info(f"Reset video {video_loops} times!")
                client.last_end_time = 0
                client.terminate = False

            rounds += 1

            timestamp += latency

            # Step3: Calculate rewards
            reward = calc_rewards_linear(last_accuracy, accuracy, last_latency, latency,
                                         last_bandwidth, bandwidth)

            cum_reward += reward

            write_log_without_motion(fname,rounds,last_res,last_fps,last_qp,accuracy,
                                         bandwidth,latency,num_objs,reward,cum_reward)





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

        main(args)

