import os
import re
import logging
import sys
import csv
import copy
sys.path.append('../')

from backend.server import Server
from frontend.client import Client
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
from utils.utils import (normalize, normalize_sk)

# dirty fix
NN_MODEL = './summary/nn_model_ep_1090.ckpt'
#NN_MODEL = None
CUM_REWARD = 0
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
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
motion = 0

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
QP_LEVEL = [10, 20, 25, 30, 35, 40, 45, 50]


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
    header = ("timestamp,resolution,fps,qp,accuracy,bandwidth,latency,objnum,reward,cumreward,entropy").split(",")
    csv_writer.writerow(header)
    results_files.close()


def write_log_without(fname,timestamp,resolution,fps,qp,accuracy,bandwidth,latency,objnums,reward,cumreward,entropy):
    stats = (f"{timestamp},{resolution},{fps},{qp},{accuracy},{bandwidth},{latency},{objnums},{reward},{cumreward},{entropy}").split(",")

    results_files = open(fname, "a", newline='')
    csv_writer = csv.writer(results_files)
    csv_writer.writerow(stats)
    results_files.close()



    #

def init_loss_log_file(fname):
    # if not os.path.exists(fname):
    results_files = open(fname, "a",  newline='')
    csv_writer = csv.writer(results_files)
    header = ("epoch,td_loss,avg_reward,avg_entropy").split(",")
    csv_writer.writerow(header)
    results_files.close()

def write_loss_log(fname,epoch,td_loss,avg_reward,avg_entropy):
    stats = (f"{epoch},{td_loss},{avg_reward},{avg_entropy}").split(",")
    results_files = open(fname, "a", newline='')
    csv_writer = csv.writer(results_files)
    csv_writer.writerow(stats)
    results_files.close()

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
        mirrored_strategy = tf.distribute.MirroredStrategy()
        # Init network
        actor = a3c.ActorNetwork(sess,
                            state_dim=S_DIM, action_dim=A_DIM, o_dim = O_DIM,
                            learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                            state_dim=S_DIM,
                            learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep = 8)  # save neural net parameters


        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            logger.info("Model restored.")

        if (nn_model is not None):
            epoch = int(nn_model.split('.')[1].split('_')[3])
        else:
            epoch = 0
        timestamp = 0

        fname = os.path.join("./logs",f"{client.config.raw_video_name}_ep_{epoch}.csv")
        init_log_file(fname)
        loss_fname = os.path.join("./loss_logs",f"{client.config.raw_video_name}_loss_ep_{epoch}.csv")
        init_loss_log_file(loss_fname)


        last_res = DEFAULT_RES
        last_fps = DEFAULT_FPS
        last_qp = DEFAULT_QP
        res = DEFAULT_RES
        fps = DEFAULT_FPS
        qp = DEFAULT_QP
        max_reward = 1
        min_reward = -1

        action_vec = np.zeros((A_INFO, A_LEN))
        action_vec[0][last_res] = 1
        action_vec[1][last_fps] = 1
        action_vec[2][last_qp] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        # for cum_reward
        if (nn_model is not None):
            r_record = [CUM_REWARD]
        else:
            r_record = []


        actor_gradient_batch = []
        critic_gradient_batch = []  

        rounds = 0
        last_latency = 0
        last_accuracy = 0
        last_bandwidth = 0
        video_loops = 0


        while (True ):
            # actor.prints0()
            
            # Step1: Update config with last round action
            res = RES_LEVEL[res]
            fps = FPS_LEVEL[fps]
            qp = QP_LEVEL[qp]

            # store for logging
            last_res = res
            last_fps = fps
            last_qp = qp

            client.update(res, fps, qp)
            logger.info(f"NEW ACTION: [{client.config.low_resolution}]RES [{client.config.fps}]FPS [{client.config.low_qp}]QP")


            # Step2: Use new config for inference
            # accuracy, bandwidth, latency, num_objs, motion_degree, no_objs, terminate = client.analyze_video()
            accuracy, bandwidth, latency, num_objs, terminate = client.analyze_video()

            # When terminate, start from the beginning and continue to train
            if (terminate):
                video_loops += 1
                logger.info(f"Reset video {video_loops} times!")
                client.last_end_time = 0
                client.terminate = False
                terminate = False

            
            # if (no_objs):
            #     res = RES_LEVEL.index(res)
            #     fps = FPS_LEVEL.index(fps)
            #     qp = QP_LEVEL.index(qp)
            #     print("no objs, skip")
            #     continue

            rounds += 1

            timestamp += latency

            # Step3: Calculate rewards
            reward = calc_rewards_linear(last_accuracy, accuracy, last_latency, latency,
                                        last_bandwidth, bandwidth)
            # if(reward > max_reward):
            #     max_reward = reward
            # if(reward < min_reward):
            #     min_reward = reward
            # reward = (reward - min_reward)/(max_reward-min_reward) * 10
            # print(f"reward: {reward}")

            r_batch.append(reward)
            r_record.append(reward)

            # update memory
            last_accuracy = accuracy
            last_bandwidth = bandwidth
            last_latency = latency   

            # Step4: Update state
            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            if(1):
                if(rounds == 1):
                    max_bandwidth = 112
                    min_bandwidth = 0.5
                    max_latency = 3.5
                    min_latency = 0.1
                    max_accuracy = 1
                    min_accuracy = 0.3
                    max_res = 1
                    min_res = 0.3
                    max_fps = 30
                    min_fps = 1
                    max_qp = 20
                    min_qp = 48

                nor_bandwidth, nor_latency, nor_accuracy, nor_res, nor_fps, nor_qp, \
                max_bandwidth, max_latency, max_accuracy, max_res, max_fps, max_qp, \
                min_bandwidth, min_latency, min_accuracy, min_res, min_fps, min_qp = \
                normalize_without_motion(bandwidth, latency, accuracy, res, fps, qp,
                max_bandwidth, max_latency, max_accuracy, max_res, max_fps, max_qp,
                min_bandwidth, min_latency, min_accuracy, min_res, min_fps, min_qp)

            state[0, -1] = nor_bandwidth
            state[1, -1] = nor_latency
            state[2, -1] = nor_accuracy
            state[3, -1] = nor_res
            state[4, -1] = nor_fps
            state[5, -1] = nor_qp

            # Step6: Predict
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            # print("========state========")
            # print(state)
            # print("========action prob========")
            # print(action_prob)

            action = []
            for i in range(A_INFO):

                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                action_cumsum = np.cumsum(action_prob[i])
                act = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                print(f"action-{i}'cumsum = {action_cumsum} => act = {act}")

                # if(np.random.randint(1,11) <= 9):
                # act = action_prob[i].argmax()
                # print(f"action-{i}'prob = {action_prob[i]} => act = {act}")
                # else:
                #     act = np.random.randint(0,7)
                #     print(f"random act = {act}")
                action.append(act)

            res = action[0]
            fps = action[1]
            qp = action[2]

            # Step7: Logging and Calculate and record entropy
            # TODO: compute entropy
            entropy_list = []
            for i in range(A_INFO):
                entropy_list.append(a3c.compute_entropy(action_prob[i]))

            # print("=======entropy list=======")
            # print(entropy_list)
            entropy = np.mean(entropy_list)
            entropy_record.append(entropy)

            # cum reward
            if (nn_model is not None):
                cum_reward = np.sum(r_record)
            else:
                if (len(r_record) == 1):
                    cum_reward = r_record[0]
                else:
                    cum_reward = np.sum(r_record) - r_record[0]

            write_log(fname,rounds,last_res,last_fps,last_qp,accuracy,
                    bandwidth,latency,motion_degree,num_objs,reward,cum_reward,entropy)


            # if len(r_batch) >= TRAIN_SEQ_LEN or terminate:  # do training once
            if len(r_batch) >= TRAIN_SEQ_LEN:  # do training once

                logger.info("Perform TRAINING!!!")

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(s_batch=np.stack(s_batch[1:], axis=0),  # ignore the first chuck
                                          a_batch=np.vstack(a_batch[1:]),  # since we don't have the
                                          r_batch=np.vstack(r_batch[1:]),  # control over it
                                          terminal=terminate, actor=actor, critic=critic)
                # print("actor gradient")
                # print(actor_gradient)
                # print("critic gradient")
                # print(critic_gradient)
                # print("td_batch")
                # print(td_batch)

                td_loss = np.mean(td_batch)
                avg_reward = np.mean(r_batch)
                avg_entropy = np.mean(entropy_record)


                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                logger.info("========================================")
                logger.info(f"Epoch {epoch}")
                logger.info(f"TD_loss:{td_loss} Avg_reward:{avg_reward} Avg_entropy:{avg_entropy}")
                logger.info("========================================")
                write_loss_log(loss_fname,epoch,td_loss,avg_reward,avg_entropy)



                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: td_loss,
                    summary_vars[1]: np.mean(r_batch),
                    summary_vars[2]: np.mean(entropy_record)
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()

                entropy_record = []

                if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:

                    assert len(actor_gradient_batch) == len(critic_gradient_batch)
                    # assembled_actor_gradient = actor_gradient_batch[0]
                    # assembled_critic_gradient = critic_gradient_batch[0]
                    # assert len(actor_gradient_batch) == len(critic_gradient_batch)
                    # for i in range(len(actor_gradient_batch) - 1):
                    #     for j in range(len(actor_gradient)):
                    #         assembled_actor_gradient[j] += actor_gradient_batch[i][j]
                    #         assembled_critic_gradient[j] += critic_gradient_batch[i][j]
                    # actor.apply_gradients(assembled_actor_gradient)
                    # critic.apply_gradients(assembled_critic_gradient)
                    print("Start to apply gradients:")
                    for i in range(len(actor_gradient_batch)):
                        # print(f"actor{i}'s = {actor_gradient_batch[i]}")
                        # print(f"critic{i}'s = {critic_gradient_batch[i]}")

                        actor.apply_gradients(actor_gradient_batch[i])
                        critic.apply_gradients(critic_gradient_batch[i])

                    actor_gradient_batch = []
                    critic_gradient_batch = []

                    epoch += 1
                    if epoch % MODEL_SAVE_INTERVAL == 0:
                        # Save the neural net parameters to disk.
                        save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                               str(epoch) + ".ckpt")
                                               
                        logger.info("Model saved in file: %s" % save_path)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

            if terminate:
                last_res = DEFAULT_RES
                last_fps = DEFAULT_FPS
                last_qp = DEFAULT_QP
                res = DEFAULT_RES
                fps = DEFAULT_FPS
                qp = DEFAULT_QP   # use the default action here

                action_vec = np.zeros((A_INFO, A_LEN))
                action_vec[0][res] = 1
                action_vec[1][fps] = 1
                action_vec[2][qp] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append([action_vec])

            else:
                s_batch.append(state)
                # logger.info("Put current state back to s_batch")

                action_vec = np.zeros((A_INFO, A_LEN))
                action_vec[0][res] = 1
                action_vec[1][fps] = 1
                action_vec[2][qp] = 1
                a_batch.append([action_vec])
                # logger.info("Put act vec back to a_batch")








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

