import numpy as np
import math

# Reward
ACC_THRE = 0.7 # accuracy threshold
LAT_PENALTY = 40
#BW_PENALTY = 9
BW_PENALTY = 10



ACC_PENALTY = 1
ACC_WEIGHT = 1
ACC_REWARD = 1
LAT_THRE = 30 # latency threshold
ACC_LOW_REWARD = 3
ACC_HIGH_REWARD = 6
ACC_LOW_PENALTY = 4
ACC_HIGH_PENALTY = 6

LAT_REWARD = 2

BW_REWARD = 2
ENE_PENALTY = 3
ENE_REWARD = 2
LOG_COMPENSATE = 100

RES_LEVEL = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
FPS_LEVEL = [30, 25, 20, 15, 10, 5, 2, 1]
QP_LEVEL = [20, 24, 28, 32, 36, 40, 44, 48]



def calc_rewards_linear(last_accuracy, accuracy, last_latency, latency, 
        last_bandwidth, bandwidth, acc_method=0, bw_method=1, is_log=1, last_process_time=0, process_time=0):


    accuracy_reward = 0
    latency_reward = 0
    bandwidth_reward = 0
    energy_reward = 0

    # ACCURACY
    if (acc_method == 0): # accuracy calculate without compare with threshold
        
        if (is_log):
            if(accuracy - ACC_THRE  ==  0):
                accuracy_reward = 0
            elif(accuracy - ACC_THRE < 0):
                accuracy_reward = -1 * np.log((accuracy - ACC_THRE)*-100) * LOG_COMPENSATE
            else:
                accuracy_reward = np.log((accuracy - ACC_THRE)*100) * LOG_COMPENSATE #Max:1.47 * 100

        else:
            accuracy_lag = (accuracy - ACC_THRE)

            if (accuracy_lag >= 0):
                accuracy_reward = abs(accuracy_lag * ACC_REWARD) * ACC_WEIGHT
            else:
                accuracy_reward = (-1) * abs(accuracy_lag * ACC_PENALTY) * ACC_WEIGHT

    elif (acc_method == 1):
        accuracy_reward = accuracy * ACC_REWARD * ACC_WEIGHT
    

    # BANDWIDTH and LATENCY
    if (bw_method == 0): # bandwidth & latency count directly
        # latency
        latency_diff = latency - last_latency
        if (latency_diff < 0):
            latency_reward = abs(latency_diff * LAT_REWARD)
        else:
            latency_reward = (-1) * abs(latency_diff * LAT_PENALTY)

        # bandwidth
        bandwidth_diff = bandwidth - last_bandwidth
        if (bandwidth_diff < 0):
            bandwidth_reward = abs(bandwidth_diff * BW_REWARD)
        else:
            bandwidth_reward = (-1) * abs(bandwidth_diff * BW_PENALTY)  

    elif (bw_method == 1): # bandwidth & latency count by comparing with last round result
        # latency
        latency_reward = -1 * latency * LAT_PENALTY

        # bandwidth
        bandwidth_reward = -1 * bandwidth * BW_PENALTY

    # ENERGY
    # process_time_diff = process_time - last_process_time
    # if (process_time_diff < 0):
    #     energy_reward = abs(process_time_diff * ENE_REWARD)
    # else:
    #     energy_reward = (-1) * abs(process_time_diff * BW_PENALTY)

    # TOTAL REWARD = accuracy_reward + latency_reward
    # + bandwidth_reward + energy_reward
    reward = accuracy_reward + latency_reward + bandwidth_reward

    return reward





    # # Method 3:
    # accuracy_lag = (accuracy - ACC_THRE)
    # if (accuracy_lag > 0):
    #     accuracy_reward = abs(accuracy_lag * ACC_REWARD) * ACC_WEIGHT
    # else:
    #     accuracy_reward = (-1) * abs(accuracy_lag * ACC_PENALTY) * ACC_WEIGHT


    # Method 2:
    # accuracy_diff = (accuracy - last_accuracy)
    # accuracy_lag = (accuracy - ACC_THRE)
    # if (accuracy_lag < 0): # not accurate
    #     if (accuracy_diff > 0): # low reward
    #         accuracy_reward = abs(accuracy_lag * ACC_LOW_REWARD) * ACC_WEIGHT
    #     else:                   # high penalty
    #         accuracy_reward = (-1) * abs(accuracy_lag * ACC_HIGH_PENALTY) * ACC_WEIGHT
    # else: # accurate
    #     if (accuracy_diff > 0): # low penalty
    #         accuracy_reward = (-1) * abs(accuracy_lag * ACC_LOW_PENALTY) * ACC_WEIGHT
    #     else:                   # high reward
    #         accuracy_reward = abs(accuracy_lag * ACC_HIGH_REWARD) * ACC_WEIGHT


    # Method 1:
    # accuracy_diff = (accuracy - last_accuracy)
    # if (accuracy < ACC_THRE and accuracy_diff >= 0):
    #     accuracy_reward = abs(accuracy_diff * ACC_LOW_REWARD)
    # elif ((accuracy > ACC_THRE and accuracy_diff <= 0)):
    #     accuracy_reward = abs(accuracy_diff * ACC_HIGH_REWARD)
    # elif ((accuracy > ACC_THRE and accuracy_diff >= 0)):
    #     accuracy_reward = (-1) * abs(accuracy_diff * ACC_LOW_PENALTY)
    # else:
    #     accuracy_reward = (-1) * abs(accuracy_diff * ACC_HIGH_PENALTY)




    # LATENCY
    # Method 2
    # latency_reward = latency * LAT_PENALTY

    # Method 1
    # latency_diff = latency - last_latency
    # # if (   (latency < LAT_THRE and latency_diff > 0)
    # #     or (latency > LAT_THRE and latency_diff < 0)):
    # if (latency_diff < 0):
    #         latency_reward = abs(latency_diff * LAT_REWARD)
    # else:
    #     latency_reward = (-1) * abs(latency_diff * LAT_PENALTY)

    # BANDWIDTH
    # Method 2
    # bandwidth_reward = bandwidth * BW_PENALTY

    # Method 1
    # bandwidth_diff = bandwidth - last_bandwidth
    # if (bandwidth_diff < 0):
    #     bandwidth_reward = abs(bandwidth_diff * BW_REWARD)
    # else:
    #     bandwidth_reward = (-1) * abs(bandwidth_diff * BW_PENALTY)






# TODO: finish log reward caluculation
def calc_rewards_log(last_accuracy, accuracy, last_latency, latency, 
        last_process_time, process_time, bandwidth, est_bandwidth):

    accuracy_reward = 0
    latency_reward = 0
    bandwidth_reward = 0
    energy_reward = 0
        
    # ACCURACY
    accuracy_diff = accuracy - last_accuracy
    if (   (accuracy < ACC_THRE and accuracy_diff > 0)
        or (accuracy > ACC_THRE and accuracy_diff < 0)):
            accuracy_reward = accuracy_diff * ACC_REWARD
    else:
        accuracy_reward = accuracy_diff * ACC_PENALTY

    # LATENCY
    latency_diff = latency - last_latency
    if (   (latency < ACC_THRE and latency_diff > 0)
        or (latency > ACC_THRE and latency_diff < 0)):
            latency_reward = latency_diff * LAT_REWARD
    else:
        latency_reward = latency_diff * LAT_PENALTY

    # BANDWIDTH
    bandwidth_diff = bandwidth - est_bandwidth
    if (bandwidth_diff < 0):
        bandwidth_reward = bandwidth_diff * BW_REWARD
    else:
        bandwidth_reward = bandwidth_diff * BW_PENALTY

    # ENERGY
    process_time_diff = process_time - last_process_time
    if (process_time_diff < 0):
        energy_reward = process_time_diff * ENE_REWARD
    else:
        energy_reward = process_time_diff * BW_PENALTY

    # TOTAL REWARD = accuracy_reward + latency_reward 
    # + bandwidth_reward + energy_reward
    reward = accuracy_reward + latency_reward \
        + bandwidth_reward + energy_reward

    return reward