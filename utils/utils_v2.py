import re
import os
import csv
import shutil
import subprocess
import numpy as np
from sklearn import preprocessing
# import cv2 as cv
# import networkx
import random
# from networkx.algorithms.components.connected import connected_components


def normalize_sk(state, mode=0): # state is a 2D-array
    res = np.zeros((state.shape[0], state.shape[1]))

    for i in range(state.shape[0]):
        data = np.reshape(state[i], (state.shape[1], 1))
        # mode=0, 缩放到均值为0，方差为1（Standardization——StandardScaler()）
        if (mode == 0):
            standard_scaler_data = preprocessing.StandardScaler().fit_transform(data)

        # mode=1, 缩放到0和1之间（Standardization——MinMaxScaler()）
        elif (mode == 1):
            standard_scaler_data = preprocessing.MinMaxScaler().fit_transform(data)

        # mode=2, 缩放到-1和1之间（Standardization——MaxAbsScaler()）
        elif (mode == 2):
            standard_scaler_data = preprocessing.MaxAbsScaler().fit_transform(data)

        # mode=3, 缩放到0和1之间，保留原始数据的分布（Normalization——Normalizer()）
        elif (mode == 3):
            standard_scaler_data = preprocessing.Normalizer().fit_transform(data)

        # mode=4, RobustScaler()
        elif (mode == 4):
            standard_scaler_data = preprocessing.RobustScaler().fit_transform(data)

        res[i] = np.reshape(standard_scaler_data, (1, state.shape[1]))
        # print(f"========dim-{i}=========")
        # print(data)
        # print(res[i])

    return res


def normalize_without_motion(bandwidth, latency, accuracy, res, fps, qp,
              max_bandwidth, max_latency, max_accuracy, max_res, max_fps, max_qp,
              min_bandwidth, min_latency, min_accuracy, min_res, min_fps, min_qp ):

    if(bandwidth > max_bandwidth) : max_bandwidth = bandwidth
    if(bandwidth < min_bandwidth) : min_bandwidth = bandwidth
    if(latency > max_latency) : max_latency = latency
    if(latency < min_latency) : min_latency = latency
    if(accuracy > max_accuracy) : max_accuracy = accuracy
    if(accuracy < min_accuracy) : min_accuracy = accuracy
    if(res > max_res) : max_res = res
    if(res < min_res) : min_res = res
    if(fps > max_fps) : max_fps = fps
    if(fps < min_fps) : min_fps = fps
    if(qp > max_qp) : max_qp = qp
    if(qp < min_qp) : min_qp = qp


    nor_bandwidth = (bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth)
    nor_latency = (latency - min_latency) / (max_latency - min_latency)
    nor_accuracy = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)
    nor_res = (res - min_res) / (max_res - min_res)
    nor_fps = (fps - min_fps) / (max_fps - min_fps)
    nor_qp = (qp - min_qp) / (max_qp - min_qp)

    return nor_bandwidth, nor_latency, nor_accuracy, nor_res, nor_fps, nor_qp, \
           max_bandwidth, max_latency, max_accuracy, max_res, max_fps, max_qp, \
           min_bandwidth, min_latency, min_accuracy, min_res, min_fps, min_qp

def normalize(bandwidth, latency, accuracy, res, fps, qp, motion,
                max_bandwidth, max_latency, max_accuracy, max_res, max_fps, max_qp, max_motion,
                    min_bandwidth, min_latency, min_accuracy, min_res, min_fps, min_qp, min_motion ):

    if(bandwidth > max_bandwidth) : max_bandwidth = bandwidth
    if(bandwidth < min_bandwidth) : min_bandwidth = bandwidth
    if(latency > max_latency) : max_latency = latency
    if(latency < min_latency) : min_latency = latency
    if(accuracy > max_accuracy) : max_accuracy = accuracy
    if(accuracy < min_accuracy) : min_accuracy = accuracy
    if(res > max_res) : max_res = res
    if(res < min_res) : min_res = res
    if(fps > max_fps) : max_fps = fps
    if(fps < min_fps) : min_fps = fps
    if(qp > max_qp) : max_qp = qp
    if(qp < min_qp) : min_qp = qp
    if(motion > max_motion) : max_motion = motion
    if(motion < min_motion) : min_motion = motion

    nor_bandwidth = (bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth)
    nor_latency = (latency - min_latency) / (max_latency - min_latency)
    nor_accuracy = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)
    nor_res = (res - min_res) / (max_res - min_res)
    nor_fps = (fps - min_fps) / (max_fps - min_fps)
    nor_qp = (qp - min_qp) / (max_qp - min_qp)
    nor_motion = (motion - min_motion) / (max_motion - min_motion)

    return nor_bandwidth, nor_latency, nor_accuracy, nor_res, nor_fps, nor_qp, nor_motion, \
           max_bandwidth, max_latency, max_accuracy, max_res, max_fps, max_qp, max_motion, \
           min_bandwidth, min_latency, min_accuracy, min_res, min_fps, min_qp, min_motion

def copy(src, target, filelist):
    if os.path.isdir(src) and os.path.isdir(target):
        if len(filelist) == 0:
            filelist = os.listdir(src)
            
        for file in filelist:
            source = os.path.join(src,file)
            shutil.copy(source, target)
    else:
        print("dir not correct")



def get_folder_size(folder_path):
    # folder_path = r'/Users/alanzyt/Desktop/coding/pensieve_ac/sim/results'
    full_size = 0
    for parent, dirs, files in os.walk(folder_path):
        full_size = sum(os.path.getsize(os.path.join(parent, file)) for file in files)
    # print(full_size, "%.2f MB" % (full_size/1024/1024))

    return full_size


def generate_images(gt_sample_batch_size, video_name, fps, qp, resolution, start_time, duration):
    dataset_dir = "../dataset"
    video_name_dir_gt = f"{video_name}_gt"
    video_name_dir_gt_tmp = f"{video_name}_gt_tmp"
    video_name_dir_fps = f"{video_name}_fps{fps}"
    video_name_dir_fps_tmp = f"{video_name}_fps{fps}_tmp"

    images_path_gt = os.path.join(dataset_dir, video_name_dir_gt, 'src')
    images_path_gt_tmp = os.path.join(dataset_dir, video_name_dir_gt_tmp, 'src')
    images_path_fps = os.path.join(dataset_dir, video_name_dir_fps, 'src')
    images_path_fps_tmp = os.path.join(dataset_dir, video_name_dir_fps_tmp, 'src')


    # handling directory
    if not os.path.exists(images_path_gt):
        os.makedirs(images_path_gt)

    for fname in os.listdir(images_path_gt):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path_gt, fname))

    if not os.path.exists(images_path_gt_tmp):
        os.makedirs(images_path_gt_tmp)

    for fname in os.listdir(images_path_gt_tmp):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path_gt_tmp, fname))


    if not os.path.exists(images_path_fps):
        os.makedirs(images_path_fps)

    for fname in os.listdir(images_path_fps):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path_fps, fname))

    if not os.path.exists(images_path_fps_tmp):
        os.makedirs(images_path_fps_tmp)

    for fname in os.listdir(images_path_fps_tmp):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path_fps_tmp, fname))

    

    # Step1: [DECODE] decode raw video with best config (fps30) between [start_time, start_time+duration]
    encoded_vid_path = os.path.join(dataset_dir, f"{video_name}.mp4")
    extracted_images_path_gt = os.path.join(images_path_gt_tmp, "%010d.png")

    # extract frames according to FPS_GT=30
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                    #   "-qp", "20",
                                    #   "-qp", f"{qp}",
                                    #   "-g", "8",
                                      "-ss", f"{start_time}",
                                      "-t", f"{duration}",
                                      "-q:v", "2",
                                    #   "-r", "30", # default 30
                                    #   "-vsync", "0", 
                                      "-start_number", "0",
                                      extracted_images_path_gt],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)

    
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()
    
    else:
        # Step2: [ENCODE] Apply qp and resoulution to raw image frames to encode the video
        start_id = 0
        number_of_frames = len([x for x in os.listdir(images_path_gt_tmp) if "png" in x])
        #print(f"******1st round images count: {number_of_frames}")
        encoded_vid_path = os.path.join(images_path_fps_tmp, "temp.mp4")
        scale = f"scale=trunc(iw*{resolution}/2)*2:trunc(ih*{resolution}/2)*2"

        encoding_result = subprocess.run(["ffmpeg", "-y",
                                            "-loglevel", "error",
                                            "-start_number", str(start_id),
                                            '-i', f"{images_path_gt_tmp}/%010d.png",
                                            "-vcodec", "libx264",
                                            # "-g", "15",
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


        # Step3: [DECODE] Extract frame images from modified video
        if not os.path.isdir(images_path_fps_tmp):
            print("%s not a valid directory", images_path_fps_tmp)
            return

        for fname in os.listdir(images_path_fps_tmp):
            if "png" not in fname:
                continue
            else:
                os.remove(os.path.join(images_path_fps_tmp, fname))

        encoded_vid_path = os.path.join(images_path_fps_tmp, "temp.mp4")
        extracted_images_path_gt = os.path.join(images_path_fps_tmp, "%010d.png")

        decoding_result = subprocess.run(["ffmpeg", "-y",
                                        "-i", encoded_vid_path,
                                        "-pix_fmt", "yuvj420p",
                                        # "-g", "8", 
                                        "-q:v", "2",
                                        #   "-r", f"{fps}",
                                        "-vsync", "0", 
                                        "-start_number", "0",
                                        extracted_images_path_gt],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)



        # Step 4.1: start to sample frames according to sample batch size FOR "GT SET"
        flist = []
        pick_list = []
        length = 30

        num_of_frames_gt = len([x for x in os.listdir(images_path_gt_tmp) if "png" in x])
        times = int(num_of_frames_gt / 30)

        for i in range(times):
            count = 0
            while (count < gt_sample_batch_size) :
                pick = random.randint(length * i, length * (i+1) - 1)
                if (pick not in pick_list):
                    pick_list.append(pick)
                    count += 1

        for i in range(len(pick_list)):
            fname = "%010d.png"%pick_list[i]
            # print(fname)
            flist.append(fname)


        # COPY THE SELECTED FRAMES TO NEW DIR
        copy(images_path_gt_tmp, images_path_gt, flist)
        num_of_frames_gt = len([x for x in os.listdir(images_path_gt) if "png" in x])


        # rename serializably
        fnames = sorted(
            [os.path.join(images_path_gt, name)
            for name in os.listdir(images_path_gt) if "png" in name])

        # print(fnames)

        # rename
        count = 0
        for fname in fnames:
            os.rename(fname, os.path.join(images_path_gt, f"{str(count).zfill(10)}.png"))
            count += 1

        fnames = sorted(
            [os.path.join(images_path_gt, name)
            for name in os.listdir(images_path_gt) if "png" in name])





        # Step 4.2: start to sample frames according to fps FOR "FPS SET"
        flist = []
        pick_list = []
        divisible = [1,2,5,10,15,20,25,30]
        
        if (fps in divisible):
            fps_gt = 30
            ratio = fps_gt / fps

            if ratio == 1:
                flist = os.listdir(images_path_fps_tmp)
            
            else:
                num_of_frames_fps = len([x for x in os.listdir(images_path_fps_tmp) if "png" in x])
                times = int(num_of_frames_fps / 30)
                length = 30
                batch_size = fps

                # print("batch size:", batch_size)

                for i in range(times):
                    count = 0
                    while (count < batch_size) :
                        pick = random.randint(length * i, length * (i+1) - 1)
                        if (pick not in pick_list):
                            pick_list.append(pick)
                            count += 1

                # print(f"len: {len(pick_list)}")
                # print(pick_list)



                for i in range(len(pick_list)):
                    fname = "%010d.png"%pick_list[i]
                    # print(fname)
                    flist.append(fname)

                # print(pick_list)
                # print(flist)


        # # FPS = 20/25, sample randomly
        # else:
        #     num_of_frames_gt = len([x for x in os.listdir(images_path_gt_tmp) if "png" in x])
        #     desired_len = num_of_frames_gt * fps / 30

        #     while (len(pick_list) < desired_len):
        #         pick = random.randint(1, num_of_frames_gt) - 1
        #         if (pick not in pick_list):
        #             pick_list.append(pick)

        #     for i in range(len(pick_list)):
        #         fname = "%010d.png"%pick_list[i]
        #         print(fname)
        #         flist.append(fname)

        #     print(pick_list)
        #     print(flist)


        # COPY THE SELECTED FRAMES TO NEW DIR
        copy(images_path_fps_tmp, images_path_fps, flist)
        num_of_frames_fps = len([x for x in os.listdir(images_path_fps) if "png" in x])


        # rename serializably
        fnames = sorted(
            [os.path.join(images_path_fps, name)
            for name in os.listdir(images_path_fps) if "png" in name])

        # print(fnames)

        # rename
        count = 0
        for fname in fnames:
            os.rename(fname, os.path.join(images_path_fps, f"{str(count).zfill(10)}.png"))
            count += 1

        fnames = sorted(
            [os.path.join(images_path_fps, name)
            for name in os.listdir(images_path_fps) if "png" in name])

        # print(fnames)

        # for fid, fname in fids_mapping:
        #     os.rename(os.path.join(f"{fname}_temp"),
        #             os.path.join(images_path, f"{str(fid).zfill(10)}.png"))


        # num_of_frames_gt = len([x for x in os.listdir(images_path_gt_tmp) if "png" in x])
        # print(f"Generate {num_of_frames_gt} images for GT batch")
        # print("Generate images for this batch:")
        # print("fps:", fps, "start:", start_time, "duration:", duration, "num of frames:", num_of_frames_fps)

    return images_path_fps, images_path_gt, number_of_frames
 

def sample_images(video_name, fps, fps_gt=30):
    dataset_dir = "../dataset"
    video_name_gt = f"{video_name}_gt"
    images_path_gt = os.path.join(dataset_dir, video_name_gt, 'src')

    number_of_frames_gt = len(
            [f for f in os.listdir(images_path_gt) if ".png" in f])

    images_path = os.path.join(dataset_dir, video_name, 'src')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for fname in os.listdir(images_path):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))

    fps_diff = fps_gt - fps


