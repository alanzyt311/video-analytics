import re
import os
import csv
import shutil
import subprocess
import numpy as np
import cv2 as cv
import networkx
from networkx.algorithms.components.connected import connected_components
import pandas as pd
import math
BW_LENGTH = 8

def mv_extractor(bw_list):
    print("Estimate bandwidth...")
    est_bw = 0
    return est_bw


def get_mvs_bak(fname):


    cmd = f"./extract_mvs {fname}"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    with open("test.csv", "w") as csvfile:
        writer = csv.writer(csvfile)

        for line in out.splitlines():

            csvfile.write(str(line, encoding = "gbk"))
            csvfile.write("\n")


def get_motion_bak(fname):

    df = pd.read_csv(fname)
    # print(df.head())

    srcx = df['srcx']
    srcy = df['srcy']
    dstx = df['dstx']
    dsty = df['dsty']

    total_srcx = {}
    total_srcy = {}
    total_dstx = {}
    total_dsty = {}


    dist_sq_list = []
    dist_list = []
    id_list = []
    for i in range(len(srcx)):
        id = df['framenum'][i]
        # if id != 2:
        #     break

        if id in id_list:
            total_srcx[id].append(srcx[i])
            total_srcy[id].append(srcy[i])
            total_dstx[id].append(dstx[i])
            total_dsty[id].append(dsty[i])
        else:
            id_list.append(id)
            total_srcx[id] = [srcx[i]]
            total_srcy[id] = [srcy[i]]
            total_dstx[id] = [dstx[i]]
            total_dsty[id] = [dsty[i]]

    total_mv_sq = []
    total_mv = []

    for id in range(len(id_list)):
        srcx_list = total_srcx[id_list[id]]
        srcy_list = total_srcy[id_list[id]]
        dstx_list = total_dstx[id_list[id]]
        dsty_list = total_dsty[id_list[id]]

        dist_sq_list = []
        dist_list = []

        for i in range(len(srcx_list)):

            dist_sq = math.pow((int(dsty_list[i]) - int(srcy_list[i])), 2) \
                      + math.pow((int(dstx_list[i]) - int(srcx_list[i])), 2)
            dist = math.pow(dist_sq, 0.5)

            dist_sq_list.append(dist_sq)
            dist_list.append(dist)

        mv_sq = sum(dist_sq_list)
        mv = sum(dist_list)

        total_mv_sq.append(mv_sq)
        total_mv.append(mv)
        print(f"FRAME ID[{id_list[id]}] TYPE[{frame_type_list[id]}] MV[{mv}]")

    avg_motion = round(sum(total_mv)/len(total_mv), 2)
    print(f"Avg motion in {id_list[-1]} frames = {avg_motion}")

    return total_mv, avg_motion

    # print(total_mv_sq)
    # print(total_mv)


def get_mvs(fname, outfile):


    cmd = f"../mvs/extract_mvs {fname}"
    # print(fname)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    # print(out)
    # print(err)


    with open(outfile, "w") as csvfile:
        writer = csv.writer(csvfile)

        for line in out.splitlines():

            csvfile.write(str(line, encoding = "gbk"))
            csvfile.write("\n")

    print("write mv file done!")



def get_motion(fname,start_time,duration):


    # Step 1: slice video
    dataset_dir = "../dataset"
    result_dir = "../mv_results"
    encoded_vid_path = os.path.join(dataset_dir, f"{fname}.mp4")
    mvs_dir = os.path.join(result_dir, f"{fname}")
    out_vid_path = os.path.join(mvs_dir, f"{fname}.mp4")
    if not os.path.exists(mvs_dir):
        os.makedirs(mvs_dir)

    # print(encoded_vid_path)
    # print(out_vid_path)

    # extract frames according to FPS_GT=30
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",

                                    # GOP setting
                                      "-g", "15",
                                      "-keyint_min", "15",
                                      "-sc_threshold", "0",

                                      "-ss", f"{start_time}",
                                      "-t", f"{duration}",
                                      "-q:v", "2",
                                    #   "-vsync", "0", 
                                      "-start_number", "0",
                                      out_vid_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
                                     
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()
    
    else:
        print("Encode video done!")


    # # Step2: get mvs
    mvs_file = os.path.join(mvs_dir, f"tmp.csv")
    get_mvs(out_vid_path, mvs_file)


    # Step3: get degree of motion
    df = pd.read_csv(mvs_file)
    # print(df.head())

    srcx = df['srcx']
    srcy = df['srcy']
    dstx = df['dstx']
    dsty = df['dsty']
    frame_type = df['frametype']


    total_srcx = {}
    total_srcy = {}
    total_dstx = {}
    total_dsty = {}


    dist_sq_list = []
    dist_list = []
    id_list = []
    frame_type_list = []

    for i in range(len(srcx)):
        id = df['framenum'][i]
        # if id != 2:
        #     break

        if id in id_list:
            total_srcx[id].append(srcx[i])
            total_srcy[id].append(srcy[i])
            total_dstx[id].append(dstx[i])
            total_dsty[id].append(dsty[i])
        else:
            id_list.append(id)
            frame_type_list.append(frame_type[i])
            total_srcx[id] = [srcx[i]]
            total_srcy[id] = [srcy[i]]
            total_dstx[id] = [dstx[i]]
            total_dsty[id] = [dsty[i]]

    total_mv_sq = []
    total_mv = []

    for id in range(len(id_list)):
        srcx_list = total_srcx[id_list[id]]
        srcy_list = total_srcy[id_list[id]]
        dstx_list = total_dstx[id_list[id]]
        dsty_list = total_dsty[id_list[id]]

        dist_sq_list = []
        dist_list = []

        for i in range(len(srcx_list)):

            dist_sq = math.pow((int(dsty_list[i]) - int(srcy_list[i])), 2) \
                      + math.pow((int(dstx_list[i]) - int(srcx_list[i])), 2)
            dist = math.pow(dist_sq, 0.5)

            dist_sq_list.append(dist_sq)
            dist_list.append(dist)

        mv_sq = sum(dist_sq_list)
        mv = sum(dist_list)

        total_mv_sq.append(mv_sq)
        total_mv.append(mv)
        # print(f"FID-{id_list[id]} TYPE-{frame_type_list[id]} MV-{mv}")

    avg_motion = round(sum(total_mv)/len(total_mv), 2)
    print(f"Avg motion in {len(total_mv)} frames = {avg_motion}")

    return avg_motion


    # print(total_mv_sq)
    # print(total_mv)

    # return total_mv, avg_motion



if __name__ == "__main__":
    # fname = "./test.mp4"
    # get_mvs(fname)

    res_fname = 'test'
    get_motion(res_fname, 0, 2)



