import os
import re
import csv
import logging
import sys
sys.path.append('..')
from .differ import Differencer, get_frame_image, select_frames_with_thresh
import math
RES_LEVEL = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FPS_LEVEL = [i for i in range(1, 31)]
QP_LEVEL = [i for i in range(20, 50)]

RA = 22.6
RB = 19.4
RC = -3.31

QA = -7.38
QB = 62.4

BA = 1.27
BB = -0.0768


def find_res_qp(fps, available_bw_pred):


    # est_file_size_bw = BA*fps + BB
    # est_file_size_best_fps = BA*30 + BB
    # ratio = est_file_size_best_fps / est_file_size_bw

    est_bw_fps = BA*fps + BB

    # est_file_size_best_fps = BA*30 + BB

    # ratio_fps = est_bw / est_file_size_best_fps  # < 1

    # est_file_size_res = (RA * pow(res,2) + RB*res + RC) / ratio
    # est_file_size_qp = (QA * math.log(qp) + QB) / ratio

    # est_bw = (RA * pow(res,2) + RB*res + RC) / ratio
    # est_bw = (QA * math.log(qp) + QB) / ratio

    # res = (-RB + math.sqrt(pow(RB,2)-4*RA*(RC - est_bw/ratio))) / 2*RA

    r_opt = RES_LEVEL[0]
    qp_opt= QP_LEVEL[-1]
    min_diff = float('inf')
    
    est_bw_best_res = RA * pow(RES_LEVEL[-1],2) + RB*RES_LEVEL[-1] + RC
    est_bw_best_qp = QA * math.log(QP_LEVEL[0]) + QB
    # print(est_bw_best_res, est_bw_best_qp, 'should be same')

    est_bw_current_final_config = 0

    for r in RES_LEVEL:
        est_bw_res_r = RA * pow(r,2) + RB*r + RC
        ratio_res = est_bw_res_r / est_bw_best_res

        for q in QP_LEVEL:
            est_bw_qp_q = QA * math.log(q) + QB
            ratio_qp = est_bw_qp_q / est_bw_best_qp

            est_bw_current_config = est_bw_fps * ratio_res * ratio_qp
            diff = available_bw_pred - est_bw_current_config

            # print(f"RES=[{r}] - QP=[{q}] - Difference=[{diff}]")
            if (diff < min_diff and diff >= 0):
                r_opt = r
                qp_opt = q
                min_diff = diff
                est_bw_current_final_config = est_bw_current_config

    return r_opt, qp_opt, est_bw_current_final_config


def gen_new_config(est_bw):



    ###############################
    #####   1.Determine FPS   #####
    ###############################
    
    file_path = "/Users/alanzyt/Desktop/video-analytics/dataset/trafficcam_1_gt_tmp/src/"

    start_fid = 0
    end_fid = 29
    frame_range = [start_fid,end_fid]

    # tunable
    thresh = 0.02


    # differencing
    #              0      1       2      3       4      5      6      7
    feat_set = ['pixel','area','edge','corner','hist','hog','sift','surf']
    feat_type = feat_set[1]
    differencer = Differencer.str2class(feat_type)()

    feats = [differencer.get_frame_feature(get_frame_image(file_path,i))
                for i in range(frame_range[0], frame_range[1]+1)]
    # print(len(feats))

    # diff is the fraction changed over pre-determined threshold
    diff_vec = [differencer.cal_frame_diff(ft1, ft0)
                for ft0, ft1 in zip(feats[:-1], feats[1:])]
    # print(diff_vec)

    select_frames = select_frames_with_thresh(diff_vec,thresh)

    fps = min(FPS_LEVEL, key=lambda x: abs(x - len(select_frames)))
    # fps = select_frames

    print(f"Estimated fps: {len(select_frames)}; Rounded fps: {fps}")


    ###############################
    ##### 2.Determine Res/QP  #####
    ###############################

    # est_bw = 10
    res, qp, est_bw_config = find_res_qp(fps, est_bw)





    new_res = RES_LEVEL.index(res)
    new_fps = FPS_LEVEL.index(fps)
    new_qp = QP_LEVEL.index(qp)

    return new_res, new_fps, new_qp, est_bw_config