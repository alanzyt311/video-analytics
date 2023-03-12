import re
import os
import csv
import shutil
import subprocess
# import numpy as np
# import cv2 as cv
# import networkx
import random
# from networkx.algorithms.components.connected import connected_components


def copy(src, target, filelist):
    if os.path.isdir(src) and os.path.isdir(target):
        for file in filelist:
            source = os.path.join(src,file)
            shutil.copy(source, target)
    else:
        print("dir not correct")



def generate_images(video_name, fps, qp, resolution, start_time, duration, is_gt=True):
    dataset_dir = "../dataset"
    video_name_dir_gt = f"{video_name}_gt"
    video_name_dir_fps = f"{video_name}_fps{fps}"
    images_path_gt = os.path.join(dataset_dir, video_name_dir_gt, 'src')
    images_path_fps = os.path.join(dataset_dir, video_name_dir_fps, 'src')

    # handling directory
    if not os.path.exists(images_path_gt):
        os.makedirs(images_path_gt)

    for fname in os.listdir(images_path_gt):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path_gt, fname))

    if not os.path.exists(images_path_fps):
        os.makedirs(images_path_fps)

    for fname in os.listdir(images_path_fps):
        if "png" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path_fps, fname))


    

    # Step1: [DECODE] decode raw video with best config (fps30) between [start_time, start_time+duration]
    encoded_vid_path = os.path.join(dataset_dir, f"{video_name}.mp4")
    extracted_images_path_gt = os.path.join(images_path_gt, "%010d.png")

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

        # Step2: [ENCODE] Imply qp and resoulution to raw image frames to encode the video
        start_id = 0
        number_of_frames = len([x for x in os.listdir(images_path_gt) if "png" in x])
        print(f"1st round images count: {number_of_frames}")
        encoded_vid_path = os.path.join(images_path_gt, "temp.mp4")
        scale = f"scale=trunc(iw*{resolution}/2)*2:trunc(ih*{resolution}/2)*2"

        encoding_result = subprocess.run(["ffmpeg", "-y",
                                            "-loglevel", "error",
                                            "-start_number", str(start_id),
                                            '-i', f"{images_path_gt}/%010d.png",
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
        if not os.path.isdir(images_path_gt):
            print("%s not a valid directory", images_path_gt)
            return

        for fname in os.listdir(images_path_gt):
            if "png" not in fname:
                continue
            else:
                os.remove(os.path.join(images_path_gt, fname))

        encoded_vid_path = os.path.join(images_path_gt, "temp.mp4")
        extracted_images_path_gt = os.path.join(images_path_gt, "%010d.png")

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




        # Step4: start to sample frames according to fps
        flist = []
        pick_list = []
        divisible = [1,2,5,10,15,20,25]
        
        if (fps in divisible):
            fps_gt = 30
            ratio = fps_gt / fps

            if ratio == 1:
                flist = os.listdir(images_path_gt)
            
            else:
                num_of_frames_gt = len([x for x in os.listdir(images_path_gt) if "png" in x])
                times = int(num_of_frames_gt / 30)
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


        # FPS = 20/25, sample randomly
        else:
            num_of_frames_gt = len([x for x in os.listdir(images_path_gt) if "png" in x])
            desired_len = num_of_frames_gt * fps / 30

            while (len(pick_list) < desired_len):
                pick = random.randint(1, num_of_frames_gt) - 1
                if (pick not in pick_list):
                    pick_list.append(pick)

            for i in range(len(pick_list)):
                fname = "%010d.png"%pick_list[i]
                print(fname)
                flist.append(fname)

            print(pick_list)
            print(flist)


        # COPY THE SELECTED FRAMES TO NEW DIR
        copy(images_path_gt, images_path_fps, flist)
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


        num_of_frames_gt = len([x for x in os.listdir(images_path_gt) if "png" in x])
        print(f"Generate {num_of_frames_gt} images for GT batch")
        print("Generate images for this batch:")
        print("fps:", fps, "start:", start_time, "duration:", duration, "num of frames:", num_of_frames_fps)

    return images_path_fps
 

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


