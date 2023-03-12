# video-analytics

## 1. Intro
This is a video analytics system built on top of [DDS](https://github.com/KuntaiDu/dds). It serves as the simulation environment of [Maxim](https://ieeexplore.ieee.org/document/9859614/).

## 2. Install Instructions

To run our code, please make sure that conda is installed. Then, under dds repo, run

```conda env create -f conda_environment_configuration.yml```

to install dds environment. Note that this installation assumes that you have GPU resources on your machine. If not, please edit ```tensorflow-gpu=1.14``` to ```tensorflow=1.14``` in ```conda_environment_configuration.yml```.

Now run

```conda activate dds```

to activate dds environment, and 

```cd workspace```

and run 

```wget people.cs.uchicago.edu/~kuntai/frozen_inference_graph.pb```

to download the object detection model (FasterRCNN-ResNet101).

## 3. Run our code

Under ```DDSrepo/workspace```, run

```python play_video.py```

to run DDS!