[简体中文](../../zh-CN/dataset/k400.md) | English
# AVA Data Preparation
This document mainly introduces the preparation process of AVA dataset. 
It mainly includes three parts: Data Download, Extract the frames of the video file, Generate the path list for the frames and video files.
Before we start, please make sure that the directory is located at `$PaddleVideo/data/ava/shell`.


---

## 1. Data Download
For basic dataset information, you can refer to the official website[AVA](https://research.google.com/ava/index.html).
For the dataset download, you can refer to the [AVA Download](https://github.com/cvdfoundation/ava-dataset) ，
which introduce the way to download the dataset. Furthermore,considering the difficulty in downloading, 
we upload the video files to Baidu cloud disk in the form of zip packages, and users can download it by themselves according to their needs.
The total dataset needs about storage space. [Link]() <sup>coming soon</sup>.
**Note:** the video files should be placed in data/ava/videos

---

---
## 2.Prepare Annotations

Next, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

This command will download `ava_v2.1.zip` for AVA `v2.1` annotation. If you need the AVA `v2.2` annotation, you can try the following script.

```shell
VERSION=2.2 bash download_annotations.sh
```

**note:** In fact,we will also provide the zip files in Baidu cloud disk,so you needn't download it

---
## 3. Cut each video from its 15th to 30th minute and make them at 30 fps.

```shell
bash cut_videos.sh
```
---

## 4. Extract RGB Frames

you can use the ffmpeg to extract RGB frames by the following script.

```shell
bash extract_rgb_frames_ffmpeg.sh
```

---

## 5.Pulling Proposal Files

The scripts are adapted from FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks).

Run the follow scripts to fetch pre-computed proposal list.

```shell
bash fetch_ava_proposals.sh
```

---
## 6.Folder Structure

After the whole data pipeline for AVA preparation.
you can get the rawframes (RGB), videos and annotation files for AVA.

In the context of the whole project (for AVA only), the folder structure will look like:
```
PaddleVideo
├── configs
├── paddlevideo
├── docs
├── tools
├── data
│   ├── ava
│   │   ├── annotations
│   │   |   ├── ava_dense_proposals_train.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_val.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_test.FAIR.recall_93.9.pkl
│   │   |   ├── ava_train_v2.1.csv
│   │   |   ├── ava_val_v2.1.csv
│   │   |   ├── ava_train_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_val_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_action_list_v2.1_for_activitynet_2018.pbtxt
│   │   ├── videos
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── videos_15min
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── rawframes
│   │   │   ├── 053oq2xB3oU
|   │   │   │   ├── img_00001.jpg
|   │   │   │   ├── img_00002.jpg
|   │   │   │   ├── ...
```