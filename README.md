# Code for URSEE: *Unified Reconstruction of Static and Dynamic Scenes from Events* (CVPR 2025 Highlight)

<p align="center">
  <a href="https://youtu.be/Zh8KF_SDGrU" target="_blank">
    <img src="https://img.youtube.com/vi/Zh8KF_SDGrU/maxresdefault.jpg" alt="Watch the video" width="960">
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/Zh8KF_SDGrU" target="_blank">▶ Click here to watch the demo video on YouTube</a>
</p>

## Abstraction

This paper addresses the challenge that current event-based video reconstruction methods cannot produce static background information. Recent research has uncovered the potential of event cameras in capturing static scenes. Nonetheless, image quality deteriorates due to noise interference and detail loss, failing to provide reliable background information. We propose a two-stage reconstruction strategy to address these challenges and reconstruct static scene images comparable to frame cameras. Building on this, we introduce the URSEE framework designed for reconstructing motion videos with static backgrounds. This framework includes a parallel channel that can simultaneously process static and dynamic events, and a network module designed to reconstruct videos encompassing both static and dynamic scenes in an end-to-end manner. We also collect a real-captured dataset for static reconstruction, containing both indoor and outdoor scenes. Comparison results indicate that the proposed method achieves state-of-the-art performance on both synthetic and real data.

## Requirements

- conda or miniconda  
- CUDA-enabled GPU with at least 10GB of memory

## Datasets

- **E-Static Dataset:**
  This is a real-world dataset captured from diverse indoor and outdoor static scenes, comprising 200 sets of raw event streams alongside corresponding high-quality ground truth frames. Download from this link.

- **E-StaDyn Dataset:**
  This is a synthetic dataset comprising 130 distinct scenes, each characterized by a unique static background and dynamic foreground. Download from this link.

## Setup instruction

1. Initialize virtual env
```
conda create -n virtualenv_name python=3.10
conda activate virtualenv_name
```
   
2. Install dependencies
```
conda install pytorch~=2.1.0 torchvision==0.16.1 torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
(optional, only for visualization) conda install -c conda-forge jupyterlab nodejs ipympl matplotlib
```

## Reconstruct Static Scene only

The static reconstruction branch of the URSEE framework enables the independent reconstruction of static scenes. Users simply need to convert the static event stream into a <code>.csv</code> file (timestamp, x, y, and polarity).

**NOTE:** Adaptively adjust the ON/OFF threshold according to the static scene to obtain sufficient events for reconstruction.

1. Obtain the **initial event frame** using the `convolutional integration method`.
```
python conv_integral_reconstruction.py
```
2. Obtain the **high-quality static frame** by denoising the initial one using <code>SRD module</code>.
   Download the checkpoint from this link.
```
python conv_integral_reconstruction.py
```
