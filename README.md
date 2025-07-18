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
  This is a real-world dataset captured from diverse indoor and outdoor static scenes, comprising 200 sets of raw event streams alongside corresponding high-quality ground truth frames. [Download from this link](https://www.dropbox.com/scl/fo/ftpw99ohu9uzkopswnyjb/AERu2EwripoVPwddQX_gZb4?rlkey=yfynvufnvy5v34vib2i85318u&st=9m7stipp&dl=0).

  **- NOTE:**
  - All data in ```E-Static``` were captured using the **SilkyEvCam hybrid sensor**, except for the ```DAVIS_color``` folder, which contains data collected with the **DAVIS346 Color** camera.
  - For each scene, the raw event stream is stored in the ```event``` folder, and the corresponding high-quality frame is stored in the ```frame``` folder, with matching filenames.
  - The ```color``` folder contains raw inputs used for color static reconstruction, acquired by placing optical filters in front of the SilkyEvCam.

- **E-StaDyn Dataset:**
  This is a synthetic dataset comprising 130 distinct scenes, each characterized by a unique static background and dynamic foreground. [Download from this link](https://www.dropbox.com/scl/fo/ftpw99ohu9uzkopswnyjb/AERu2EwripoVPwddQX_gZb4?rlkey=yfynvufnvy5v34vib2i85318u&st=9m7stipp&dl=0).

  **- NOTE:**
  - Each folder in ```E-StaDyn``` represents a distinct scene. For each scene, we used **Blender** to configure a static background and introduced a foreground object following a randomly generated motion trajectory. We then rendered a sequence of continuous frames at 60 FPS. Synthetic event streams were generated using the **DVS-Voltmeter** simulator.
  - The generated event stream was segmented into multiple chunks based on timestamps for subsequent voxel grid construction, with each chunk aligned to its corresponding key frame.
  - Both the ```train``` and ```test``` folders contain a ```background``` subfolder, which includes high-quality static background frames for all scenes.
  - Within each scene folder, ```event.txt``` lists the file paths to all segmented event streams, while ```frame.txt``` provides the file paths to all rendered continuous frames.
    
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

## Reconstruct Static Scene Only

The static reconstruction branch of the URSEE framework enables the independent reconstruction of static scenes. Users simply need to convert the static event stream into a <code>.csv</code> file (timestamp, x, y, and polarity).

**- NOTE:** Adaptively adjust the ON/OFF threshold according to the static scene to obtain sufficient events for reconstruction.

1. Obtain the **initial event frame** using the `convolutional integration method`.
```
python conv_integral_reconstruction.py
```
2. Obtain the **high-quality static frame** by denoising the initial one using <code>SRD module</code>. Download the checkpoint named ```SRD_module.pth``` [link](https://www.dropbox.com/scl/fo/ji6v77ahxabv0v6os6xt6/AJcWBuBdwmlKPePhglDHXY0?rlkey=cqkiulqluq9xczn48c7xp88hu&st=f0op2wor&dl=0).
```
python SRD_test_gqy.py
```
- **Color Reconstruction of Static Scenes (Option).** Obtain static event streams for the R, G, and B channels separately using a DAVIS color camera or by placing color filters in front of a standard event camera. Then, apply the proposed method to reconstruct high-quality intensity frames for each channel and fuse them to generate the final color reconstruction.

## Reconstruct Motion Video with Static Background

URSEE enables simultaneous processing of motion-triggered dynamic events and background-generated static events, allowing users to reconstruct motion videos with static backgrounds through several steps.

### Step 1 Separate the raw event stream into distinct dynamic and static event streams

```
python separate.py
```

### Step 2 Reconstruct high-fidelity static background frames

Follow the steps outlined in the Reconstruct Static Scene Only section



### Step 3 Reconstruct motion sequences with static backgrounds

Take the dynamic event stream and the reconstructed static background frame as inputs of the ```dataset_withoutflow.py``` file.
Download the checkpoint of the ERSD Module named ```ERSD_checkpoint.pth``` [link](https://www.dropbox.com/scl/fo/ji6v77ahxabv0v6os6xt6/AJcWBuBdwmlKPePhglDHXY0?rlkey=cqkiulqluq9xczn48c7xp88hu&st=f0op2wor&dl=0).
```
python predict_ERSD.py -c predict_ERSD.json
```

## Citation

```
@inproceedings{gao2025unified,
  title={Unified Reconstruction of Static and Dynamic Scenes from Events},
  author={Gao, Qiyao and Duan, Peiqi and Lou, Hanyue and Teng, Minggui and Cai, Ziqi and Chen, Xu and Shi, Boxin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={27914--27923},
  year={2025}
}
```
