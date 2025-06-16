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
  This is a real-world dataset captured from diverse indoor and outdoor static scenes, comprising 200 sets of raw event streams alongside corresponding high-quality ground truth frames.

- **E-StaDyn Dataset:**
  This is a synthetic dataset comprising 130 distinct scenes, each characterized by a unique static background and dynamic foreground.
