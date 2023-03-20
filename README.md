# Semantic Ray: Learning a Generalizable Semantic Field with Cross-Reprojection Attention

Official implementation of ['Semantic-Ray: Learning a Generalizable Semantic Field with Cross-Reprojection Attention'](https://liuff19.github.io/S-Ray/).

The paper has been accepted by **CVPR 2023** 🔥.

## Introduction
We propose a generalizable semantic field named Semantic Ray, which is able to learn from multiple scenes and generalize to unseen scenes. Different from Semantic NeRF which relies on positional encoding thereby limited to the specific single scene, we design a Cross-Reprojection Attention module to fully exploit semantic information from multiple reprojections of the ray. In order to collect dense connections of reprojected rays in an efficient manner, we decompose the problem into consecutive intra-view radial and cross-view sparse attentions, so that we extract informative features at small computational costs. Experiments on both synthetic and real scene data demonstrate the strong generalization ability of our S-Ray. We have also conducted extensive ablation studies to further show the effectiveness of our proposed Cross-Reprojection Attention module. With the generalizable semantic field, we believe that S-Ray will encourage more explorations of potential NeRF-based high-level vision problems in the future.

<div align="center">
  <img src="imgs/teaser.png"/>
</div>

## Code
Comming soon.

## Acknowledgement
This repo benefits from [NeuRay](https://github.com/liuyuan-pal/NeuRay), [IBRNet](https://github.com/googleinterns/IBRNet), [Semantic-NeRF](https://github.com/Harry-Zhi/semantic_nerf), and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). Thanks for their wonderful works.

## Citation
If you found this work to be useful in your own research, please consider citing the following:
```
@inproceedings{liu2023semantic,
author = {Liu, Fangfu and Zhang, Chubin and Zheng, Yu and Duan, Yueqi},
title = {Semantic Ray: Learning a Generalizable Semantic Field with Cross-Reprojection Attention},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2023}
                }
```