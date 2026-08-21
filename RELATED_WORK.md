# Related work — survey + differentiation (2026-08-21)

Positioning for the ShadowWeave workshop paper. All titles/authors/venues/arXiv IDs
verified against arXiv/publisher pages. **Framing is amodal completion + calibrated
in-shadow uncertainty, NOT temporal forecasting** (see the completion/forecasting
decomposition in RESULTS.md).

## Closest papers (differentiate explicitly)

**ProxMaP — Sharma, Chen & Tokekar, IROS 2023 (arXiv:2305.05519). ⚠ STRONGEST NOVELTY THREAT.**
Self-supervised model predicting occupancy in the reachable/occluded region around an
indoor robot from a single partial top-down (RGB-D→BEV) view, for navigation efficiency.
*Our delta:* ProxMaP emits a **point estimate** — no explicit shadow/visibility mask, **no
calibrated uncertainty in the occluded region**, no completion-vs-forecasting attribution.
ShadowWeave's defensible novelty over it is precisely the calibrated uncertainty (ECE)
inside the hidden cells and the honest completion/forecasting decomposition. Must be cited
and differentiated head-on.

**VisHall3D — Lu et al., ICCV 2025 (arXiv:2507.19188). Closest on "complete the invisible."**
Two-stage monocular SSC: reconstruct visible geometry, then hallucinate invisible geometry
via an occlusion-masked generative stage.
*Our delta:* RGB-driven **outdoor voxel SSC** (SemanticKITTI/KITTI-360), reports **mIoU/IoU
only — no uncertainty calibration** in the hallucinated region, no egocentric-BEV nav
target, no persistent-structure-vs-dynamics separation. We are depth-only, indoor-egocentric
BEV, and we *measure the honesty* (calibration) of hidden-space predictions rather than only
rendering plausible ones.

**OccWorld — Zheng et al., ECCV 2024 (arXiv:2311.16038). The temporal-forecasting foil.**
Autoregressive 3D-occupancy world model forecasting future scene + ego motion from past
occupancy.
*Our delta:* OccWorld is **temporal forecasting** on **multi-frame** lidar/multi-camera
occupancy — exactly the framing our decomposition argues against for our own gains. We are
single-frame monocular-depth and show the gain is **spatial amodal completion, not temporal
forecasting** (dynamic signal ≈ 0).

## Related-work paragraph (~170 words, drop-in)

Predicting the structure of unobserved space has been approached from three angles.
Semantic scene completion, from SSCNet through MonoScene (Cao & de Charette, 2022) and its
monocular successors VoxFormer (Li et al., 2023) and VisHall3D (Lu et al., 2025), infers
dense voxel geometry — and even hallucinates geometry beyond the visible frontier — from a
single image, but targets outdoor driving voxels evaluated purely by mIoU/IoU, without
calibrated uncertainty in the occluded region. Occupancy world models such as OccWorld
(Zheng et al., 2024), together with occupancy-forecasting benchmarks (Occ3D; Tian et al.,
2023) and self-supervised raycasting forecasters (Khurana et al., 2023), instead predict
the *temporal* evolution of multi-frame occupancy for autonomous vehicles. Closest in
spirit, ProxMaP (Sharma et al., 2023) completes indoor proximal occupancy from a single
view for navigation, yet emits only point estimates. ShadowWeave is distinct in delivering
egocentric BEV amodal completion into explicitly masked occluded space from a single
monocular-depth frame, with calibrated uncertainty inside the hidden region, and a
completion-versus-forecasting decomposition showing the gain is spatial completion rather
than dynamics.

## BibTeX

```bibtex
@inproceedings{sharma2023proxmap,
  title     = {ProxMaP: Proximal Occupancy Map Prediction for Efficient Indoor Robot Navigation},
  author    = {Sharma, Vishnu Dutt and Chen, Jingxi and Tokekar, Pratap},
  booktitle = {2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2023},
  note      = {arXiv:2305.05519}
}

@inproceedings{lu2025vishall3d,
  title     = {VisHall3D: Monocular Semantic Scene Completion from Reconstructing the Visible Regions to Hallucinating the Invisible Regions},
  author    = {Lu, Haoang and Su, Yuanqi and Zhang, Xiaoning and Gao, Longjun and Xue, Yu and Wang, Le},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  note      = {arXiv:2507.19188}
}

@inproceedings{zheng2024occworld,
  title     = {OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving},
  author    = {Zheng, Wenzhao and Chen, Weiliang and Huang, Yuanhui and Zhang, Borui and Duan, Yueqi and Lu, Jiwen},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
  note      = {arXiv:2311.16038}
}

@inproceedings{cao2022monoscene,
  title     = {MonoScene: Monocular 3D Semantic Scene Completion},
  author    = {Cao, Anh-Quan and de Charette, Raoul},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  note      = {arXiv:2112.00726}
}

@inproceedings{li2023voxformer,
  title     = {VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion},
  author    = {Li, Yiming and Yu, Zhiding and Choy, Christopher and Xiao, Chaowei and Alvarez, Jose M. and Fidler, Sanja and Feng, Chen and Anandkumar, Anima},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  note      = {arXiv:2302.12251}
}

@inproceedings{tian2023occ3d,
  title     = {Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving},
  author    = {Tian, Xiaoyu and Jiang, Tao and Yun, Longfei and Mao, Yucheng and Yang, Huitong and Wang, Yue and Wang, Yilun and Zhao, Hang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2023},
  note      = {arXiv:2304.14365}
}

@inproceedings{khurana2023pointcloud,
  title     = {Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting},
  author    = {Khurana, Tarasha and Hu, Peiyun and Held, David and Ramanan, Deva},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
  note      = {arXiv:2302.13130}
}
```

## Not cited (kept tight) — pull in if a differentiator needs fortifying
- Khurana et al. (2022), "Differentiable Raycasting for Self-Supervised Occupancy
  Forecasting," ECCV 2022 — predecessor to the CVPR'23 proxy paper.
- To fortify the **calibrated-uncertainty** differentiator specifically: SCOPE
  (Stochastic Cartographic Occupancy Prediction Engine, arXiv:2407.00144) and the
  conformal-navigation line "Perceive With Confidence" (arXiv:2403.08185). Author lists
  NOT yet verified — confirm before citing.
- Han et al. (2020), "Planning Paths Through Unknown Space by Imagining What Lies
  Therein," CoRL 2020.
