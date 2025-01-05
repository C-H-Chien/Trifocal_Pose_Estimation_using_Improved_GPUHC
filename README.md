## Solving Trifocal Pose Estimation by Accelerated GPU-HC (BMVC 2024 / IPDPS 2025)
### Research @ LEMS + ICL, Brown University + Innovative Computing Lab
## Introduction
This repository hosts the code for two papers, one published in BMVC 2024 and one accepted by IPDPS 2025 (see Papers below. IPDPS paper is coming soon). It demonstrates the significant efficiency improvements over [GPU-HC](https://github.com/C-H-Chien/Homotopy-Continuation-Tracker-on-GPU/tree/main) (which has now been integrated to the GPU-HC repository) and is applied to solving a trifocal pose estimation problem under a RANSAC scheme, showing a factor of 9.17×, 13.75×, 17.06×, and 17× speedups on NVIDIA V100, A100, H100, and GH200
GPUs, respectively. Strategies for speed improvements are four-fold: _(i)_ direct parameter homotopy evaluation, _(ii)_ GPU utilization improvement (reducing register pressure and instruction cache misses), _(iii)_ pruning homotopy paths, and _(iv)_ early termination of RANSAC process. The code released here also supports multiple GPUs. Note that in the BMVC paper we called the improved GPU-HC solver GPU-HC++. Refer to the papers for more details.

## Dependencies
- CMake 3.2X or above <br />
- CUDA 11.X or 12.X (depending on the GPU you are using) <br />
- MAGMA 2.5.4 or above (we rely on MAGMA for complex number computation, both on the host and device sides) <br />
- OpenBLAS 0.3.X <br />
- [YAML-CPP](https://github.com/jbeder/yaml-cpp) (this is used to parse data from a .yaml file.) <br />

## Build and run the code
Make sure the directories for the dependencies are changed based on your machine/server (See ``CMakeLists.txt`` and ``magmaHC/CMakeLists.txt`` for more information). Follow the standard steps to build the code:
```bash
$ mkdir build && cd build
$ cmake ..
$ make -j
```
THe executive file resides in ``build/bin/`` and can be run by
```bash
$ ./magmaHC-main -p trifocal_2op1p_30x30
```
where ``-p`` is the flag for the problem name. Here, we are solving the trifocal pose estimation problem code-named as "trifocal_2op1p_30x30" (It means "trifocal pose estimation problem using 2 oriented point correspondence and 1 point correspondence in a formulation of 30 polynomial equations in 30 unknowns"). This flag follows the input pattern of the original [GPU-HC](https://github.com/C-H-Chien/Homotopy-Continuation-Tracker-on-GPU/tree/main) code. <br /><br />
Note that:
- The main file is under ``cmd/magmaHC-main.cpp``.
When executing the code, make sure to use as many as CPU cores to run the CPU-HC solver as it could take a long time upon completion. Alternatively, you could comment out the [CPU-HC](https://github.com/C-H-Chien/Trifocal_Pose_Estimation_using_GPUHC_plusplus/blob/95d97031abe5675e15f93021798631a8d72df4da/cmd/magmaHC-main.cpp#L257) in ``cmd/main.cpp``.
- The trifocal pose solutions can be found under ``Output_Write_Files/``.

## Settings in the YAML file
Most of the settings in the ``problems/trifocal_2op1p_30x30/gpuhc_settings.yaml`` file can be remained unchanged; however, the following parameters shall be specified depending on user's need:
- ``Abort_RANSAC_by_Good_Sol``: true if the GPU solver is early stopped when a good hypothesis (trifocal pose) is found; false otherwise. This is proposed in the IPDPS paper but not in the BMVC paper.
- ``Num_Of_GPUs``: number of GPUs used to solve the problem.
- ``Num_Of_Cores``: number of CPUs for CPU-HC to solve the problem.
- ``RANSAC_Dataset``: dataset to be used. The current released code provides a noiseless [synthetic curve dataset](https://github.com/rfabbri/synthcurves-multiview-3d-dataset). Real-world dataset shall be released soon.

## Order of System Target Parameters
Target parameters are assigned and defined in the [Prepare_Target_Params function](https://github.com/C-H-Chien/Trifocal_Pose_Estimation_using_Improved_GPUHC/blob/85f79b03143805a35dd40768544f0322e5a7b7bf/magmaHC/GPU_HC_Solver.cpp#L252C6-L252C19), or you could see it in the Julia's script ``problems/trifocal_2op1p_30x30/trifocal_2op1p_30x30_monodromySolve.jl`` for generating the problem formulation.  <br />
Specifically, let ($\gamma_{i1}$, $\gamma_{i2}$, $\gamma_{i3}$) be $i$-th triplet correspondence. The minimal data of the trifocal pose estimation is three triplet correspondences, _i.e._, $i$=1,2,3, with two of them additionally associated with orientations described by unit vectors ($d_{j1}$, $d_{j2}$, $d_{j3}$), $j$=1,2. The order of system parameters is ($\gamma_{11}$, $\gamma_{12}$, $\gamma_{13}$, $\gamma_{21}$, $\gamma_{22}$, $\gamma_{23}$, $\gamma_{31}$, $\gamma_{32}$, $\gamma_{33}$, $d_{11}$, $d_{12}$, $d_{13}$, $d_{21}$, $d_{22}$, $d_{23}$, $\rho_{11}$, $\xi_{2}$, $\xi_{3}$) where $\rho_{11}$ is the depth of $\gamma_{11}$ and $\xi_{2}$ as well as $\xi_{3}$ can be arbitrary values. Notations used here are consistent with the papers.


## Incremental Speed Improvements
If you'd like to recreate the experiments of incremental performance improvements reported in the papers, a set of GPU kernels and device functions are arxived in ``arxived_GPU_code/``. It should not be hard to include them in the ``magmaHC/GPU_HC_Solver.cpp`` to run the code. See [README_arxived_GPU_code.md](https://github.com/C-H-Chien/Trifocal_Pose_Estimation_using_GPUHC_plusplus/blob/master/arxived_GPU_code/README_arxived_GPU_code.md) for more information. 

## Papers
BMVC paper is the first paper for embedding the GPU-HC solver in a RANSAC scheme for solving a real-world problem, while IPDPS paper gives a more detailed anaylsis of the incremental improvements of the solver. 
```BibTeX
@InProceedings{chien:etal:BMVC:2024,
  title={Recovering {SLAM} Tracking Lost by Trifocal Pose Estimation using {GPU-HC++}},
  author={Chien, Chiang-Heng and Abdelfattah, Ahmad and Kimia, Benjamin},
  booktitle={Proceedings of the 35th British Machine Vision Conference (BMVC)},
  pages={},
  year={2024}
}
```
```BibTeX
@InProceedings{chien:etal:IPDPS:2025,
  title={Accelerating Homotopy Continuation with {GPUs}: Application to Trifocal Pose Estimation},
  author={Chien, Chiang-Heng and Abdelfattah, Ahmad and Kimia, Benjamin},
  booktitle={Accepted by the 39th IEEE International Parallel & Distributed Processing Symposium (IPDPS)},
  pages={},
  year={2025}
}
```

## TODO List
- [ ] Provide real-world datasets
- [ ] Release code for using trifocal pose estimation for recovering SLAM tracking failure

## Contributors
Chiang-Heng Chien* (chiang-heng_chien@brown.edu) <br />
Ahmad Abdelfattah (ahmad@icl.utk.edu) <br />
*corresponding author