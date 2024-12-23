## GPU kernels and device functions for the incremental performance improvements of the GPU-HC solver

Below is a list of GPU kernels corresponding to the strategies used in the papers for performance improvement. Note that these are used to solve trifocal pose estimation problem, but some of them can be easily applied to other problems. Device functions used in each kernel can be found in the code. <br />
- ``*_P2C.cu``: The most naive implementation of GPU-HC solver. "P2C" refers to "parameter to coefficient" which converts system parameters to coefficients before evaluating Jacobians/Homotopy.
- ``*_PH.cu`` uses direct homotopy evaluation. "PH" is "parameter homotopy" which preserves system parameters for Jacobians/Homotopy evaluation, as opposed to "P2C".
- ``*_PH_CodeOpt.cu`` uses _(i)_ direct homotopy evaluation and _(ii)_ code optimization (GPU utilization improvement). ``*_PH_CodeOpt_Volta.cu`` is used for NVIDIA Volta GPU.
- ``*_PH_CodeOpt_TrunPaths.cu`` uses _(i)_ direct homotopy evaluation, _(ii)_ code optimization, and _(iii)_ truncating homotopy paths by signs of depth variables. This is the same as ``magmaHC/gpu-kernels/*_PH_CodeOpt_TrunPaths.cu``. ``*_PH_CodeOpt_TrunPaths_Volta.cu`` is used for NVIDIA Volta GPU.
- ``*_PH_CodeOpt_TrunPaths_TrunRANSAC.cu`` uses all strategies and is the same as ``magmaHC/gpu-kernels/*_PH_CodeOpt_TrunPaths_TrunRANSAC.cu``. ``*_PH_CodeOpt_TrunPaths_TrunRANSAC_Volta.cu``
    is used for NVIDIA Volta GPU.