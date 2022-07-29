Exploring the use of single-precision GEMM to achieve near double-precision accuracy on tiled TRSM. This method exploits the performance of single-precision BLAS-3 operations on tensor cores. However, this implementation runs entirely on the CPU and serves to analyze the numerical stability of each single-precision method.


Usage
=====
To compile the c file use the make file:
- On Leconte(or IBM machine with the same directory layout) use `make l` to build a shared library file named `libmixedtrsm.so`. Alternatively use `make lb` to build stand-alone binary file named `test`.
- On Equinox(or Intel machine with the same directory layout) use `make e` to build a shared library file named `libmixedtrsm.so`. Alternatively use `make eb` to build stand-alone binary file named `test`.
- On Cousteau(or AMD machine with the same directory layout) use `make c` to build a shared library file named `libmixedtrsm.so`. Alternatively use `make cb` to build stand-alone binary file named `test`.


Once the shared library file has been generated and using a machine with an Xserver run `python numerical_analysis.py` using python >=3.8(with NumPy and Matplotlib). This will generate error plots for each different method. Edit python file to change parameters.


Numerical Analysis
==================
The following plots were ran on the equinox machine

Average Tile Error Across Precisions
(Varying Matrix Sizes)
------------------------------------
![](plots/lapacke_plots/ComparingErrorsLog128,16.png)
![](plots/lapacke_plots/ComparingErrorsLog1024,128.png)
![](plots/lapacke_plots/ComparingErrorsLog2048,256.png)
![](plots/lapacke_plots/ComparingErrorsLog4096,512.png)
![](plots/lapacke_plots/ComparingErrorsLog8192,1024.png)

Matrix Generation Methods
(Average Tile Error, Matrix Size: 128, Tile Size: 16)
-------------------------

![](plots/other_generators_plots/AverageTileErrorRegularGenLog128,16,8.png)
![](plots/other_generators_plots/AverageTileErrorXGenLog128,16,8.png)
![](plots/other_generators_plots/AverageTileErrorLapackeGenLog128,16,8.png)

Average Matrix Error
(Varying Tile Sizes)
--------------------

![](plots/other_plots/AverageMatrixErrorNumTiles128Log.png)
![](plots/other_plots/AverageMatrixErrorNumTiles256Log.png)


All of the non-normalized error methods produced plots with the same general trend and shape. The only noticable differences were that some were either scaled or shifted in relation to the original "Average Tile Error" plot.
The normalized error method removed the "steps" that can be seen in the other, non-normalized error methods. Although, the normalized error was more "noisy" than the other plots.


Conclusion
==========
The new algorithm (mbr) consistently yields lower error on all different error metrics in comparison to the other algorithms (ssbr,sbr).