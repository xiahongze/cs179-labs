CS179 Homework/Labs
===================

This repository contains all of the homework and lab assignments for CS179 (as of year 2023). The assignments are organized by labs and there are six labs in this year (where lab5 and lab6 are merged into one).

There are some notes that you should be aware of:

## My hosting environment

It is critical that you get supporting driver and CUDA installed in your system. Otherwise, it is very likely that you will get
a segfault or invalid values without any error messages. Reboot your host machine after installing new driver and CUDA.

```
Windows 11 with WSL 2 Ubuntu 20.04 having CUDA 12.2 & 11.8 installed. Both seem to work fine.

NVIDIA GeForce RTX 3080.

~$ uname -a
Linux 5.15.90.1-microsoft-standard-WSL2 #1 SMP Fri Jan 27 02:56:13 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux

~$ nvidia-smi
Sun Oct 15 09:27:48 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 528.49       CUDA Version: 12.0     |

~$ g++ --version
g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0

$ /usr/local/cuda/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

## Alternation of the `Makefile`

- `NVCC_GENCODES` can either be commented out or adapted to one's own GPU architecture. For lab5, it is important that
one should use the right architecture (e.g. `sm_86` for RTX 3080) to compile the code. Otherwise, the code will not work.
The symptom is that the loss and accuracy will be constantly `0`.
- I do not have `libsndfile` installed and hence would only use the `noaudio` target for lab1 & lab3.

## Running the code

For most labs, just simply go to the folder and hit `make` to compile the code. Then, run the executable file. For lab1 & lab3,
I would use the `noaudio` target to compile the code.

## Expected output

### Lab1

```
$ make noaudio
$ ./noaudio-blur 128 1024
....
Successful output

CPU time: 527.228 milliseconds
GPU time: 20.0652 milliseconds

Speedup factor: 26.2757
```

### Lab2

```
$ make
$ ./transpose
...
Size 4096 naive CPU: 84.499329 ms
Size 4096 GPU memcpy: 0.289888 ms
Size 4096 naive GPU: 0.615424 ms
Size 4096 shmem GPU: 0.241664 ms
Size 4096 optimal GPU: 0.203776 ms
```

### Lab3

```
$ make noaudio
$ ./noaudio-fft 128 1024
...
CPU normalization constant: 0.502063
GPU normalization constant: 0.502063

CPU time (normalization): 28.6285 milliseconds
GPU time (normalization): 0.513472 milliseconds

Speedup factor (normalization): 55.7548
```

### Lab4

Getting the parameters right for each of the blas functions is critical. If not, you can
get a segfault or you can get invalid values without errors (which is the worst).

Pay attention to the leading dimension, transpose operation and ncol/nrow for each matrix.

```
$ make
$ ./point_alignment  resources/bunny2.obj resources/bunny2_trans.obj bunny2_rotated.obj
Aligning resources/bunny2.obj with resources/bunny2_trans.obj
Reading resources/bunny2.obj, which has 14290 vertices
Reading resources/bunny2_trans.obj, which has 14290 vertices
xx4x4 = Transpose[x1mat] . x1mat status: 0
x1Tx2 = Transpose[x1mat] . x2mat status: 0
0.000398147 -6.75226e-11 -0.5 2.22572e-08 
3.69351e-08 0.5 1.00468e-08 1.50906e-08 
0.5 1.97539e-08 0.000398134 9.65071e-09 
0 0 0 1 
trans_mat . point_mat^T status: 0
```

One can use [Online 3D Viewer](https://3dviewer.net/) to view the output. Check that `bunny2_rotated.obj` is
roughtly the same angle as `resources/bunny2_trans.obj`.

### Lab5

Download the training and testing sets from LeCun's website. Then, run the following commands:

```$ gunzip *.gz```

Move the files under `mnist` in lab5, then

```
$ make
$ bin/dense-neuralnet --dir ./mnist
...
Epoch 25
--------------------------------------------------------------
Loss: 0.273156, Accuracy: 0.924083

Image Magic        :803                            2051
Image Count        :2710                           10000
Image Rows         :1C                              28
Image Columns      :1C                              28
Label Magic        :801                            2049
Label Count        :2710                           10000
Loaded test set.
Validation
----------------------------------------------------
Loss: 0.272189, Accuracy: 0.9242

# OR
$ bin/conv-neuralnet --dir ./mnist
...
Epoch 25
--------------------------------------------------------------
Loss: 0.0759577,        Accuracy: 0.977134

Image Magic        :803                            2051
Image Count        :2710                           10000
Image Rows         :1C                              28
Image Columns      :1C                              28
Label Magic        :801                            2049
Label Count        :2710                           10000
Loaded test set.
Validation
----------------------------------------------------
Loss: 0.0729156,        Accuracy: 0.9766
```

As can be seen, the CNN network performs better than the dense network.