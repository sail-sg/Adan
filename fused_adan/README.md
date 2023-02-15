# Adan Optimizer fused kernel

## Dependence

1. Libtorch/Pytorch (ATen is required, Compile passed on Pytorch 1.13.1)
2. CUDA Toolkit (Compile passed on CUDA 11.6+)
3. ninja

## Usage

Using `Adan(..., foreach=False, fused=True)` enables fused Adan kernel with single tensor access.
Using `Adan(..., foreach=True, fused=True)` enables fused Adan kernel with multi-tensor access.

`foreach=True` is recommended for better performance.

**Single tensor access**
A *for loop* is used to traverse each layer when calculating the gradient of each Layer, requiring multiple kernels starts. Theoretically, accessing only one layer of parameters at a time is good for reducing peak memory usage, but it introduces kernel launch overhead.

**Multi tensor access**
The parameters of all layers are passed into the kernel at once, and the kernel internally uses a for loop to traverse each layer, requiring only one kernel start. Theoretically, this will lead to an increase in peak memory usage but will reduce the overhead of kernel startup. In actual tests, the increase in memory usage is not significant, but the kernel launch overhead is reduced.

## Benchmarking Results

Benchmarking peak memory and wall duration of optimizers: Adam v.s. FusedAdan. The benchmarking uses GPT-2 with the different numbers of heads, layers, and Emb. Dim on a single NVIDIA A100 GPU (40G).

The benchmarking is conducted based on the following config:

- vocab size: 49280
- batch size: 1
- sequence length: 2048

#### Memory Comparison

| Head | Layers | Emb. Dim | Model Size (MB) | Adam Peak (MB) | FusedAdan Peak (MB) | Δ (%) |
| :--: | :----: | :------: | :-------------: | :------------: | :-----------------: | :---: |
|  6   |   6    |   768    |       81        |      4490      |        4490         | 0.00  |
|  12  |   6    |   768    |       81        |      5848      |        5848         | 0.00  |
|  16  |   6    |   768    |       81        |      6775      |        6775         | 0.00  |
|  6   |   12   |   768    |       124       |      7151      |        7153         | 0.03  |
|  12  |   12   |   768    |       124       |      9869      |        9871         | 0.02  |
|  16  |   12   |   768    |       124       |     11733      |        11735        | 0.02  |
|  16  |   6    |   1024   |       128       |      7302      |        7302         | 0.00  |
|  16  |   12   |   1024   |       203       |     12719      |        12719        | 0.00  |
|  6   |   24   |   768    |       209       |     12471      |        12473        | 0.02  |
|  12  |   24   |   768    |       209       |     17907      |        17909        | 0.01  |
|  16  |   24   |   768    |       209       |     21596      |        21598        | 0.01  |
|  6   |   6    |   1536   |       248       |      6880      |        7308         | 6.22  |
|  12  |   6    |   1536   |       248       |      8235      |        8235         | 0.00  |
|  16  |   6    |   1536   |       248       |      9141      |        9141         | 0.00  |
|  16  |   24   |   1024   |       354       |     23530      |        23532        | 0.01  |
|  16  |   6    |   2048   |       407       |     11098      |        11098        | 0.00  |
|  6   |   12   |   1536   |       418       |     11137      |        12213        | 9.66  |
|  12  |   12   |   1536   |       418       |     13855      |        13857        | 0.01  |
|  16  |   12   |   1536   |       418       |     15667      |        15669        | 0.01  |
|  16  |   6    |   2560   |       603       |     13967      |        15965        | 14.30 |
|  16  |   12   |   2048   |       709       |     18851      |        18853        | 0.01  |
|  6   |   24   |   1536   |       758       |     19660      |        21997        | 11.88 |
|  12  |   24   |   1536   |       758       |     25096      |        25100        | 0.02  |
|  16  |   24   |   1536   |       758       |     28720      |        28724        | 0.01  |
|  16  |   24   |   2048   |      1313       |     34357      |        34363        | 0.02  |

#### Time Comparison

The duration time is the total time of 200 `optimizer.step()`.

| Head | Layers | Emb. Dim | Model Size (MB) | Adam Time (ms) | FusedAdan Time (ms) | FusedAdan/Adam (%) |
| :--: | :----: | :------: | :-------------: | :------------: | :-----------------: | :----------------: |
|  6   |   6    |   768    |       81        |      5.40      |        4.07         |        81.6        |
|  12  |   6    |   768    |       81        |      5.41      |        4.16         |        76.9        |
|  16  |   6    |   768    |       81        |      5.41      |        4.11         |        76.0        |
|  6   |   12   |   768    |       124       |      8.47      |        6.25         |        73.8        |
|  12  |   12   |   768    |       124       |      8.46      |        6.18         |        73.0        |
|  16  |   12   |   768    |       124       |      8.48      |        6.20         |        73.1        |
|  16  |   6    |   1024   |       128       |      7.57      |        6.28         |        83.0        |
|  16  |   12   |   1024   |       203       |     12.10      |        10.25        |        84.7        |
|  6   |   24   |   768    |       209       |     16.40      |        10.56        |        64.4        |
|  12  |   24   |   768    |       209       |     16.40      |        10.47        |        63.8        |
|  16  |   24   |   768    |       209       |     16.35      |        10.56        |        64.6        |
|  6   |   6    |   1536   |       248       |     15.92      |        12.29        |        77.2        |
|  12  |   6    |   1536   |       248       |     15.94      |        12.35        |        77.5        |
|  16  |   6    |   1536   |       248       |     15.94      |        12.36        |        77.5        |
|  16  |   24   |   1024   |       354       |     21.05      |        17.51        |        83.2        |
|  16  |   6    |   2048   |       407       |     25.05      |        19.84        |        79.2        |
|  6   |   12   |   1536   |       418       |     27.24      |        20.58        |        75.6        |
|  12  |   12   |   1536   |       418       |     27.25      |        20.54        |        75.4        |
|  16  |   12   |   1536   |       418       |     27.25      |        20.46        |        75.1        |
|  16  |   6    |   2560   |       603       |     36.86      |        29.55        |        80.1        |
|  16  |   12   |   2048   |       709       |     44.00      |        34.89        |        79.3        |
|  6   |   24   |   1536   |       758       |     49.87      |        37.52        |        75.2        |
|  12  |   24   |   1536   |       758       |     49.87      |        37.42        |        75.0        |
|  16  |   24   |   1536   |       758       |     49.92      |        37.56        |        75.2        |
|  16  |   24   |   2048   |      1313       |     81.81      |        64.48        |        77.9        |

## Conclusion

- The extra memory consumption does not increase linearly with the model's size.

- In most cases, FusedAdan has no additional memory footprint and the time consumption is only 80% of Adam's.

- In the extreme case, FusedAdan's additional memory footprint does not exceed 15%.
