Authors: Vikash Kumar, Sarthak Srivastava, Rohit Lal, Anirban Chakraborty
Affiliation: Indian Institute of Science, Bengaluru

## Abstract

This work explores the usage of Fourier Transform for reducing the domain gap between the Source (e.g. Synthetic Image) and Target domain (e.g. Real Image) towards solving the Domain Adaptation problem. Most of the Unsupervised Domain Adaptation (UDA) algorithms reduce the global domain shift between labelled Source and unlabelled Target domain by matching the marginal distribution. UDA performance deteriorates for the cases where the domain gap between Source and Target is significant. To improve the overall performance of the existing UDA algorithms the proposed method attempts to bring the Source domain closer to the Target domain with the help of pseudo label based class consistent low-frequency swapping. This traditional image processing technique results in computational efficiency, especially compared to the state-of-the-art deep learning methods that use complex adversarial training. The proposed method Class Aware Frequency Transformation (CAFT) can easily be plugged into any existing UDA algorithm to improve its performance. We evaluate CAFT on various domain adaptation datasets and algorithms and have achieved performance gains across all the popular benchmarks.

## BibTex

If you find our work useful please cite our paper.
```
@InProceedings{Kumar_2021_ICCV,
    author    = {Kumar, Vikash and Srivastava, Sarthak and Lal, Rohit and Chakraborty, Anirban},
    title     = {CAFT: Class Aware Frequency Transform for Reducing Domain Gap},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {2525-2534}
}
```