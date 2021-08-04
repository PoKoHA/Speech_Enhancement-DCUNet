학습이 제대로 이루어지지 않고 있음 

TODO

- 코드의 수정이 필요함

> 이때까지는 STFT 후 spectrogram으로 Magnitude만 뽑아서 사용하였지만 본 논문에서는 Phase와 Magnitude를 함께 사용하여
> 좋은 결과를 만들어냄

## DCUnet

 1. Noisy Spectrogram -> Clean Spectrogram(Mask)

![1](https://user-images.githubusercontent.com/76771847/127819187-c25d1db2-0504-4c60-a0e8-2422d658e3d6.png)

 2. Tanh을 사용했을 경우 Bounded Region

![2](https://user-images.githubusercontent.com/76771847/127819335-b0467ac3-66a8-4d59-bb73-20be048ddf8f.png)

 3. Real-Img 따로 conv 수행 

![3](https://user-images.githubusercontent.com/76771847/127819646-5d76de1b-024d-4184-a4bd-3e283522ac6a.png)

 4. Architecture(Unet)
 
![4](https://user-images.githubusercontent.com/76771847/127819727-7e7c7b8e-b915-41eb-b9d3-817d215c7bda.png)

## Reference

paper: https://openreview.net/pdf?id=SkeRTsAcYm

Unet: https://arxiv.org/abs/1505.04597

Phase: https://angeloyeo.github.io/2019/10/11/Fourier_Phase.html
(Phase와 Magnitude 지식 필요)

Code: https://github.com/sweetcocoa/DeepComplexUNetPyTorch

COde: https://github.com/chanil1218/DCUnet.pytorch
