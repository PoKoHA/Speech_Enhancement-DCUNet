(2022.06.07) - 코드에 Error가 있음 차후 수정
(2021.09.01) ~~ASR Transformer  오류 수정~~

(2021.09.07) - Freq and Temporal CBAM 추가

**This**
- Encoder와 Decoder Skip Connection에 **CBAM**을 적용
- **ASR Transformer**에 사용하였던 ScheduleAdam을 사용
- Self Attention 적용한 **Discriminator(SAGAN)** 적용
- Loss를 wSDR이 아닌 **SI-SNR**로 변경


## Architecture

![1](https://user-images.githubusercontent.com/76771847/127819187-c25d1db2-0504-4c60-a0e8-2422d658e3d6.png)
![cbam](https://user-images.githubusercontent.com/76771847/131243269-834125a7-5088-455d-8380-b4c8a385570f.png)
**- DCUnet(16block)을 Base로 각 SKip-Connection 마다 CBAM(Convolutional Block Attention Module)적용**

**: Transformer에서 쓰는 Self Attention을 쓰기에는 [batch, channel=1, freq=1539, time=213]이 너무 크기 때문에 / CBAM
을 사용하여 어떤 Channel(What) 과 Spatial(Where)을 집중 할 것인지를 찾음**


![ASR-transformer](https://user-images.githubusercontent.com/76771847/131243336-3a50a027-1a40-446b-a2b5-19f95c7104c0.png)

**- DCUnet에서 나온 Mask 뒷단에다가 ASR-Transformer를 붙힘(Decoder Input = Target Spectrogram)**

**: Mask와 Target의 Spectrogram을 Encoder, Decoder Input으로 받아서 서로 MultiHead Cross Attention 적용**

**(그림과 다르게 Decoder도 Encoder와 같은 Spectrogram이므로 VGGExtractor 사용)**


![Discriminator](https://user-images.githubusercontent.com/76771847/131243460-48f69bbd-2bb9-4155-9336-bc7e2ba34fbc.png)
![sagan](https://user-images.githubusercontent.com/76771847/131243461-05a65c12-7701-47fe-8b7a-ef40484a9f9c.png)

**- Denoising된 Spectrogram을 GAN 관점으로 보면 *Fake Image*로 볼 수있다고
생각하여 Self-Attention이 적용된 Discriminator를 적용**

**:https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Divakar_Image_Denoising_via_CVPR_2017_paper.pdf**


![loss](https://user-images.githubusercontent.com/76771847/129316595-bad11735-78e6-45e1-8fa5-fe67e422423f.png)

**- DCUnet에서는 원래 wSDR을 사용하였지만 대중적으로 많이 쓰고 있다고 하는 SI-SNR
Loss를 사용**

## Reference

paper: https://openreview.net/pdf?id=SkeRTsAcYm

Unet: https://arxiv.org/abs/1505.04597

Phase: https://angeloyeo.github.io/2019/10/11/Fourier_Phase.html
(Phase와 Magnitude 지식 필요)

**Code: https://github.com/pheepa/DCUnet/blob/master/dcunet.ipynb**

Code: https://github.com/sweetcocoa/DeepComplexUNetPyTorch

Code: https://github.com/chanil1218/DCUnet.pytorch

Code: https://github.com/sweetcocoa/DeepComplexUNetPyTorch/tree/c68510a4d822f19fa366f1da84eff8c0c25ff88a

## Implement

**Training**

> `python main.py --gpu(or MultiGPU) --batch-size ** --warm-steps (batch에 따라 유연성있게) --epochs 40
> --decay_epoch 20``

