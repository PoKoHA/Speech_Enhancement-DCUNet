## Spectrogram 
> 실제 Mask를 입히는 것은 Mag / Phase 아니고 Real 과 Imag 값임.
> 
> 사실 Real Image 만 제대로 Mask를 입힐 수 만 있다면 Mag, Phase도 같이 Mask 했다고 볼 수 있음. 

- **Real**

| Mix | Denoising | GT |
|---|---|---|
|![real_1](https://user-images.githubusercontent.com/76771847/129692781-6b2d7d34-cf20-4e1d-acdb-7851f8cf0e9c.png)|![output_real_1](https://user-images.githubusercontent.com/76771847/129692794-755cf1f5-e213-49ee-bc09-e355b163ce4c.png)|![real_1_GT](https://user-images.githubusercontent.com/76771847/129692783-dbe8bbbf-3ff5-445e-9473-a78d11dd746f.png)

- **Imag**

| Mix | Denoising | GT |
|---|---|---|
|![imag_1](https://user-images.githubusercontent.com/76771847/129692984-38b8434c-d0ec-4041-9bc5-fe12983cec9a.png)|![output_imag_1](https://user-images.githubusercontent.com/76771847/129692999-06fb7f63-1fc2-4035-a071-366808b81b17.png)|![imag_1_GT](https://user-images.githubusercontent.com/76771847/129693004-e1deecd3-2189-42a9-b871-73ea3bdb1dbf.png)

- **Magnitude**

| Mix | Denoising | GT |
|---|---|---|
|![mag_1](https://user-images.githubusercontent.com/76771847/129691497-df913ff3-2084-4644-a0e6-355f5e115505.png)|![output_mag_1](https://user-images.githubusercontent.com/76771847/129691666-7212095c-a854-4251-94ac-b37036819650.png)|![mag_1_GT](https://user-images.githubusercontent.com/76771847/129691096-89398036-4b89-4e0d-80c5-f2d002407172.png)|

- **Phase**

| Mix | Denoising | GT |
|---|---|---|
|![phase_1](https://user-images.githubusercontent.com/76771847/129692148-efc00d39-56c4-4006-8785-2397aef4891e.png)|![output_phase_1](https://user-images.githubusercontent.com/76771847/129692268-1e05ea33-a744-47f5-9d09-143aa38ac683.png)|![phase_1_GT](https://user-images.githubusercontent.com/76771847/129692294-b5169c89-1c55-4eb4-be03-fb1b0656f3bb.png)|

**비교적 간단하고 단조로운 Noise에 대해서는 좋은 Denoising performance를 보여주었지만 복잡하고 힘든 noise 또는 한번도 보지 못한 noise에 대해서는
denoising performance가 떨어짐**