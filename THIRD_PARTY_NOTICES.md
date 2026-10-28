# Third-party code and license scope

The root MIT license covers original NAGO code and documentation. It does not
replace the licenses of the components below. Installed dependencies keep their
own licenses; consult their distributions for those terms.

| Released component | Source and copyright | Applicable terms |
| --- | --- | --- |
| `src/functional.py`, `src/utilities.py`: upstream portions and modifications | [PCFM](https://github.com/cpfpengfei/PCFM), modifications © 2025 Pengfei Cai and Utkarsh; original portions © Amazon.com, Inc. or its affiliates, from [ECI-sampling](https://github.com/amazon-science/ECI-sampling) | [PCFM MIT](licenses/PCFM-MIT.txt), [Apache 2.0](licenses/Apache-2.0.txt), and [upstream notice](licenses/PCFM-NOTICE.txt). Existing file notices are retained. NAGO adapts the flow-matching and utility code for its experiments. |
| `models/unet.py`, `models/unet1d.py`: adapted U-Net implementations | [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), © 2020 Phil Wang | [MIT](licenses/Phil-Wang-MIT.txt). NAGO adapts the models for field experiments. |
| Embedding code referenced inside the U-Net implementations | [v-diffusion-jax](https://github.com/crowsonkb/v-diffusion-jax), © 2021 Katherine Crowson and John David Pressman | [MIT](licenses/Crowson-Pressman-MIT.txt). See the source comments for the referenced implementation. |

| `models/fno.py:get_time_embedding`: adapted sinusoidal embedding | [Phil Wang's SinusoidalPosEmb](https://github.com/lucidrains/denoising-diffusion-pytorch), © 2020 Phil Wang | [MIT](licenses/Phil-Wang-MIT.txt). NAGO uses a configurable frequency scale, float32 time input and odd-dimension zero padding. |
