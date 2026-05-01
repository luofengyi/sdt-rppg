# SDT
This repository is the implementation for our paper *[A Transformer-based Model with Self-distillation for Multimodal Emotion Recognition in Conversations](https://ieeexplore.ieee.org/abstract/document/10109845)*.

## Model Architecture
<!-- ![Image of SDT](fig/SDT.jpg) -->
<div align="center">
    <img src="fig/SDT.jpg" width="85%" title="SDT."</img>
</div>

## Setup
- Check the packages needed or simply run the command:
```console

pip install -r requirements.txt
```
- Download the preprocessed datasets from [here](https://drive.google.com/drive/folders/1J1mvbqQmVodNBzbiOIxRiWOtkP6qqP-K?usp=sharing), and put them into `data/`.
- If you want to extract rPPG from raw IEMOCAP videos, install extra packages:
```console
pip install opencv-python scipy
```

## Run SDT model
- Run the model on IEMOCAP dataset:
```console

bash exec_iemocap.sh
```
- Run the model on MELD dataset:
```console

bash exec_meld.sh
```

## One-Click Hyperparameter Tuning
- Use the built-in tuner to run multiple training trials automatically and keep the best result.
- Example command:
```console
python tune_one_click.py --config configs/tune_iemocap_example.json --mode grid
```
- Random search (limit number of trials):
```console
python tune_one_click.py --config configs/tune_iemocap_example.json --mode random --max-trials 20 --seed 42
```
- Output files are saved under `tuning_runs/<run_name>_<timestamp>/`:
  - `results.csv`: every trial metrics and args
  - `summary.json`: best trial summary
  - `logs/trial_*.log`: full training logs
- To tune your ULGM parameters later, add them to:
  - `fixed_args` if you want to keep them fixed
  - `param_space` if you want the tuner to search them
- ULGM alpha normalization:
  - Add `--ulgm-normalize-alpha` to normalize active modality weights to sum to 1.
  - In tuner json, set `"ulgm-normalize-alpha": [true]` in `param_space` or `fixed_args`.
- Hard-constraint mode in tuner:
  - Add `hard_constraints.enabled=true` in tuner config to filter invalid trials before training.
  - Built-in example enforces active ULGM alpha sum to 1 (with tolerance), non-negative alpha, and ignores `alpha_r` when `use-rppg=false`.

## Extract rPPG From Raw IEMOCAP Videos (Session1 Example)
```console
python extract_rppg_iemocap.py \
  --iemocap-pkl data/iemocap_multimodal_features.pkl \
  --video-dir "D:/研究课题/SDT/Session1/Session1/dialog/avi/DivX" \
  --transcription-dir "D:/研究课题/SDT/Session1/Session1/dialog/transcriptions" \
  --output-rppg-npz data/iemocap_rppg_features_ses01.npz \
  --output-pkl data/iemocap_multimodal_features_rppg_ses01.pkl \
  --session-prefix Ses01
```
- The pipeline follows: face ROI tracking -> POS rPPG -> 0.7-4.0Hz band-pass -> 64-d PSD -> quality gate -> 4-layer encoder.
- `videoRppg342` is aligned to SDT `videoVisual` dimension (`342`) and utterance timeline.
- `videoRppg1024` keeps the last encoder layer representation (`1024`).

## Acknowledgements
- Special thanks to the [COSMIC](https://github.com/declare-lab/conv-emotion) and [MMGCN](https://github.com/hujingwen6666/MMGCN) for sharing their codes and datasets.

## Citation
If you find our work useful for your research, please kindly cite our paper. Thanks!
```
@article{ma2024sdt,
  author={Ma, Hui and Wang, Jian and Lin, Hongfei and Zhang, Bo and Zhang, Yijia and Xu, Bo},
  journal={IEEE Transactions on Multimedia}, 
  title={A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations}, 
  year={2024},
  volume={26},
  number={},
  pages={776-788},
  keywords={Emotion recognition;Transformers;Oral communication;Context modeling;Task analysis;Visualization;Logic gates;Multimodal emotion recognition in conversations;intra- and inter-modal interactions;multimodal fusion;modal representation},
  doi={10.1109/TMM.2023.3271019}}

```
