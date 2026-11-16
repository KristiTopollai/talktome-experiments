# TALKtoME learning trajectories

Tests 103 checkpoints of a ResNet-50 trained on ImageNet
([LSL-datasets/resnet50-imagenet-trajectory](https://huggingface.co/LSL-datasets/resnet50-imagenet-trajectory))
on the [TALKtoME](https://huggingface.co/datasets/LSL-datasets/TALKtoME) concepts.
Each script's docstring explains what it computes.

## Run everything

```bash
pip install -r requirements.txt
python run_all.py                      # 16 checkpoints, ~2 h on a laptop CPU
python run_all.py --checkpoints all    # all 103 checkpoints, ~6x longer
```

- **Disk:** you need ~20 GB of free space. TALKtoME (~16 GB) and the checkpoints are downloaded automatically.
- **Outputs:** results go to `runs/results/` and figures to `runs/figures/`.
- **Restarting:** you can stop at any time and rerun; finished work is skipped.
- **GPU:** one is used automatically if available, but none is needed.

## Run one step at a time

Always run from the repository root.

```bash
python 1_prepare_talktome.py                 # download TALKtoME, resize photos, extract video frames
python 2_extract_features.py                 # download checkpoints, compute features
python tasks/task0_classify_probe.py         # Task 0: classification (needed by Tasks 1 and 5)
python tasks/task1_error_structure.py        # Task 1: which wrong answers
python tasks/task2_new_combinations.py       # Task 2: new action-object combinations
python tasks/task3_teaching_to_mastery.py    # Task 3: sessions to mastery
python tasks/task4_photo_to_video.py         # Task 4: taught with photos, tested on videos
python tasks/task5_receptive_expressive.py   # Task 5: receptive vs expressive
python tasks/task6_maintenance.py            # Task 6: maintenance and interference
python plots/plot_task3.py                   # one plot script per task (plot_task0 ... plot_task6, plot_imagenet)
```

Fine-tuning the whole network (optional) needs an NVIDIA GPU, e.g. Google Colab:
```bash
python tasks/task0_classify_finetune.py --checkpoints step0000000 step0112590
```

## Options

- **Data location:** set `TALKTOME_DIR` (default `data/TALKtoME`) and `RUNS_DIR` (default `runs`) to keep data elsewhere.
- **Checkpoints:** in `2_extract_features.py`, choose them with `--checkpoints default`, `--checkpoints all`, or by name (e.g. `step0112590`).
