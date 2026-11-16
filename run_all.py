"""Run the whole pipeline in order: prepare the data, extract features, run every task, draw every plot.

Every step skips what is already done, so you can stop (Ctrl-C) and run this again at any time.
Usage: python run_all.py                       16 checkpoints spread over ImageNet training (default)
       python run_all.py --checkpoints all     all 103 checkpoints (smoother curves, ~6x longer)
"""
import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoints", default="default", choices=["default", "all"])
args = ap.parse_args()

steps = [
    ["1_prepare_talktome.py"],
    ["2_extract_features.py", "--checkpoints", args.checkpoints],
    ["tasks/task0_classify_probe.py"],
    ["tasks/task1_error_structure.py"],
    ["tasks/task2_new_combinations.py"],
    ["tasks/task3_teaching_to_mastery.py"],
    ["tasks/task4_photo_to_video.py"],
    ["tasks/task5_receptive_expressive.py"],
    ["tasks/task6_maintenance.py"],
    ["plots/plot_imagenet.py"],
] + [[f"plots/plot_task{i}.py"] for i in range(7)]

for step in steps:
    print(f"\n=== python {' '.join(step)}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *step], check=True)       # stops here if a step fails
    print(f"=== done in {(time.time() - t0) / 60:.1f} min", flush=True)
print("\nAll done. Results: runs/results/, figures: runs/figures/")
