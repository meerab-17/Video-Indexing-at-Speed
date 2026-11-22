# Hardware used for the full 50-file run

- Where: Kaggle notebook (Save & Run All)
- GPU: 2 x NVIDIA Tesla T4, 15 GB each (one copy of run.py per GPU)
- CPU: Intel Xeon @ 2.00GHz, 4 vCPUs
- RAM: 31 GB
- Disk: SSD (rotational flag 0 on the disks listed); videos were kept on temporary disk
- OS: Ubuntu 24.04.4 LTS, Python 3.13
- Total time start to finish: 171.5 minutes for the whole notebook (package install, a 3-video smoke test, downloads and processing). The two GPUs ran in parallel on 25 files each, in five chunks of 5 files that took 31 to 36 minutes each.

Development and testing of the script (smaller runs on a few videos) was done on a single Tesla T4 in Google Colab.
