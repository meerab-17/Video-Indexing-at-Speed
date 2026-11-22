# Video indexing with YOLO11l

Reads FineVideo parquet files and writes, for every second of every video, which of
person / car / dog / laptop / cell phone appear and how many people are visible.

## System packages
```
sudo apt-get install -y libgl1 libglib2.0-0
```

## Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation (needs internet, run once, not timed)
Downloads the official `yolo11l.pt` into the repository folder.
```
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run
```
python run.py --input <folder of .parquet files> --output results.jsonl
```
Optional flags: `--resume` (continue an interrupted run; writes `.done` and `.state` files next to
the output), `--max-videos N` (stop after N videos, for testing).

## Output
One JSON line per second per video, for example
`{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}`.
A video that cannot be processed gets one line with an `error` field.

## How it works
- A decoder thread reads each video with OpenCV (frames are rotated by the file's tag) and keeps
  only the frames needed for each second, picked by timestamp. A bounded queue holds a few seconds at
  a time, so decoding and the GPU work overlap without using unbounded memory.
- Each second starts with 2 frames. If they disagree on an object or on the people count, more frames
  (up to 12) are run for that second. Detections use YOLO11l at 640, confidence 0.4, fp16.
- About 6% of seconds (3% after the first 300 audits) are also run with the full frame set as an
  audit. If the audits show the cheap setting disagreeing too often, the script moves to a more
  thorough setting (3 to 12 frames, then 4 to 16, then every frame). It never moves back.
- Damaged videos and unreadable parquet files produce an error line and the run continues.

## How the included results were produced
`results.jsonl.gz` was made on Kaggle (2 x T4) by `kaggle_full_run.ipynb`: two copies of `run.py`,
one per GPU, each on half of the 50 files, then merged. See `HARDWARE.md`. Those results used the
CUDA 13 build of torch; `requirements.txt` pins the CUDA 12.6 build because the grading machine has
a CUDA 12 driver.
