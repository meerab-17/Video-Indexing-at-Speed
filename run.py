import argparse
import glob
import json
import logging
import os
import queue
import random
import shutil
import tempfile
import threading
import time

import cv2
import pyarrow.parquet as pq
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

CLASS_IDS = [0, 2, 16, 63, 67]
CLASS_NAMES = ["person", "car", "dog", "laptop", "cell phone"]
# (frames looked at first, frames when they disagree; 0 means every frame)
LADDER = [(2, 12), (3, 12), (4, 16), (3, 0)]
LIMITS = [3.0, 5.0, 1.0]  # alarm limits in percent: object flips, people misses, people off by 2+
WINDOW = 300
FAST_RATE = 0.06
SLOW_RATE = 0.03
GROUP = 8


def summarize(counts):
    # Objects present in at least half the frames, and the median people count
    n = len(counts)
    present = [0, 0, 0, 0, 0]
    if n == 0:
        return present, 0
    for c in range(5):
        seen = 0
        for row in counts:
            if row[c] > 0:
                seen += 1
        if 2 * seen >= n:
            present[c] = 1
    people = sorted([row[0] for row in counts])
    if n % 2 == 1:
        return present, people[n // 2]
    return present, (people[n // 2 - 1] + people[n // 2]) // 2


def disagree(counts):
    # True if the sampled frames do not agree on an object or on the people count
    if len(counts) < 2:
        return False
    for c in range(5):
        seen = 0
        for row in counts:
            if row[c] > 0:
                seen += 1
        if seen > 0 and seen < len(counts):
            return True
    for row in counts:
        if row[0] != counts[0][0]:
            return True
    return False


def run_model(model, frames):
    counts = []
    if len(frames) == 0:
        return counts
    results = model.predict(frames, imgsz=640, conf=0.4, iou=0.7, classes=CLASS_IDS,
                            half=True, batch=32, verbose=False)
    for r in results:
        ids = r.boxes.cls.tolist()
        row = []
        for cid in CLASS_IDS:
            row.append(ids.count(float(cid)))
        counts.append(row)
    return counts


class Shared:
    # The rung the decoder should use when it starts a new second
    def __init__(self, rung):
        self.rung = rung


def decode_video(data, vid, q, shared, tmp_dir):
    # Puts one packet per second on the queue; returns (error, frames decoded, frames reported)
    path = os.path.join(tmp_dir, "video_%d.mp4" % os.getpid())
    with open(path, "wb") as f:
        f.write(data)
    cap = cv2.VideoCapture(path)
    decoded = 0
    reported = 0
    error = None
    try:
        if not cap.isOpened():
            return "could not decode video", 0, 0
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if count and count > 0:
            reported = int(count)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 240:
            fps = 25.0

        first_t = None
        last_t = 0.0
        cur = 0
        rung = shared.rung
        n0, m = LADDER[rung]
        j0 = 0
        jm = 0
        first = []
        extra = []
        while cap.grab():
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if first_t is None:
                first_t = t
            elif t <= last_t:
                # timestamps that stand still or go backwards are pushed forward
                t = last_t + 1.0 / fps
            last_t = t
            decoded += 1
            rel = t - first_t
            sec = int(rel)
            if sec - cur > 3600:
                raise ValueError("timestamp jump")

            while cur < sec:
                q.put(("sec", vid, cur, first, extra, rung))
                cur += 1
                rung = shared.rung
                n0, m = LADDER[rung]
                j0 = 0
                jm = 0
                first = []
                extra = []

            take_first = False
            while j0 < n0 and rel >= cur + j0 / n0:
                j0 += 1
                take_first = True
            take_extra = False
            if m == 0:
                take_extra = True
            else:
                while jm < m and rel >= cur + jm / m:
                    jm += 1
                    take_extra = True

            if take_first or take_extra:
                ok, frame = cap.retrieve()
                if ok:
                    if take_first:
                        first.append(frame)
                    else:
                        extra.append(frame)
        if decoded > 0:
            q.put(("sec", vid, cur, first, extra, rung))
    except Exception as e:
        error = "decode failed: " + str(e)[:60]
    finally:
        cap.release()
        os.remove(path)
    return error, decoded, reported


def producer(files, done, q, shared, tmp_dir, max_videos):
    handled = 0
    for path in files:
        name = os.path.basename(path)
        try:
            pf = pq.ParquetFile(path)
            row = 0
            for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
                vid = name + ":" + str(row)
                row += 1
                if vid in done:
                    continue
                if max_videos > 0 and handled >= max_videos:
                    q.put(("done",))
                    return
                handled += 1
                data = batch.column(0)[0].as_py()
                if not data:
                    q.put(("end", vid, "no video data", 0, 0))
                    continue
                try:
                    error, decoded, reported = decode_video(data, vid, q, shared, tmp_dir)
                except Exception as e:
                    error, decoded, reported = "decode failed: " + str(e)[:60], 0, 0
                q.put(("end", vid, error, decoded, reported))
        except Exception as e:
            q.put(("end", name, "could not read parquet file: " + str(e)[:60], 0, 0))
    q.put(("done",))


def audit_rate(ctl):
    if len(ctl["window"]) < WINDOW:
        return FAST_RATE
    return SLOW_RATE


def check_alarm(ctl, shared):
    w = ctl["window"]
    if len(w) < WINDOW or ctl["rung"] >= len(LADDER) - 1:
        return
    for j in range(3):
        n = 0
        for flags in w:
            if flags[j]:
                n += 1
        if 100.0 * n / len(w) > LIMITS[j]:
            ctl["rung"] += 1
            shared.rung = ctl["rung"]
            ctl["window"] = []
            ctl["switches"].append((ctl["seconds"], LADDER[ctl["rung"]]))
            print("alarm: moving to", LADDER[ctl["rung"]], "after", ctl["seconds"], "seconds", flush=True)
            return


def process_group(model, group, ctl, rng, out):
    firsts = []
    for p in group:
        firsts.extend(p["first"])
    counts = run_model(model, firsts)

    pos = 0
    info = []
    extras = []
    for p in group:
        n = len(p["first"])
        mine = counts[pos:pos + n]
        pos += n
        m = LADDER[p["rung"]][1]
        escalate = disagree(mine)
        audit = False
        if m != 0 and p["rung"] == ctl["rung"]:
            if rng.random() < audit_rate(ctl):
                audit = True
        want = (escalate or audit) and len(p["extra"]) > 0
        if want:
            extras.extend(p["extra"])
        info.append([mine, escalate, audit, want])
    extra_counts = run_model(model, extras)
    ctl["frames"] += len(firsts) + len(extras)

    pos = 0
    for k in range(len(group)):
        p = group[k]
        mine, escalate, audit, want = info[k]
        everything = mine
        if want:
            n = len(p["extra"])
            everything = mine + extra_counts[pos:pos + n]
            pos += n
        present, people = summarize(everything)

        if audit:
            if escalate:
                ctl["window"].append([False, False, False])
            else:
                present1, people1 = summarize(mine)
                ctl["window"].append([present1 != present, people1 != people,
                                      abs(people1 - people) > 1])
            if len(ctl["window"]) > WINDOW:
                ctl["window"].pop(0)

        names = []
        for c in range(5):
            if present[c] == 1:
                names.append(CLASS_NAMES[c])
        out.write(json.dumps({"video": p["vid"], "second": p["sec"],
                              "objects": names, "people": people}) + "\n")
        ctl["seconds"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start-rung", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()

    start = time.time()
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    done_path = args.output + ".done"
    state_path = args.output + ".state"

    done = set()
    rung = args.start_rung
    if args.resume and os.path.exists(done_path) and os.path.exists(args.output):
        for line in open(done_path):
            done.add(line.strip())
        if os.path.exists(state_path):
            rung = int(open(state_path).read().strip())
        # keep only the lines of finished videos
        kept = []
        for line in open(args.output):
            try:
                if json.loads(line)["video"] in done:
                    kept.append(line)
            except Exception:
                pass
        with open(args.output, "w") as f:
            f.writelines(kept)
        out = open(args.output, "a")
    else:
        out = open(args.output, "w")

    here = os.path.dirname(os.path.abspath(__file__))
    weights = os.path.join(here, "yolo11l.pt")
    if not os.path.exists(weights):
        weights = "yolo11l.pt"
    model = YOLO(weights)

    # /dev/shm can be tiny on some machines, so only use it when it has room
    tmp_dir = tempfile.gettempdir()
    if os.path.isdir("/dev/shm") and shutil.disk_usage("/dev/shm").free > 1000000000:
        tmp_dir = "/dev/shm"
    shared = Shared(rung)
    ctl = {"rung": rung, "window": [], "switches": [], "seconds": 0, "frames": 0}
    q = queue.Queue(maxsize=8)
    t = threading.Thread(target=producer, args=(files, done, q, shared, tmp_dir, args.max_videos),
                         daemon=True)
    t.start()

    rng = random.Random(0)
    pending = []
    failed = {"vid": None, "msg": None}
    videos_done = 0

    def flush():
        if len(pending) == 0:
            return
        group = list(pending)
        del pending[:]
        try:
            process_group(model, group, ctl, rng, out)
            check_alarm(ctl, shared)
        except Exception as e:
            failed["vid"] = group[0]["vid"]
            failed["msg"] = "model failed: " + str(e)[:60]

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            flush()
            item = q.get()

        if item[0] == "sec":
            if failed["vid"] == item[1]:
                continue
            pending.append({"vid": item[1], "sec": item[2], "first": item[3],
                            "extra": item[4], "rung": item[5]})
            if len(pending) >= GROUP:
                flush()
        elif item[0] == "end":
            flush()
            vid, error, decoded, reported = item[1], item[2], item[3], item[4]
            message = error
            if failed["vid"] == vid:
                message = failed["msg"]
            elif message is None:
                if decoded == 0:
                    message = "could not decode video"
                elif reported > 0 and decoded < 0.5 * reported:
                    message = "video stopped early"
            failed["vid"] = None
            if message is not None:
                out.write(json.dumps({"video": vid, "error": message}) + "\n")
            out.flush()
            if args.resume:
                with open(done_path, "a") as f:
                    f.write(vid + "\n")
                with open(state_path, "w") as f:
                    f.write(str(ctl["rung"]))
            videos_done += 1
            if videos_done % 20 == 0:
                spent = time.time() - start
                print("videos", videos_done, "| seconds", ctl["seconds"], "| model frames/s",
                      round(ctl["frames"] / spent), "| step", LADDER[ctl["rung"]],
                      "| elapsed", round(spent), "s", flush=True)
        else:
            flush()
            break

    out.close()
    spent = time.time() - start
    print("finished:", videos_done, "videos,", ctl["seconds"], "seconds,", round(spent), "s,",
          "switches", ctl["switches"], flush=True)


if __name__ == "__main__":
    main()
