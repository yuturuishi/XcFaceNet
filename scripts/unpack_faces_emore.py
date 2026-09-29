"""
把 InsightFace faces_emore (.rec/.idx) 解包成 XcFaceNet 可用的图片目录，并生成 train.txt。

用法（在项目根目录执行）：
  # 先导出 50 张样图
  python -m scripts.unpack_faces_emore --max_images 50

  # 全部解包（约 580 万张，建议后台跑）
  python -m scripts.unpack_faces_emore --all

  # 断点续传（中断后可重新执行同一命令）
  python -m scripts.unpack_faces_emore --all
"""
import argparse
import os
import struct
import time
from pathlib import Path


MAGIC = 0xCED7230A


def read_idx(idx_path: Path):
    keys, offsets = [], []
    with open(idx_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            keys.append(int(parts[0]))
            offsets.append(int(parts[1]))
    return keys, offsets


def read_record(fp, offset: int):
    fp.seek(offset)
    header = fp.read(8)
    if len(header) < 8:
        return None
    magic, length = struct.unpack("II", header)
    if magic != MAGIC:
        raise ValueError(f"Bad magic at offset {offset}: {hex(magic)}")
    payload = fp.read(length)
    pad = (4 - (length % 4)) % 4
    if pad:
        fp.read(pad)
    return payload


def unpack_header_and_image(payload: bytes):
    if len(payload) < 16:
        raise ValueError("Record too short")
    flag, label, img_id, id2 = struct.unpack("IfII", payload[:16])
    img_bytes = payload[16:]
    if not (img_bytes.startswith(b"\xff\xd8") or img_bytes.startswith(b"\x89PNG")):
        for skip in (24, 32, 48):
            candidate = payload[skip:]
            if candidate.startswith(b"\xff\xd8") or candidate.startswith(b"\x89PNG"):
                img_bytes = candidate
                break
    return int(label), img_bytes


def extract_image(payload: bytes):
    try:
        label, img_bytes = unpack_header_and_image(payload)
        if img_bytes.startswith(b"\xff\xd8") or img_bytes.startswith(b"\x89PNG"):
            return label, img_bytes
    except Exception:
        pass

    pos = payload.find(b"\xff\xd8")
    if pos < 0:
        raise ValueError("JPEG marker not found")
    label = int(struct.unpack("f", payload[4:8])[0])
    return label, payload[pos:]


def main():
    parser = argparse.ArgumentParser(description="Unpack faces_emore RecordIO for XcFaceNet")
    parser.add_argument("--src", default=r"E:\download\faces_emore", help="faces_emore directory")
    parser.add_argument("--dst", default=r"D:\datasets\face\ms1mv2\train", help="output train image directory")
    parser.add_argument("--train_txt", default=r"D:\datasets\face\ms1mv2\train.txt", help="output train.txt path")
    parser.add_argument("--max_images", type=int, default=50, help="export at most N images")
    parser.add_argument("--all", action="store_true", help="export all images")
    parser.add_argument("--log_every", type=int, default=5000, help="print progress every N images")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    train_txt = Path(args.train_txt)
    idx_path = src / "train.idx"
    rec_path = src / "train.rec"
    progress_path = dst.parent / "unpack_progress.log"

    if not idx_path.exists() or not rec_path.exists():
        raise FileNotFoundError(f"Missing train.idx/train.rec under {src}")

    keys, offsets = read_idx(idx_path)
    image_items = [(k, o) for k, o in zip(keys, offsets) if k != 0]
    if not args.all:
        image_items = image_items[: args.max_images]

    dst.mkdir(parents=True, exist_ok=True)
    train_txt.parent.mkdir(parents=True, exist_ok=True)

    print(f"Source: {src}")
    print(f"Output images: {dst}")
    print(f"Output train.txt: {train_txt}")
    print(f"Total to export: {len(image_items)}")
    print(f"Resume supported: existing jpg files will be skipped")

    saved = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    # Rewrite train.txt completely each full run to keep indexes consistent.
    # For resume, existing files are skipped but still listed.
    with open(rec_path, "rb") as fp, open(train_txt, "w", encoding="utf-8") as txt_f, open(progress_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} total={len(image_items)} ====\n")
        log_f.flush()

        for i, (key, offset) in enumerate(image_items, 1):
            try:
                # We need label for folder naming; if file already exists under unknown label,
                # still need to unpack header once, so always parse record.
                payload = read_record(fp, offset)
                if payload is None:
                    failed += 1
                    continue
                label, img_bytes = extract_image(payload)
                person_dir = dst / f"{label:06d}"
                person_dir.mkdir(parents=True, exist_ok=True)
                out_file = person_dir / f"{key}.jpg"
                abs_path = str(out_file.resolve())

                if out_file.exists() and out_file.stat().st_size > 0:
                    skipped += 1
                else:
                    out_file.write_bytes(img_bytes)
                    saved += 1

                # XcFaceNet format: class_id;absolute_path
                txt_f.write(f"{label};{abs_path}\n")

            except Exception as e:
                failed += 1
                if failed <= 10:
                    msg = f"Skip key={key}: {e}"
                    print(msg)
                    log_f.write(msg + "\n")

            if i % args.log_every == 0 or i == len(image_items):
                elapsed = max(time.time() - t0, 1e-6)
                speed = i / elapsed
                remain = (len(image_items) - i) / max(speed, 1e-6)
                msg = (
                    f"Progress {i}/{len(image_items)} "
                    f"({i * 100 / len(image_items):.2f}%) "
                    f"saved={saved} skipped={skipped} failed={failed} "
                    f"speed={speed:.1f} img/s ETA={remain / 3600:.2f}h"
                )
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()
                txt_f.flush()

        log_f.write(f"==== done {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")

    print("\nDone.")
    print(f"Saved new images: {saved}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Images folder: {dst}")
    print(f"train.txt: {train_txt}")
    print(f"Progress log: {progress_path}")


if __name__ == "__main__":
    main()
