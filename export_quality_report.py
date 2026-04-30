import argparse
import csv
from pathlib import Path

import numpy as np

import extract_rppg_iemocap as rppg


def main():
    parser = argparse.ArgumentParser(description="导出每个utterance的rPPG质量报告")
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="IEMOCAP视频目录（dialog/avi/DivX）",
    )
    parser.add_argument(
        "--transcription-dir",
        type=str,
        required=True,
        help="IEMOCAP转写目录（dialog/transcriptions）",
    )
    parser.add_argument(
        "--emo-eval-dir",
        type=str,
        required=True,
        help="IEMOCAP情感标注目录（dialog/EmoEvaluation）",
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default="Ses01",
        help="会话前缀过滤（默认Ses01）",
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="视频下采样比例，建议>=1.0避免全零片段",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="质量通过阈值，默认0.3",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/quality_report_v3.csv",
        help="质量报告输出路径",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    trans_dir = Path(args.transcription_dir)
    emo_dir = Path(args.emo_eval_dir)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    below = 0

    emo_files = sorted(emo_dir.glob(f"{args.session_prefix}*.txt"))
    for emo_file in emo_files:
        conv_id = emo_file.stem
        video_path = video_dir / f"{conv_id}.avi"
        trans_path = trans_dir / f"{conv_id}.txt"
        if not video_path.exists():
            continue

        rgb_ts, fps = rppg.extract_rgb_trace_from_video(video_path, downsample=args.downsample)
        pulse = rppg.pos_rppg(rgb_ts, fps=fps, window_sec=1.6)
        pulse_bp = rppg.bandpass_filter(pulse, fps=fps, low=0.7, high=4.0, order=3)
        T = len(pulse_bp)

        utt2span_emo, utt2emo = rppg.parse_emo_evaluation(emo_file)
        utt2span_trans = rppg.parse_transcription(trans_path)
        ordered = rppg.parse_emo_order(emo_file)

        for utt, _, _, _ in ordered:
            span = utt2span_emo.get(utt, utt2span_trans.get(utt))
            emo = utt2emo.get(utt, "unk")
            if span is None:
                q_stat = 0.0
                q_freq = 0.0
                q_total = 0.0
                passed = 0
                fs, fe = 0, 0
            else:
                s, e = span
                fs = rppg.sec_to_frame(s, fps, T)
                fe = rppg.sec_to_frame(e, fps, T)
                if fe <= fs + 2:
                    q_stat = 0.0
                    q_freq = 0.0
                    q_total = 0.0
                    passed = 0
                else:
                    seg = pulse_bp[fs:fe]
                    _, q_freq = rppg.psd_64_feature(seg, fps)
                    q_stat = 1.0 if (np.var(seg) > 0.01 and abs(rppg.skewness(seg)) < 2.0) else 0.0
                    q_total = 0.4 * q_stat + 0.6 * q_freq
                    passed = 1 if q_total >= args.threshold else 0

            total += 1
            if q_total < args.threshold:
                below += 1

            rows.append(
                {
                    "conv_id": conv_id,
                    "utt_id": utt,
                    "emo": emo,
                    "start_frame": fs,
                    "end_frame": fe,
                    "num_frames": max(0, fe - fs),
                    "Q_stat": f"{q_stat:.6f}",
                    "Q_freq": f"{q_freq:.6f}",
                    "Q_total": f"{q_total:.6f}",
                    "passed": passed,
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conv_id",
                "utt_id",
                "emo",
                "start_frame",
                "end_frame",
                "num_frames",
                "Q_stat",
                "Q_freq",
                "Q_total",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ratio = below / total if total > 0 else 0.0
    print(f"saved_csv: {out_csv}")
    print(f"total_utterances: {total}")
    print(f"below_threshold_count: {below}")
    print(f"below_threshold_ratio: {ratio:.6f}")
    print(f"threshold: {args.threshold}")


if __name__ == "__main__":
    main()
