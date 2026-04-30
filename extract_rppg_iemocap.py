import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("需要先安装 opencv-python 才能运行本脚本。") from exc

try:
    from scipy import signal
except ImportError as exc:
    raise ImportError("需要先安装 scipy 才能运行本脚本。") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("需要先安装 matplotlib 才能绘制时序波形图与频谱图。") from exc


TRANS_RE = re.compile(
    r"^(?P<utt>\S+)\s+\[(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)\]:"
)
EMO_RE = re.compile(
    r"^\[(?P<start>\d+\.\d+)\s*-\s*(?P<end>\d+\.\d+)\]\s+(?P<utt>\S+)\s+(?P<emo>\w+)\s+\["
)


def load_iemocap_pickle(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def parse_transcription(trans_path: Path) -> Dict[str, Tuple[float, float]]:
    utt2span = {}
    if not trans_path.exists():
        return utt2span
    for line in trans_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = TRANS_RE.match(line.strip())
        if m is None:
            continue
        utt = m.group("utt")
        utt2span[utt] = (float(m.group("start")), float(m.group("end")))
    return utt2span


def parse_emo_evaluation(emo_path: Path):
    utt2span = {}
    utt2emo = {}
    if not emo_path.exists():
        return utt2span, utt2emo
    for line in emo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = EMO_RE.match(line.strip())
        if m is None:
            continue
        utt = m.group("utt")
        utt2span[utt] = (float(m.group("start")), float(m.group("end")))
        utt2emo[utt] = m.group("emo")
    return utt2span, utt2emo


def parse_emo_order(emo_path: Path):
    entries = []
    if not emo_path.exists():
        return entries
    for line in emo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = EMO_RE.match(line.strip())
        if m is None:
            continue
        entries.append(
            (
                m.group("utt"),
                float(m.group("start")),
                float(m.group("end")),
                m.group("emo"),
            )
        )
    return entries


def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-8
    return inter / union


def choose_stable_face(faces, prev_face, min_iou=0.2):
    if len(faces) == 0:
        return None
    # 优先最大框，若有历史框则倾向时序一致
    areas = [w * h for (_, _, w, h) in faces]
    best = faces[int(np.argmax(areas))]
    if prev_face is None:
        return best
    ious = [iou_xywh(face, prev_face) for face in faces]
    best_idx = int(np.argmax(ious))
    if ious[best_idx] >= min_iou:
        return faces[best_idx]
    return best


def face_skin_roi_mean_rgb(frame_bgr, face_xywh):
    x, y, w, h = face_xywh
    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    face = frame_bgr[y1:y2, x1:x2]
    # 简化皮肤ROI：去除边缘与下巴，保留中上区域（额头+脸颊）
    fh, fw = face.shape[:2]
    rx1, rx2 = int(0.20 * fw), int(0.80 * fw)
    ry1, ry2 = int(0.15 * fh), int(0.75 * fh)
    roi = face[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1, 3).mean(axis=0)
    return rgb.astype(np.float32)


def interpolate_nan_rgb(rgb_ts: np.ndarray) -> np.ndarray:
    if len(rgb_ts) == 0:
        return rgb_ts
    out = rgb_ts.copy()
    for c in range(3):
        x = out[:, c]
        isnan = np.isnan(x)
        if np.all(isnan):
            out[:, c] = 0.0
            continue
        idx = np.arange(len(x))
        out[isnan, c] = np.interp(idx[isnan], idx[~isnan], x[~isnan])
    return out


def pos_rppg(rgb_ts: np.ndarray, fps: float, window_sec: float = 1.6) -> np.ndarray:
    # POS算法（Wang et al.），输入 Tx3，输出长度T的一维rPPG
    T = rgb_ts.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.float32)
    w = max(4, int(window_sec * fps))
    pulse = np.zeros(T, dtype=np.float32)
    proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
    for n in range(max(1, T - w + 1)):
        C = rgb_ts[n:n + w].T
        mean_c = np.mean(C, axis=1, keepdims=True) + 1e-6
        Cn = C / mean_c - 1.0
        S = proj @ Cn
        std0 = np.std(S[0]) + 1e-6
        std1 = np.std(S[1]) + 1e-6
        h = S[0] + (std0 / std1) * S[1]
        h = h - np.mean(h)
        pulse[n:n + w] += h.astype(np.float32)
    pulse = pulse - np.mean(pulse)
    pulse = pulse / (np.std(pulse) + 1e-6)
    return pulse.astype(np.float32)


def bandpass_filter(sig: np.ndarray, fps: float, low=0.7, high=4.0, order=3) -> np.ndarray:
    if len(sig) < 8:
        return np.zeros_like(sig, dtype=np.float32)
    nyq = 0.5 * fps
    low_n = max(1e-4, low / nyq)
    high_n = min(0.999, high / nyq)
    if low_n >= high_n:
        return sig.astype(np.float32)
    b, a = signal.butter(order, [low_n, high_n], btype="band")
    try:
        out = signal.filtfilt(b, a, sig).astype(np.float32)
    except ValueError:
        out = sig.astype(np.float32)
    return out


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.mean(x))
    s = float(np.std(x) + 1e-6)
    return float(np.mean(((x - m) / s) ** 3))


def psd_64_feature(seg: np.ndarray, fps: float) -> Tuple[np.ndarray, float]:
    if len(seg) < max(8, int(0.5 * fps)):
        return np.zeros(64, dtype=np.float32), 0.0
    win = np.hanning(len(seg))
    seg = (seg - np.mean(seg)) * win
    freqs, power = signal.welch(seg, fs=fps, nperseg=min(len(seg), 256))
    band_mask = (freqs >= 0.7) & (freqs <= 4.0)
    if np.sum(band_mask) < 2:
        return np.zeros(64, dtype=np.float32), 0.0
    fb = freqs[band_mask]
    pb = power[band_mask] + 1e-12
    target_f = np.linspace(0.7, 4.0, 64)
    feat = np.interp(target_f, fb, pb).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-6)

    # 频域质量：峰值集中度
    q_freq = float(np.max(feat) / (np.sum(feat) + 1e-6))
    return feat, q_freq


class FixedRPPGEncoder:
    """
    4层线性映射编码器：
    64 -> 256 -> 512 -> 768 -> 1024
    再经对齐头 1024 -> 342 以对齐SDT视频模态维度。
    """

    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        self.w1 = rng.normal(0, 0.05, size=(64, 256)).astype(np.float32)
        self.w2 = rng.normal(0, 0.04, size=(256, 512)).astype(np.float32)
        self.w3 = rng.normal(0, 0.03, size=(512, 768)).astype(np.float32)
        self.w4 = rng.normal(0, 0.03, size=(768, 1024)).astype(np.float32)
        self.w_align = rng.normal(0, 0.04, size=(1024, 342)).astype(np.float32)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0.0, dtype=np.float32)

    def encode_1024(self, x64):
        h1 = self.relu(x64 @ self.w1)
        h2 = self.relu(h1 @ self.w2)
        h3 = self.relu(h2 @ self.w3)
        h4 = self.relu(h3 @ self.w4)
        return h4.astype(np.float32)

    def align_342(self, h1024):
        z = h1024 @ self.w_align
        z = z / (np.linalg.norm(z) + 1e-6)
        return z.astype(np.float32)


def extract_rgb_trace_from_video(video_path: Path, downsample=0.35):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-6:
        fps = 30.0

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    prev_face = None
    fail_cnt = 0
    rgbs = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if downsample != 1.0:
            frame = cv2.resize(
                frame, None, fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = []
        # 多尺度检测（0.4, 0.5, 0.6）通过最小人脸占比近似实现
        try:
            for min_ratio in (0.4, 0.5, 0.6):
                ms = int(min(gray.shape[:2]) * min_ratio * 0.2)
                det = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=4,
                    minSize=(max(24, ms), max(24, ms)),
                )
                if len(det) > 0:
                    faces.extend(det.tolist())
        except cv2.error:
            faces = []

        face = choose_stable_face(faces, prev_face)
        if face is None:
            fail_cnt += 1
            if prev_face is not None and fail_cnt <= 50:
                face = prev_face
            else:
                rgbs.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                prev_face = None
                continue
        else:
            fail_cnt = 0
            prev_face = face

        rgb = face_skin_roi_mean_rgb(frame, face)
        if rgb is None:
            rgbs.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
        else:
            rgbs.append(rgb)

    cap.release()
    rgb_ts = np.asarray(rgbs, dtype=np.float32)
    rgb_ts = interpolate_nan_rgb(rgb_ts)
    return rgb_ts, fps


def sec_to_frame(sec: float, fps: float, max_len: int) -> int:
    idx = int(round(sec * fps))
    return max(0, min(max_len, idx))


def pick_span(utt, emo_span_map, trans_span_map):
    if utt in emo_span_map:
        return emo_span_map[utt]
    return trans_span_map.get(utt, None)


def plot_wave_and_spectrum(seg: np.ndarray, fps: float, save_path: Path, title: str):
    if len(seg) < 4:
        return
    t = np.arange(len(seg), dtype=np.float32) / max(fps, 1e-6)
    nfft = int(2 ** np.ceil(np.log2(max(16, len(seg)))))
    sp = np.abs(np.fft.rfft(seg - np.mean(seg), n=nfft))
    fr = np.fft.rfftfreq(nfft, d=1.0 / max(fps, 1e-6))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(t, seg, linewidth=1.0)
    axes[0].set_title(f"{title} | rPPG Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    band = (fr >= 0.0) & (fr <= 6.0)
    axes[1].plot(fr[band], sp[band], linewidth=1.0)
    axes[1].set_title(f"{title} | Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].axvline(0.7, color="r", linestyle="--", linewidth=0.8)
    axes[1].axvline(4.0, color="r", linestyle="--", linewidth=0.8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def build_conv_rppg_features(
    conv_id: str,
    utterance_ids: List[str],
    rgb_ts: np.ndarray,
    fps: float,
    utt2span_emo: Dict[str, Tuple[float, float]],
    utt2span_trans: Dict[str, Tuple[float, float]],
    utt2emo: Dict[str, str],
    encoder: FixedRPPGEncoder,
    plot_dir: Path,
    plot_limit_per_conv: int,
):
    pulse = pos_rppg(rgb_ts, fps=fps, window_sec=1.6)
    pulse_bp = bandpass_filter(pulse, fps=fps, low=0.7, high=4.0, order=3)
    T = len(pulse_bp)

    feats_342 = []
    feats_1024 = []
    aligned_meta = []
    plotted = 0
    for utt in utterance_ids:
        span = pick_span(utt, utt2span_emo, utt2span_trans)
        emo = utt2emo.get(utt, "unk")
        if span is None:
            feats_342.append(np.zeros(342, dtype=np.float32))
            feats_1024.append(np.zeros(1024, dtype=np.float32))
            aligned_meta.append((utt, emo, None, None, 0))
            continue
        s, e = span
        fs = sec_to_frame(s, fps, T)
        fe = sec_to_frame(e, fps, T)
        if fe <= fs + 2:
            feats_342.append(np.zeros(342, dtype=np.float32))
            feats_1024.append(np.zeros(1024, dtype=np.float32))
            aligned_meta.append((utt, emo, s, e, 0))
            continue

        seg = pulse_bp[fs:fe]
        x64, q_freq = psd_64_feature(seg, fps)
        q_stat = 1.0 if (np.var(seg) > 0.01 and abs(skewness(seg)) < 2.0) else 0.0
        q_total = 0.4 * q_stat + 0.6 * q_freq
        if q_total < 0.3:
            x64 = np.zeros(64, dtype=np.float32)

        h1024 = encoder.encode_1024(x64)
        z342 = encoder.align_342(h1024)
        feats_1024.append(h1024)
        feats_342.append(z342)
        aligned_meta.append((utt, emo, s, e, 1))

        if plotted < plot_limit_per_conv:
            fig_name = f"{conv_id}__{utt}__{emo}.png"
            plot_wave_and_spectrum(seg, fps, plot_dir / conv_id / fig_name, f"{conv_id} | {utt} | {emo}")
            plotted += 1

    return (
        np.asarray(feats_342, dtype=np.float32),
        np.asarray(feats_1024, dtype=np.float32),
        aligned_meta,
    )


def main():
    parser = argparse.ArgumentParser(description="IEMOCAP原始视频rPPG特征提取并对齐SDT视觉模态")
    parser.add_argument(
        "--iemocap-pkl",
        type=str,
        default="data/iemocap_multimodal_features.pkl",
        help="SDT原始IEMOCAP特征pkl路径",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="IEMOCAP DivX目录（例如 Session1/.../dialog/avi/DivX）",
    )
    parser.add_argument(
        "--transcription-dir",
        type=str,
        required=True,
        help="IEMOCAP transcriptions目录（例如 Session1/.../dialog/transcriptions）",
    )
    parser.add_argument(
        "--emo-eval-dir",
        type=str,
        required=True,
        help="IEMOCAP EmoEvaluation目录（例如 Session1/.../dialog/EmoEvaluation）",
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default="data/iemocap_multimodal_features_rppg.pkl",
        help="输出新pkl（将videoVisual替换为对齐后的rPPG 342维）",
    )
    parser.add_argument(
        "--output-rppg-npz",
        type=str,
        default="data/iemocap_rppg_features.npz",
        help="额外输出rPPG特征字典（342维与1024维）",
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default="Ses01",
        help="仅处理指定session前缀（Session1对应Ses01）",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=0,
        help="调试用，>0时仅处理前N个会话",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="data/rppg_plots",
        help="时序波形图和频谱图输出目录",
    )
    parser.add_argument(
        "--plot-limit-per-conv",
        type=int,
        default=8,
        help="每个会话最多绘制多少个utterance图",
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=0.35,
        help="逐帧处理时的视频缩放比例，越小越省内存",
    )
    parser.add_argument(
        "--alignment-report",
        type=str,
        default="data/rppg_alignment_report_ses01.csv",
        help="时序与标签对齐报告CSV",
    )
    args = parser.parse_args()

    pkl_path = Path(args.iemocap_pkl)
    video_dir = Path(args.video_dir)
    trans_dir = Path(args.transcription_dir)
    emo_dir = Path(args.emo_eval_dir)
    plot_dir = Path(args.plot_dir)
    align_report = Path(args.alignment_report)
    if not pkl_path.exists():
        raise FileNotFoundError(f"未找到pkl: {pkl_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"未找到视频目录: {video_dir}")
    if not trans_dir.exists():
        raise FileNotFoundError(f"未找到转写目录: {trans_dir}")
    if not emo_dir.exists():
        raise FileNotFoundError(f"未找到情感标注目录: {emo_dir}")

    data = None
    videoIDs = {}
    videoVisual = {}
    using_pkl = True
    try:
        data = list(load_iemocap_pickle(pkl_path))
        videoIDs = data[0]
        videoVisual = data[8]
    except MemoryError:
        using_pkl = False
        print("[WARN] 内存不足，无法加载完整pkl。改为基于EmoEvaluation直接提取并绘图。")
        for emo_file in sorted(emo_dir.glob("*.txt")):
            conv_id = emo_file.stem
            if not conv_id.startswith(args.session_prefix):
                continue
            ordered = parse_emo_order(emo_file)
            if len(ordered) == 0:
                continue
            videoIDs[conv_id] = [x[0] for x in ordered]

    encoder = FixedRPPGEncoder(seed=42)
    videoRppg342 = {}
    videoRppg1024 = {}
    processed = 0
    report_rows = ["conv_id,utt_id,emo,start_sec,end_sec,aligned_flag"]

    for conv_id, utt_list in videoIDs.items():
        if not conv_id.startswith(args.session_prefix):
            continue
        if args.max_conversations > 0 and processed >= args.max_conversations:
            break
        video_path = video_dir / f"{conv_id}.avi"
        trans_path = trans_dir / f"{conv_id}.txt"
        emo_path = emo_dir / f"{conv_id}.txt"

        # 为确保与SDT时序严格一致，输出长度总是与utterance_ids一致
        fallback_342 = np.zeros((len(utt_list), 342), dtype=np.float32)
        fallback_1024 = np.zeros((len(utt_list), 1024), dtype=np.float32)

        if (not video_path.exists()) or (not trans_path.exists()):
            videoRppg342[conv_id] = fallback_342
            videoRppg1024[conv_id] = fallback_1024
            continue

        print(f"[rPPG] 处理: {conv_id}")
        rgb_ts, fps = extract_rgb_trace_from_video(video_path, downsample=args.downsample)
        utt2span_trans = parse_transcription(trans_path)
        utt2span_emo, utt2emo = parse_emo_evaluation(emo_path)
        feat342, feat1024, aligned_meta = build_conv_rppg_features(
            conv_id,
            utt_list,
            rgb_ts,
            fps,
            utt2span_emo,
            utt2span_trans,
            utt2emo,
            encoder,
            plot_dir=plot_dir,
            plot_limit_per_conv=args.plot_limit_per_conv,
        )

        if feat342.shape[0] != len(utt_list):
            feat342 = fallback_342
            feat1024 = fallback_1024

        videoRppg342[conv_id] = feat342
        videoRppg1024[conv_id] = feat1024
        for row in aligned_meta:
            utt, emo, start_sec, end_sec, ok = row
            s = "" if start_sec is None else f"{start_sec:.4f}"
            e = "" if end_sec is None else f"{end_sec:.4f}"
            report_rows.append(f"{conv_id},{utt},{emo},{s},{e},{ok}")
        processed += 1

    # 1) 保存独立rPPG结果
    np.savez_compressed(
        args.output_rppg_npz,
        videoRppg342=videoRppg342,
        videoRppg1024=videoRppg1024,
    )

    # 2) 若成功加载pkl，则生成可直接用于SDT的新pkl
    if using_pkl and data is not None:
        new_videoVisual = dict(videoVisual)
        for conv_id, feat in videoRppg342.items():
            if conv_id in new_videoVisual:
                new_videoVisual[conv_id] = feat
        data[8] = new_videoVisual
        with open(args.output_pkl, "wb") as f:
            pickle.dump(tuple(data), f)

    align_report.parent.mkdir(parents=True, exist_ok=True)
    align_report.write_text("\n".join(report_rows), encoding="utf-8")

    print(f"完成。处理会话数: {processed}")
    print(f"rPPG字典输出: {args.output_rppg_npz}")
    if using_pkl:
        print(f"替换视觉模态后的pkl: {args.output_pkl}")
    else:
        print("本次未生成替换视觉模态pkl（原因：内存不足未加载原始pkl）。")
    print(f"时序与标签对齐报告: {align_report}")
    print(f"时序波形图与频谱图目录: {plot_dir}")


if __name__ == "__main__":
    main()
