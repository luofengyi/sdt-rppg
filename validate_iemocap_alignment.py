import argparse
import pickle
from pathlib import Path

import numpy as np


def load_rppg_map(npz_path: Path):
    if npz_path is None:
        return None
    pack = np.load(npz_path, allow_pickle=True)
    if "videoRppg342" in pack:
        return pack["videoRppg342"].item()
    if "videoRppg1024" in pack:
        return pack["videoRppg1024"].item()
    return None


def main():
    parser = argparse.ArgumentParser(description="校验IEMOCAP多模态与rPPG对齐情况")
    parser.add_argument(
        "--iemocap-pkl",
        type=str,
        required=True,
        help="IEMOCAP特征pkl路径（包含text/audio/visual）",
    )
    parser.add_argument(
        "--session-prefix",
        type=str,
        default="Ses01",
        help="要校验的会话前缀",
    )
    parser.add_argument(
        "--rppg-npz",
        type=str,
        default="",
        help="可选：rPPG npz路径（用于额外校验videoRppg342长度）",
    )
    args = parser.parse_args()

    pkl_path = Path(args.iemocap_pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"未找到pkl: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    videoIDs = data[0]
    videoText = data[3]
    videoAudio = data[7]
    videoVisual = data[8]
    rppg_map = load_rppg_map(Path(args.rppg_npz)) if args.rppg_npz else None

    keys = [k for k in videoIDs.keys() if k.startswith(args.session_prefix)]
    if len(keys) == 0:
        print(f"[WARN] 没有找到前缀为 {args.session_prefix} 的会话。")
        return

    bad_core = []
    bad_rppg = []
    dim_visual = set()
    dim_text = set()
    dim_audio = set()
    dim_rppg = set()

    for k in keys:
        n_ids = len(videoIDs[k])
        t = np.asarray(videoText[k])
        a = np.asarray(videoAudio[k])
        v = np.asarray(videoVisual[k])

        t_len = t.shape[0] if t.ndim >= 2 else 0
        a_len = a.shape[0] if a.ndim >= 2 else 0
        v_len = v.shape[0] if v.ndim >= 2 else 0

        if not (n_ids == t_len == a_len == v_len):
            bad_core.append((k, n_ids, t_len, a_len, v_len))

        if t.ndim >= 2:
            dim_text.add(t.shape[1])
        if a.ndim >= 2:
            dim_audio.add(a.shape[1])
        if v.ndim >= 2:
            dim_visual.add(v.shape[1])

        if rppg_map is not None:
            r = np.asarray(rppg_map.get(k, np.zeros((0, 0), dtype=np.float32)))
            r_len = r.shape[0] if r.ndim >= 2 else 0
            if r.ndim >= 2:
                dim_rppg.add(r.shape[1])
            if r_len != n_ids:
                bad_rppg.append((k, n_ids, r_len))

    print("=== 对齐校验结果 ===")
    print(f"session_prefix: {args.session_prefix}")
    print(f"会话数量: {len(keys)}")
    print(f"text维度集合: {sorted(dim_text)}")
    print(f"audio维度集合: {sorted(dim_audio)}")
    print(f"visual维度集合: {sorted(dim_visual)}")
    if rppg_map is not None:
        print(f"rppg维度集合: {sorted(dim_rppg)}")

    if len(bad_core) == 0:
        print("[OK] 图像/语音/文本 与 videoIDs 长度全部一致。")
    else:
        print(f"[FAIL] 发现 {len(bad_core)} 个会话核心模态未对齐：")
        for row in bad_core[:20]:
            print(" ", row)

    if rppg_map is not None:
        if len(bad_rppg) == 0:
            print("[OK] rPPG 与 videoIDs 长度全部一致。")
        else:
            print(f"[FAIL] 发现 {len(bad_rppg)} 个会话rPPG未对齐：")
            for row in bad_rppg[:20]:
                print(" ", row)


if __name__ == "__main__":
    main()
