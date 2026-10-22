"""擦掉柱状图「柱顶印值」，生成无印值测试图。

思路（不依赖柱身颜色）：
  1. 深色像素（文字/轴线/误差棒都是深色）→ 连通域；
  2. 分类：长横线/长竖线（轴/框）、细高线（误差棒/刻度）→ 保留；
     其余小方块 = 文字 → 候选擦除；
  3. 只擦「绘图区内部」的文字：其左侧有竖线、下方有横线（= 被轴框住）。
     轴刻度数字在框外（左/下），天然保留。

用法：
    python 对比实验/blindset/erase_values.py <输入图> <输出图> [dark_thr]
"""
import sys
from collections import deque

import numpy as np
from PIL import Image


def components(mask: np.ndarray):
    """8-连通域 → [ [(y,x), ...] ]。"""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = []
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            q = deque([(i, j)])
            visited[i, j] = True
            pix = []
            while q:
                y, x = q.popleft()
                pix.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
            out.append(pix)
    return out


def erase(src: str, dst: str, dark_thr: int = 115, verbose: bool = True) -> int:
    img = Image.open(src).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    dark = arr.max(axis=2) < dark_thr

    comps = components(dark)
    lines_v, lines_h, texts = [], [], []
    for pix in comps:
        ys = [p[0] for p in pix]
        xs = [p[1] for p in pix]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        ch, cw = y1 - y0 + 1, x1 - x0 + 1
        if cw > 0.25 * w:                       # 长横线（x 轴/上框）
            lines_h.append((y0, y1, x0, x1))
        elif ch > 0.20 * h:                     # 长竖线（y 轴/侧框）
            lines_v.append((y0, y1, x0, x1))
        elif ch >= 3 * cw and ch > 15:          # 细高（误差棒/刻度线）→ 保留
            continue
        else:                                   # 小方块 = 文字候选
            texts.append((y0, y1, x0, x1))

    out = arr.copy()
    erased = 0
    for y0, y1, x0, x1 in texts:
        has_left_line = any(v[3] < x0 and v[3] > x0 - 200 for v in lines_v)
        has_below_line = any(hh[0] > y1 and hh[0] < y1 + 300 for hh in lines_h)
        if has_left_line and has_below_line:
            out[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = 255
            erased += 1
    Image.fromarray(out).save(dst)
    if verbose:
        print(f"{src} -> {dst}: 候选文字 {len(texts)}，擦除 {erased}（线: 横{len(lines_h)} 竖{len(lines_v)}）")
    return erased


if __name__ == "__main__":
    s = sys.argv[1]
    d = sys.argv[2]
    thr = int(sys.argv[3]) if len(sys.argv) > 3 else 115
    erase(s, d, thr)
