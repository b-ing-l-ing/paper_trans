"""程序化擦除柱顶印值 —— 赤泥基篇（矢量图专用）。

比 PS 手工强在哪：无对齐误差、可复现、不会手抖擦到柱子。

原理（只对「矢量图」有效，位图请看 README）：
1. 柱子 = 底边全部贴同一条基线的填充矩形 → 反推基线，捞出所有柱子。
2. 文字 = 填充色固定为近黑 (0.1367, 0.1220, 0.1253) 的 path。
   柱子的填充色只有白/浅灰/深灰，**永远不是这个色** → 这条判据比「字形大小」硬得多
   （字形常被 PDF 拆成 h=1.6~2.6 的碎片，按尺寸过滤必漏）。
3. 每根柱子的柱顶上方开一个窗口，落在窗口内的文字 path 就是它的标签。
   ⚠️ 窗口下缘要放到柱顶**之下** 6pt：旋转文字的小数点在字符串末尾，
   会落在柱顶边缘下方，切太紧会漏掉一个 `2026-10-16` 实测踩过。

用法：改 PDF / 页码 / CLIP / 输出名后直接跑。
"""
import fitz
from PIL import Image, ImageDraw

PDF = "PDFs/赤泥基全固废矿井充填材料的制备与性能研究.pdf"
PAGE = 3                          # 0-based
CLIP = fitz.Rect(98, 196, 516, 357.5)
ZOOM = 6
OUT = "对比实验/heldout/erased/rm_fig3.png"
# 额外窗口：零高度柱（有印值但没柱体）没有柱子可挂靠，只能手填
# 图3(a) w(CS)=5% 的 3d=0 就是这种
EXTRA_WINDOWS = [fitz.Rect(143, 306, 151, 318)]

TEXT_FILL = (0.13669031858444214, 0.12195010483264923, 0.1252918243408203)
TOL = 0.02


def is_text(it):
    f = it.get("fill")
    return f is not None and all(abs(a - b) < TOL for a, b in zip(f, TEXT_FILL))


def main():
    d = fitz.open(PDF)
    pg = d[PAGE]
    drw = pg.get_drawings()

    # 1) 找柱子：先定基线（所有柱子底边重合的那条线），再按「宽一致 + 底边贴基线」捞
    cand = [it["rect"] for it in drw if it["type"] in ("f", "fs") and 6.0 < it["rect"].width < 6.9]
    base = max(set(round(r.y1, 1) for r in cand), key=lambda y: sum(1 for r in cand if abs(r.y1 - y) < 0.6))
    bars = sorted([r for r in cand if abs(r.y1 - base) < 0.6], key=lambda r: r.x0)

    # 2) 文字路径
    text_paths = [it["rect"] for it in drw if is_text(it)]

    # 3) 每根柱子开窗口，取交
    windows = [fitz.Rect(b.x0 - 3, b.y0 - 36, b.x1 + 10, b.y0 + 6) for b in bars] + EXTRA_WINDOWS
    mask = [p for p in text_paths if any(p.intersects(w) for w in windows)]

    hit = sum(1 for b in bars if any(p.intersects(fitz.Rect(b.x0 - 3, b.y0 - 36, b.x1 + 10, b.y0 + 6))
                                    for p in text_paths))
    overlap = sum(1 for p in mask for b in bars if p.intersects(b))
    print(f"柱子 {len(bars)} | 有标签 {hit} | 拟擦碎片 {len(mask)} | 压到柱体 {overlap}")
    if overlap:
        print("⚠️ 有碎片压到柱体，检查窗口是否过宽")

    # 4) 涂白
    pix = pg.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=CLIP)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    dr = ImageDraw.Draw(img)
    for p in mask:
        dr.rectangle([(p.x0 - CLIP.x0) * ZOOM - 2, (p.y0 - CLIP.y0) * ZOOM - 2,
                      (p.x1 - CLIP.x0) * ZOOM + 2, (p.y1 - CLIP.y0) * ZOOM + 2],
                     fill=(255, 255, 255))
    img.save(OUT)
    print("saved", OUT, img.size)
    d.close()


if __name__ == "__main__":
    main()
