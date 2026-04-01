"""
蒙德里安形式规则迁移：「圆的均衡」

将蒙德里安的笛卡尔正交网格（水平线+垂直线→矩形单元）完整映射到极坐标系
（同心圆弧+放射线→扇环区域），保留全部深层规则：

    规则一  正交网格  → 同心-放射网格
    规则二  动态均衡  → 圆心偏移 +3.8% / +2.2%
    规则三  以少驭多  → ~65% 留白, ~15% 有色
    规则四  黄金比例  → 半径按 φ 递增
    规则五  主题变奏  → 赤陶 ×3, 深青 ×2
    规则六  开放构图  → 外圈超出画布被裁切
    规则七  色彩克制  → 4 色新色板, 保持冷暖均衡

输出 250mm × 250mm 正方形作品 (300 dpi = 2953 × 2953 px)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import os

# ============================================================
# 常量
# ============================================================

PHI = (1 + np.sqrt(5)) / 2

CANVAS_PX = 2953
DPI = 300
FIG_INCHES = CANVAS_PX / DPI

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# ============================================================
# 色板（心理重量映射蒙德里安原色）
# ============================================================

PALETTE = {
    'bg':         '#FAFAF5',   # 暖白（替代白色）
    'teal':       '#1B4965',   # 深青（替代蓝色，冷·沉稳，需大面积）
    'terracotta': '#C44536',   # 赤陶（替代红色，暖·强烈，小面积即可）
    'amber':      '#F0A202',   # 琥珀金（替代黄色，暖·明亮）
    'sage':       '#7D8570',   # 灰绿（替代绿色，过渡色）
    'line':       '#2D3436',   # 炭灰（替代黑色）
}

# ============================================================
# 构图参数
# ============================================================

# 规则二：动态均衡 —— 圆心偏离画布几何中心
CX = CANVAS_PX * 0.535    # 右偏 ~3.5%
CY = CANVAS_PX * 0.520    # 上偏 ~2.0%（matplotlib y 轴向上）

# 规则四：黄金比例同心圆
R_BASE = 260
RADII = [int(R_BASE * PHI ** i) for i in range(5)]
# → [260, 420, 680, 1100, 1780]

# 规则一：非均匀放射线（保留蒙德里安的"紧密对 + 宽间距"韵律）
ANGLES = [15, 42, 128, 162, 270, 305]
# 间距: 27° 86° 34° 108° 35° 70°  — 两组紧密对 + 两组宽间距

# 线宽三级（类比蒙德里安的粗/中/细线条层级）
LW_THICK  = 18.0
LW_MEDIUM = 11.0
LW_THIN   = 7.0

CIRCLE_LW = [LW_THIN, LW_MEDIUM, LW_THICK, LW_THICK, LW_MEDIUM]
RAY_LW    = [LW_THIN, LW_THICK, LW_THICK, LW_MEDIUM, LW_THIN, LW_THICK]

# ============================================================
# 有色扇环分配
# (内半径索引, 外半径索引, 起始角索引, 结束角索引, 颜色键)
#
# Ring i  = RADII[i] 到 RADII[i+1] 之间的环带（Ring 0 = 中心到 r₁）
# Wedge j = ANGLES[j] 到 ANGLES[j+1] 之间的角域
# ============================================================

SECTORS = [
    # ——— 深青（大）: Ring 2, θ₁→θ₂ (42°→128°, 86°) ———
    # 上方主要冷色，面积 ~2.5%
    (2, 3, 1, 2, 'teal'),

    # ——— 赤陶（大）: Ring 1, θ₃→θ₄ (162°→270°, 108°) ———
    # 下方主要暖色（内环）
    (1, 2, 3, 4, 'terracotta'),

    # ——— 琥珀金: Ring 2, θ₅→θ₀ (305°→15°, 70°) ———
    # 右侧暖色
    (2, 3, 5, 0, 'amber'),

    # ——— 灰绿: Ring 1, θ₂→θ₃ (128°→162°, 34°) ———
    # 左侧小面积过渡色
    (1, 2, 2, 3, 'sage'),

    # ——— 赤陶（中）: Ring 3, θ₀→θ₁ (15°→42°, 27°) ———
    # 右上·外环，部分被画布边缘裁切（开放构图）
    (3, 4, 0, 1, 'terracotta'),

    # ——— 深青（小）: Ring 0, θ₃→θ₄ (162°→270°, 108°) ———
    # 中心区域冷色（主题变奏：大青 Ring2 vs 小青 Ring0）
    (0, 1, 3, 4, 'teal'),

    # ——— 赤陶（微）: Ring 0, θ₀→θ₁ (15°→42°, 27°) ———
    # 中心附近微小暖色点缀
    (0, 1, 0, 1, 'terracotta'),
]

# ============================================================
# 绘图函数
# ============================================================

def wedge_angles(a_start_idx, a_end_idx):
    """将角度索引转换为 matplotlib Wedge 的起止角度（处理跨 360° 的情况）。"""
    t1 = ANGLES[a_start_idx % len(ANGLES)]
    t2 = ANGLES[a_end_idx % len(ANGLES)]
    if t2 <= t1:
        t2 += 360
    return t1, t2


def draw_sectors(ax):
    """绘制所有有色扇环。"""
    for r_in, r_out, a_s, a_e, ckey in SECTORS:
        r_inner = RADII[r_in] if r_in > 0 else 0
        r_outer = RADII[r_out] if r_out < len(RADII) else CANVAS_PX
        t1, t2 = wedge_angles(a_s, a_e)

        kw = dict(facecolor=PALETTE[ckey], edgecolor='none', zorder=1,
                  antialiased=True)
        if r_inner == 0:
            patch = Wedge((CX, CY), r_outer, t1, t2, **kw)
        else:
            patch = Wedge((CX, CY), r_outer, t1, t2,
                          width=r_outer - r_inner, **kw)
        ax.add_patch(patch)


def draw_circles(ax):
    """绘制同心圆结构线。"""
    for r, lw in zip(RADII, CIRCLE_LW):
        c = plt.Circle((CX, CY), r, fill=False,
                        edgecolor=PALETTE['line'], linewidth=lw,
                        zorder=3, antialiased=True)
        ax.add_patch(c)


def draw_rays(ax):
    """绘制放射线结构线。"""
    extent = RADII[-1] * 1.5
    for deg, lw in zip(ANGLES, RAY_LW):
        rad = np.radians(deg)
        xe = CX + extent * np.cos(rad)
        ye = CY + extent * np.sin(rad)
        ax.plot([CX, xe], [CY, ye],
                color=PALETTE['line'], linewidth=lw,
                solid_capstyle='butt', zorder=2, antialiased=True)


def compute_stats():
    """估算有色面积占比（解析计算，不含裁切）。"""
    total_canvas = CANVAS_PX ** 2
    colored_area = 0
    for r_in, r_out, a_s, a_e, _ in SECTORS:
        ri = RADII[r_in] if r_in > 0 else 0
        ro = RADII[r_out] if r_out < len(RADII) else CANVAS_PX
        t1, t2 = wedge_angles(a_s, a_e)
        span = t2 - t1
        area = np.pi * (ro**2 - ri**2) * (span / 360)
        colored_area += area
    return colored_area / total_canvas * 100


def measure_visible_colored(path):
    """读取输出图像，统计实际可见有色像素占比。"""
    from PIL import Image
    img = np.array(Image.open(path).convert('RGB'))
    h, w, _ = img.shape
    bg_rgb = tuple(int(PALETTE['bg'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    line_rgb = tuple(int(PALETTE['line'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    total = h * w
    bg_mask = (np.abs(img[:,:,0].astype(int) - bg_rgb[0]) < 12) & \
              (np.abs(img[:,:,1].astype(int) - bg_rgb[1]) < 12) & \
              (np.abs(img[:,:,2].astype(int) - bg_rgb[2]) < 12)
    line_mask = (np.abs(img[:,:,0].astype(int) - line_rgb[0]) < 20) & \
                (np.abs(img[:,:,1].astype(int) - line_rgb[1]) < 20) & \
                (np.abs(img[:,:,2].astype(int) - line_rgb[2]) < 20)
    bg_pct = bg_mask.sum() / total * 100
    line_pct = line_mask.sum() / total * 100
    colored_pct = 100 - bg_pct - line_pct
    return bg_pct, line_pct, colored_pct


# ============================================================
# 主函数
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(FIG_INCHES, FIG_INCHES), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_xlim(0, CANVAS_PX)
    ax.set_ylim(0, CANVAS_PX)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    draw_sectors(ax)
    draw_circles(ax)
    draw_rays(ax)

    path = os.path.join(OUTPUT_DIR, 'design_v1.png')
    fig.savefig(path, dpi=DPI,
                facecolor=fig.get_facecolor(), edgecolor='none',
                pad_inches=0)
    plt.close(fig)

    from PIL import Image, PngImagePlugin
    img = Image.open(path).convert('RGB')
    ppi = 300
    ppm = int(ppi / 0.0254)  # 11811 px/m
    info = PngImagePlugin.PngInfo()
    info.add_text('Software', 'generate_design.py')
    img.save(path, pnginfo=info, dpi=(ppi, ppi))

    pct = compute_stats()
    bg_pct, line_pct, col_pct = measure_visible_colored(path)
    print('=' * 50)
    print('蒙德里安形式规则迁移：「圆的均衡」')
    print('=' * 50)
    print(f'画布 : {CANVAS_PX}×{CANVAS_PX} px  '
          f'({CANVAS_PX / DPI * 25.4:.0f}×{CANVAS_PX / DPI * 25.4:.0f} mm)')
    print(f'圆心 : ({CX:.0f}, {CY:.0f})  '
          f'偏移 +{(CX / CANVAS_PX - 0.5) * 100:.1f}% / '
          f'+{(CY / CANVAS_PX - 0.5) * 100:.1f}%')
    print(f'半径 : {RADII}  (φ = {PHI:.4f})')
    print(f'角度 : {ANGLES}°')
    print(f'面积（解析, 含画外）: {pct:.1f}%')
    print(f'实际可见 — 背景: {bg_pct:.1f}%  结构线: {line_pct:.1f}%  有色: {col_pct:.1f}%')
    print(f'（蒙德里安原作: 背景 65.3%  线条 19.3%  有色 15.4%）')
    print(f'已保存 → {path}')


if __name__ == '__main__':
    main()
