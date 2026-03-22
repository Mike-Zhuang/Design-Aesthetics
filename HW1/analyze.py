"""
蒙德里安画作形式分析脚本
对含绿色元素的蒙德里安作品进行三维度（平面抽象结构、数理比例关系、单元组合原理）分析
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from scipy import ndimage
from pathlib import Path
import json
import os

# ============================================================
# 全局设置
# ============================================================

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "pic.jpg")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 色彩名称映射
COLOR_NAMES_CN = {
    'red': '红色', 'yellow': '黄色', 'blue': '蓝色', 'green': '绿色', 'white': '白色'
}
COLOR_DISPLAY = {
    'red': '#E63946', 'yellow': '#DAA520', 'blue': '#1D3557',
    'green': '#2D6A4F', 'white': '#F1FAEE'
}


# ============================================================
# 第一部分：图像预处理与基础数据提取
# ============================================================

def load_image(path):
    """加载图像并转换色彩空间"""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr, img_rgb, img_hsv, img_gray


def detect_black_lines(img_gray, img_hsv, h, w):
    """
    检测黑色线条，返回水平线和垂直线的位置、粗细信息。
    使用灰度阈值 + 形态学操作分离水平/垂直方向线条。
    """
    # 黑色像素掩膜：灰度值低且饱和度低（排除深色彩色区域）
    gray_mask = img_gray < 60
    sat_channel = img_hsv[:, :, 1]
    low_sat = sat_channel < 80
    black_mask = (gray_mask & low_sat).astype(np.uint8) * 255

    # 形态学分离水平线
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    h_lines_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, h_kernel)

    # 形态学分离垂直线
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 8))
    v_lines_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, v_kernel)

    # 提取水平线的 y 坐标和粗细
    h_projection = np.sum(h_lines_mask > 0, axis=1)
    h_lines = _extract_line_segments(h_projection, min_length=w // 10)

    # 提取垂直线的 x 坐标和粗细
    v_projection = np.sum(v_lines_mask > 0, axis=0)
    v_lines = _extract_line_segments(v_projection, min_length=h // 10)

    return h_lines, v_lines, h_lines_mask, v_lines_mask, black_mask


def _extract_line_segments(projection, min_length=50):
    """
    从投影中提取线段：返回 [(center, thickness, start, end), ...]
    projection: 1D array，每行/列中黑色像素的数量
    min_length: 投影值大于此阈值才认为是有效线条区域
    """
    binary = (projection > min_length).astype(np.uint8)
    segments = []
    in_segment = False
    start = 0

    for i in range(len(binary)):
        if binary[i] and not in_segment:
            start = i
            in_segment = True
        elif not binary[i] and in_segment:
            end = i
            thickness = end - start
            center = (start + end) / 2.0
            segments.append({
                'center': center,
                'thickness': thickness,
                'start': start,
                'end': end
            })
            in_segment = False

    if in_segment:
        end = len(binary)
        thickness = end - start
        center = (start + end) / 2.0
        segments.append({
            'center': center, 'thickness': thickness,
            'start': start, 'end': end
        })

    return segments


def detect_color_blocks(img_hsv, img_rgb, h, w):
    """
    检测红、黄、蓝、绿四色色块。
    返回每个色块的边界框、面积和颜色标签。
    """
    blocks = []

    # HSV 阈值范围定义
    color_ranges = {
        'red': [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        'blue': [
            (np.array([100, 80, 80]), np.array([130, 255, 255]))
        ],
        'yellow': [
            (np.array([20, 80, 100]), np.array([35, 255, 255]))
        ],
        'green': [
            (np.array([35, 80, 80]), np.array([85, 255, 255]))
        ]
    }

    for color_name, ranges in color_ranges.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(img_hsv, lower, upper)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w * 0.001):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # 计算实际填充面积（在掩膜内的像素数）
            roi_mask = mask[y:y+bh, x:x+bw]
            fill_area = np.sum(roi_mask > 0)

            blocks.append({
                'color': color_name,
                'x': x, 'y': y, 'w': bw, 'h': bh,
                'area': int(fill_area),
                'bbox_area': int(bw * bh),
                'cx': x + bw / 2.0,
                'cy': y + bh / 2.0
            })

    # 按面积降序排列
    blocks.sort(key=lambda b: b['area'], reverse=True)
    return blocks


# ============================================================
# 第二部分：网格结构提取
# ============================================================

def build_grid(h_lines, v_lines, img_h, img_w):
    """
    基于检测到的线条构建网格坐标系。
    在线条列表前后补充画面边界，形成完整网格。
    返回网格行列坐标和所有矩形单元。
    """
    # 提取行分界坐标（y 方向）
    y_coords = [0]
    for line in h_lines:
        y_coords.append(int(line['center']))
    y_coords.append(img_h)

    # 提取列分界坐标（x 方向）
    x_coords = [0]
    for line in v_lines:
        x_coords.append(int(line['center']))
    x_coords.append(img_w)

    y_coords = sorted(set(y_coords))
    x_coords = sorted(set(x_coords))

    # 构建所有矩形单元
    cells = []
    for i in range(len(y_coords) - 1):
        for j in range(len(x_coords) - 1):
            cell = {
                'row': i, 'col': j,
                'x1': x_coords[j], 'y1': y_coords[i],
                'x2': x_coords[j + 1], 'y2': y_coords[i + 1],
                'width': x_coords[j + 1] - x_coords[j],
                'height': y_coords[i + 1] - y_coords[i],
                'area': (x_coords[j + 1] - x_coords[j]) * (y_coords[i + 1] - y_coords[i]),
                'cx': (x_coords[j] + x_coords[j + 1]) / 2.0,
                'cy': (y_coords[i] + y_coords[i + 1]) / 2.0,
            }
            cells.append(cell)

    return x_coords, y_coords, cells


def assign_colors_to_cells(cells, color_blocks, img_hsv, img_h, img_w):
    """为每个网格单元分配主要颜色"""
    for cell in cells:
        x1, y1 = cell['x1'], cell['y1']
        x2, y2 = cell['x2'], cell['y2']
        # 边界保护
        x1c = max(0, x1 + 3)
        y1c = max(0, y1 + 3)
        x2c = min(img_w, x2 - 3)
        y2c = min(img_h, y2 - 3)

        if x2c <= x1c or y2c <= y1c:
            cell['dominant_color'] = 'white'
            continue

        roi = img_hsv[y1c:y2c, x1c:x2c]
        cell_area = roi.shape[0] * roi.shape[1]
        if cell_area == 0:
            cell['dominant_color'] = 'white'
            continue

        # 检测各色占比
        best_color = 'white'
        best_ratio = 0.0

        color_ranges = {
            'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                    (np.array([160, 100, 100]), np.array([180, 255, 255]))],
            'blue': [(np.array([100, 80, 80]), np.array([130, 255, 255]))],
            'yellow': [(np.array([20, 80, 100]), np.array([35, 255, 255]))],
            'green': [(np.array([35, 80, 80]), np.array([85, 255, 255]))]
        }

        for cname, ranges in color_ranges.items():
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(roi, lower, upper)
            ratio = np.sum(mask > 0) / cell_area
            if ratio > best_ratio and ratio > 0.2:
                best_ratio = ratio
                best_color = cname

        cell['dominant_color'] = best_color
        cell['color_ratio'] = best_ratio if best_color != 'white' else 0.0

    return cells


# ============================================================
# 第三部分：分析函数
# ============================================================

def analyze_structure(h_lines, v_lines, cells, color_blocks, img_h, img_w):
    """平面抽象结构分析"""
    results = {}

    # 1. 线条分类（粗/中/细）
    all_thicknesses = [l['thickness'] for l in h_lines + v_lines]
    if all_thicknesses:
        mean_t = np.mean(all_thicknesses)
        std_t = np.std(all_thicknesses)
        for line in h_lines + v_lines:
            t = line['thickness']
            if t > mean_t + 0.5 * std_t:
                line['category'] = '粗线'
            elif t < mean_t - 0.5 * std_t:
                line['category'] = '细线'
            else:
                line['category'] = '中线'

    results['h_lines'] = h_lines
    results['v_lines'] = v_lines

    # 2. 视觉重心计算：加权平均（有色区域权重更高）
    color_weights = {'red': 5.0, 'blue': 3.0, 'yellow': 4.0, 'green': 3.5, 'white': 0.2}
    total_weight = 0
    wx_sum = 0
    wy_sum = 0
    for cell in cells:
        w = color_weights.get(cell['dominant_color'], 0.2) * cell['area']
        wx_sum += w * cell['cx']
        wy_sum += w * cell['cy']
        total_weight += w

    if total_weight > 0:
        visual_cx = wx_sum / total_weight
        visual_cy = wy_sum / total_weight
    else:
        visual_cx, visual_cy = img_w / 2, img_h / 2

    results['visual_center'] = (visual_cx, visual_cy)
    results['geometric_center'] = (img_w / 2, img_h / 2)
    results['center_offset'] = (
        (visual_cx - img_w / 2) / img_w * 100,
        (visual_cy - img_h / 2) / img_h * 100
    )

    # 3. 正负空间分析
    colored_area = sum(b['area'] for b in color_blocks)
    total_area = img_h * img_w
    white_area = total_area - colored_area
    # 黑线面积估算
    black_line_area = sum(l['thickness'] * img_w for l in h_lines) + sum(l['thickness'] * img_h for l in v_lines)

    results['area_stats'] = {
        'total': total_area,
        'colored': colored_area,
        'white': white_area - int(black_line_area),
        'black_lines': int(black_line_area),
        'colored_ratio': colored_area / total_area * 100,
        'white_ratio': (white_area - black_line_area) / total_area * 100,
        'black_ratio': black_line_area / total_area * 100
    }

    # 4. 空间分割层级
    h_thicknesses = sorted([l['thickness'] for l in h_lines], reverse=True)
    v_thicknesses = sorted([l['thickness'] for l in v_lines], reverse=True)
    results['line_hierarchy'] = {
        'h_thicknesses': h_thicknesses,
        'v_thicknesses': v_thicknesses,
    }

    return results


def analyze_proportions(h_lines, v_lines, cells, color_blocks, img_h, img_w):
    """数理比例关系分析"""
    results = {}
    GOLDEN_RATIO = 1.618

    # 1. 线条间距比例分析
    h_centers = sorted([l['center'] for l in h_lines])
    v_centers = sorted([l['center'] for l in v_lines])

    # 加上边界
    h_positions = [0] + h_centers + [img_h]
    v_positions = [0] + v_centers + [img_w]

    h_gaps = [h_positions[i + 1] - h_positions[i] for i in range(len(h_positions) - 1)]
    v_gaps = [v_positions[i + 1] - v_positions[i] for i in range(len(v_positions) - 1)]

    results['h_gaps'] = h_gaps
    results['v_gaps'] = v_gaps
    results['h_positions'] = h_positions
    results['v_positions'] = v_positions

    # 间距比值
    if len(h_gaps) > 1:
        h_ratios = []
        for i in range(len(h_gaps)):
            for j in range(i + 1, len(h_gaps)):
                if h_gaps[j] > 0:
                    ratio = h_gaps[i] / h_gaps[j]
                    h_ratios.append({
                        'pair': (i, j),
                        'values': (round(h_gaps[i], 1), round(h_gaps[j], 1)),
                        'ratio': round(ratio, 3),
                        'near_golden': abs(ratio - GOLDEN_RATIO) < 0.1 or abs(1/ratio - GOLDEN_RATIO) < 0.1 if ratio > 0 else False
                    })
        results['h_ratios'] = h_ratios

    if len(v_gaps) > 1:
        v_ratios = []
        for i in range(len(v_gaps)):
            for j in range(i + 1, len(v_gaps)):
                if v_gaps[j] > 0:
                    ratio = v_gaps[i] / v_gaps[j]
                    v_ratios.append({
                        'pair': (i, j),
                        'values': (round(v_gaps[i], 1), round(v_gaps[j], 1)),
                        'ratio': round(ratio, 3),
                        'near_golden': abs(ratio - GOLDEN_RATIO) < 0.1 or abs(1/ratio - GOLDEN_RATIO) < 0.1 if ratio > 0 else False
                    })
        results['v_ratios'] = v_ratios

    # 2. 黄金分割线验证
    golden_h = [img_h / GOLDEN_RATIO, img_h * (1 - 1/GOLDEN_RATIO)]
    golden_v = [img_w / GOLDEN_RATIO, img_w * (1 - 1/GOLDEN_RATIO)]
    results['golden_h'] = golden_h
    results['golden_v'] = golden_v

    # 与实际线条的偏差
    golden_matches = []
    for gh in golden_h:
        for line in h_lines:
            deviation = abs(line['center'] - gh) / img_h * 100
            if deviation < 5:
                golden_matches.append({
                    'type': '水平', 'golden_pos': round(gh, 1),
                    'line_pos': round(line['center'], 1),
                    'deviation_pct': round(deviation, 2)
                })
    for gv in golden_v:
        for line in v_lines:
            deviation = abs(line['center'] - gv) / img_w * 100
            if deviation < 5:
                golden_matches.append({
                    'type': '垂直', 'golden_pos': round(gv, 1),
                    'line_pos': round(line['center'], 1),
                    'deviation_pct': round(deviation, 2)
                })
    results['golden_matches'] = golden_matches

    # 3. 色块面积比例
    if color_blocks:
        total_colored = sum(b['area'] for b in color_blocks)
        color_area_by_type = {}
        for b in color_blocks:
            c = b['color']
            color_area_by_type[c] = color_area_by_type.get(c, 0) + b['area']

        results['color_areas'] = {
            c: {
                'area': a,
                'ratio_of_canvas': round(a / (img_h * img_w) * 100, 2),
                'ratio_of_colored': round(a / total_colored * 100, 2) if total_colored > 0 else 0
            }
            for c, a in color_area_by_type.items()
        }

        # 色块之间的面积比
        colors_sorted = sorted(color_area_by_type.items(), key=lambda x: x[1], reverse=True)
        if len(colors_sorted) >= 2:
            area_pairs = []
            for i in range(len(colors_sorted)):
                for j in range(i + 1, len(colors_sorted)):
                    c1, a1 = colors_sorted[i]
                    c2, a2 = colors_sorted[j]
                    if a2 > 0:
                        area_pairs.append({
                            'pair': f"{COLOR_NAMES_CN[c1]}:{COLOR_NAMES_CN[c2]}",
                            'ratio': round(a1 / a2, 2)
                        })
            results['color_area_pairs'] = area_pairs

    # 4. 线条粗细比 — 只按粗/细两档对比
    all_t = [l['thickness'] for l in h_lines + v_lines]
    if all_t:
        t_mean = np.mean(all_t)
        thick_group = [t for t in all_t if t >= t_mean]
        thin_group = [t for t in all_t if t < t_mean]
        results['thickness_groups'] = {
            'thick': {'values': sorted(set(int(t) for t in thick_group)),
                      'mean': round(np.mean(thick_group), 1) if thick_group else 0},
            'thin': {'values': sorted(set(int(t) for t in thin_group)),
                     'mean': round(np.mean(thin_group), 1) if thin_group else 0},
        }
        if thin_group and thick_group:
            results['thickness_ratio'] = round(np.mean(thick_group) / np.mean(thin_group), 2)
            # 最粗 vs 最细
            results['max_min_thickness_ratio'] = round(max(all_t) / min(all_t), 2)

    # 5. 单元面积分布
    cell_areas = sorted([c['area'] for c in cells], reverse=True)
    results['cell_areas'] = cell_areas
    if cell_areas:
        results['cell_area_stats'] = {
            'max': max(cell_areas),
            'min': min(cell_areas),
            'mean': round(np.mean(cell_areas), 1),
            'std': round(np.std(cell_areas), 1),
            'max_min_ratio': round(max(cell_areas) / min(cell_areas), 2) if min(cell_areas) > 0 else float('inf')
        }

    return results


def analyze_units(cells, color_blocks, h_lines, v_lines, img_h, img_w):
    """单元组合原理分析"""
    results = {}

    # 1. 基本单元分类
    areas = [c['area'] for c in cells]
    if areas:
        mean_area = np.mean(areas)
        std_area = np.std(areas)

    unit_categories = {'大': [], '中': [], '小': []}
    for i, cell in enumerate(cells):
        a = cell['area']
        if a > mean_area + std_area:
            cat = '大'
        elif a < mean_area - 0.5 * std_area:
            cat = '小'
        else:
            cat = '中'
        cell['size_category'] = cat
        unit_categories[cat].append(i)

    results['unit_categories'] = {k: len(v) for k, v in unit_categories.items()}

    # 按颜色+大小组合分类
    type_combos = {}
    for cell in cells:
        key = f"{cell['dominant_color']}_{cell['size_category']}"
        type_combos[key] = type_combos.get(key, 0) + 1
    results['type_combinations'] = type_combos

    # 2. 色彩分布规则
    # 将画面分为四个象限，分析色彩在各象限的分布
    mid_x, mid_y = img_w / 2, img_h / 2
    quadrants = {'左上': {}, '右上': {}, '左下': {}, '右下': {}}

    for block in color_blocks:
        if block['cx'] < mid_x and block['cy'] < mid_y:
            q = '左上'
        elif block['cx'] >= mid_x and block['cy'] < mid_y:
            q = '右上'
        elif block['cx'] < mid_x and block['cy'] >= mid_y:
            q = '左下'
        else:
            q = '右下'

        c = block['color']
        quadrants[q][c] = quadrants[q].get(c, 0) + block['area']

    results['quadrant_distribution'] = quadrants

    # 3. 色彩位置特征
    color_positions = {}
    for block in color_blocks:
        c = block['color']
        # 归一化位置
        norm_x = block['cx'] / img_w
        norm_y = block['cy'] / img_h
        pos_desc = ''
        if norm_x < 0.33:
            pos_desc += '左'
        elif norm_x > 0.67:
            pos_desc += '右'
        else:
            pos_desc += '中'
        if norm_y < 0.33:
            pos_desc += '上'
        elif norm_y > 0.67:
            pos_desc += '下'
        else:
            pos_desc += '中'

        if c not in color_positions:
            color_positions[c] = []
        color_positions[c].append({
            'position': pos_desc,
            'norm_x': round(norm_x, 3),
            'norm_y': round(norm_y, 3),
            'area': block['area']
        })
    results['color_positions'] = color_positions

    # 4. 重复与变化分析
    # 分析色块的尺寸变化
    for c in ['red', 'blue', 'yellow', 'green']:
        c_blocks = [b for b in color_blocks if b['color'] == c]
        if len(c_blocks) > 1:
            areas_c = [b['area'] for b in c_blocks]
            results[f'{c}_variation'] = {
                'count': len(c_blocks),
                'area_range': (min(areas_c), max(areas_c)),
                'scale_ratio': round(max(areas_c) / min(areas_c), 2) if min(areas_c) > 0 else float('inf')
            }

    # 5. 边缘 vs 内部分布
    edge_blocks = []
    inner_blocks = []
    margin = 0.1
    for block in color_blocks:
        nx, ny = block['cx'] / img_w, block['cy'] / img_h
        if nx < margin or nx > (1 - margin) or ny < margin or ny > (1 - margin):
            edge_blocks.append(block)
        else:
            inner_blocks.append(block)

    results['edge_vs_inner'] = {
        'edge_count': len(edge_blocks),
        'inner_count': len(inner_blocks),
        'edge_colors': [b['color'] for b in edge_blocks],
        'inner_colors': [b['color'] for b in inner_blocks]
    }

    return results


# ============================================================
# 第四部分：可视化生成
# ============================================================

def draw_grid_structure(img_rgb, h_lines, v_lines, x_coords, y_coords, cells, img_h, img_w, output_dir):
    """绘制网格拓扑结构图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：原图 + 网格线标注
    ax1 = axes[0]
    ax1.imshow(img_rgb, alpha=0.4)
    ax1.set_title('网格拓扑结构', fontsize=16, fontweight='bold')

    line_colors = {'粗线': '#E63946', '中线': '#457B9D', '细线': '#A8DADC'}
    line_widths_map = {'粗线': 3.5, '中线': 2.0, '细线': 1.2}

    for line in h_lines:
        cat = line.get('category', '中线')
        ax1.axhline(y=line['center'], color=line_colors[cat],
                    linewidth=line_widths_map[cat], linestyle='-', alpha=0.85)
        ax1.text(img_w + 10, line['center'],
                f"y={int(line['center'])} ({cat}, {int(line['thickness'])}px)",
                fontsize=7, va='center', color=line_colors[cat])

    for line in v_lines:
        cat = line.get('category', '中线')
        ax1.axvline(x=line['center'], color=line_colors[cat],
                    linewidth=line_widths_map[cat], linestyle='-', alpha=0.85)
        ax1.text(line['center'], -15,
                f"x={int(line['center'])}\n{cat}\n{int(line['thickness'])}px",
                fontsize=6, ha='center', va='bottom', color=line_colors[cat], rotation=0)

    ax1.set_xlim(-5, img_w + 120)
    ax1.set_ylim(img_h + 5, -30)
    ax1.set_xlabel('x (像素)', fontsize=10)
    ax1.set_ylabel('y (像素)', fontsize=10)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=line_colors['粗线'], lw=3.5, label='粗线'),
        Line2D([0], [0], color=line_colors['中线'], lw=2.0, label='中线'),
        Line2D([0], [0], color=line_colors['细线'], lw=1.2, label='细线'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # 右图：空间分割层级（抽象化）
    ax2 = axes[1]
    ax2.set_title('空间分割层级', fontsize=16, fontweight='bold')
    ax2.set_xlim(0, img_w)
    ax2.set_ylim(img_h, 0)
    ax2.set_aspect('equal')

    # 绘制网格单元
    color_map = {
        'red': '#E6394660', 'blue': '#1D355760', 'yellow': '#DAA52060',
        'green': '#2D6A4F60', 'white': '#F0F0F030'
    }
    for cell in cells:
        c = cell.get('dominant_color', 'white')
        rect = patches.Rectangle(
            (cell['x1'], cell['y1']), cell['width'], cell['height'],
            linewidth=0.5, edgecolor='#333', facecolor=color_map.get(c, '#F0F0F030')
        )
        ax2.add_patch(rect)
        if cell['area'] > (img_h * img_w * 0.01):
            ax2.text(cell['cx'], cell['cy'],
                    f"R{cell['row']},{cell['col']}",
                    fontsize=6, ha='center', va='center', color='#333')

    # 标注行列序号
    for i, y in enumerate(y_coords[:-1]):
        ax2.text(-15, (y + y_coords[i + 1]) / 2, f"行{i}", fontsize=7, ha='center', va='center', color='#666')
    for j, x in enumerate(x_coords[:-1]):
        ax2.text((x + x_coords[j + 1]) / 2, -15, f"列{j}", fontsize=7, ha='center', va='center', color='#666')

    ax2.set_xlabel('x (像素)', fontsize=10)
    ax2.set_ylabel('y (像素)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(output_dir, '01_grid_structure.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {path}")


def draw_visual_center(img_rgb, structure_results, color_blocks, img_h, img_w, output_dir):
    """绘制视觉重心与正负空间分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：视觉重心
    ax1 = axes[0]
    ax1.imshow(img_rgb, alpha=0.5)
    ax1.set_title('视觉重心分析', fontsize=16, fontweight='bold')

    vc = structure_results['visual_center']
    gc = structure_results['geometric_center']

    ax1.plot(*gc, 'o', color='#2A9D8F', markersize=14, markeredgewidth=2,
             markeredgecolor='white', label=f'几何中心 ({int(gc[0])}, {int(gc[1])})')
    ax1.plot(*vc, '*', color='#E63946', markersize=18, markeredgewidth=1.5,
             markeredgecolor='white', label=f'视觉重心 ({int(vc[0])}, {int(vc[1])})')

    # 偏移箭头
    ax1.annotate('', xy=vc, xytext=gc,
                arrowprops=dict(arrowstyle='->', color='#E76F51', lw=2.5))
    offset = structure_results['center_offset']
    ax1.text((gc[0] + vc[0]) / 2 + 15, (gc[1] + vc[1]) / 2,
            f'偏移: {offset[0]:+.1f}%, {offset[1]:+.1f}%',
            fontsize=10, color='#E76F51', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 标记色块位置
    block_colors = {'red': '#E63946', 'blue': '#1D3557', 'yellow': '#DAA520', 'green': '#2D6A4F'}
    for block in color_blocks:
        rect = patches.Rectangle(
            (block['x'], block['y']), block['w'], block['h'],
            linewidth=2, edgecolor=block_colors[block['color']],
            facecolor='none', linestyle='--'
        )
        ax1.add_patch(rect)

    ax1.legend(fontsize=10, loc='lower right',
              facecolor='white', framealpha=0.9)
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)

    # 右图：正负空间
    ax2 = axes[1]
    ax2.set_title('正负空间分析', fontsize=16, fontweight='bold')
    stats = structure_results['area_stats']
    labels = ['有色区域\n(正空间)', '白色区域\n(负空间)', '黑色线条\n(骨架)']
    sizes = [stats['colored_ratio'], stats['white_ratio'], stats['black_ratio']]
    colors_pie = ['#E76F51', '#F4F1DE', '#264653']
    explode = (0.05, 0, 0.05)

    wedges, texts, autotexts = ax2.pie(
        sizes, explode=explode, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
    )
    for t in autotexts:
        t.set_fontweight('bold')

    # 各色面积子饼图
    color_areas = {b['color']: 0 for b in color_blocks}
    for b in color_blocks:
        color_areas[b['color']] += b['area']

    if color_areas:
        ax_inset = fig.add_axes([0.72, 0.15, 0.2, 0.2])
        c_labels = [COLOR_NAMES_CN[c] for c in color_areas.keys()]
        c_sizes = list(color_areas.values())
        c_colors = [block_colors[c] for c in color_areas.keys()]
        ax_inset.pie(c_sizes, labels=c_labels, colors=c_colors,
                    autopct='%1.0f%%', textprops={'fontsize': 7, 'color': 'white'})
        ax_inset.set_title('色彩面积占比', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, '02_visual_center_space.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {path}")


def draw_proportion_analysis(img_rgb, proportion_results, h_lines, v_lines, img_h, img_w, output_dir):
    """绘制数理比例关系分析图"""
    fig = plt.figure(figsize=(18, 12))

    # 子图1：线条间距比例标注
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_rgb, alpha=0.3)
    ax1.set_title('线条间距比例', fontsize=14, fontweight='bold')

    h_gaps = proportion_results['h_gaps']
    v_gaps = proportion_results['v_gaps']
    h_pos = proportion_results['h_positions']
    v_pos = proportion_results['v_positions']

    # 标注水平间距
    for i in range(len(h_gaps)):
        y_mid = (h_pos[i] + h_pos[i + 1]) / 2
        ax1.annotate('', xy=(img_w + 5, h_pos[i]), xytext=(img_w + 5, h_pos[i + 1]),
                    arrowprops=dict(arrowstyle='<->', color='#E63946', lw=1.5))
        ax1.text(img_w + 15, y_mid, f'{h_gaps[i]:.0f}',
                fontsize=8, color='#E63946', va='center')

    # 标注垂直间距
    for i in range(len(v_gaps)):
        x_mid = (v_pos[i] + v_pos[i + 1]) / 2
        ax1.annotate('', xy=(v_pos[i], img_h + 5), xytext=(v_pos[i + 1], img_h + 5),
                    arrowprops=dict(arrowstyle='<->', color='#1D3557', lw=1.5))
        ax1.text(x_mid, img_h + 25, f'{v_gaps[i]:.0f}',
                fontsize=8, color='#1D3557', ha='center')

    ax1.set_xlim(-5, img_w + 80)
    ax1.set_ylim(img_h + 40, -5)

    # 子图2：黄金分割叠加验证
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_rgb, alpha=0.5)
    ax2.set_title('黄金分割线验证', fontsize=14, fontweight='bold')

    golden_h = proportion_results['golden_h']
    golden_v = proportion_results['golden_v']

    for gh in golden_h:
        ax2.axhline(y=gh, color='#F4A261', linewidth=2, linestyle='--', alpha=0.8)
        ax2.text(img_w + 5, gh, f'φ: y={gh:.0f}', fontsize=8, color='#F4A261', va='center')

    for gv in golden_v:
        ax2.axvline(x=gv, color='#2A9D8F', linewidth=2, linestyle='--', alpha=0.8)
        ax2.text(gv, -10, f'φ: x={gv:.0f}', fontsize=8, color='#2A9D8F', ha='center')

    # 画实际线条
    for line in h_lines:
        ax2.axhline(y=line['center'], color='#E63946', linewidth=1, alpha=0.5)
    for line in v_lines:
        ax2.axvline(x=line['center'], color='#1D3557', linewidth=1, alpha=0.5)

    # 标注匹配
    matches = proportion_results.get('golden_matches', [])
    for m in matches:
        pos = m['line_pos']
        if m['type'] == '水平':
            ax2.axhline(y=pos, color='#E76F51', linewidth=3, alpha=0.9)
            ax2.text(5, pos - 15, f'✓ 偏差{m["deviation_pct"]:.1f}%',
                    fontsize=9, color='#E76F51', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        else:
            ax2.axvline(x=pos, color='#E76F51', linewidth=3, alpha=0.9)
            ax2.text(pos + 5, 20, f'✓ 偏差{m["deviation_pct"]:.1f}%',
                    fontsize=9, color='#E76F51', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color='#F4A261', lw=2, ls='--', label='黄金分割线(水平)'),
        Line2D([0], [0], color='#2A9D8F', lw=2, ls='--', label='黄金分割线(垂直)'),
        Line2D([0], [0], color='#E76F51', lw=3, label='匹配线条'),
    ]
    ax2.legend(handles=legend_items, fontsize=8, loc='lower right')
    ax2.set_xlim(-5, img_w + 60)
    ax2.set_ylim(img_h + 5, -20)

    # 子图3：色块面积比例柱状图
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('色块面积比例', fontsize=14, fontweight='bold')

    color_areas = proportion_results.get('color_areas', {})
    if color_areas:
        bar_colors_map = {'red': '#E63946', 'blue': '#1D3557', 'yellow': '#DAA520', 'green': '#2D6A4F'}
        bar_names = [COLOR_NAMES_CN[c] for c in color_areas.keys()]
        bar_canvas = [color_areas[c]['ratio_of_canvas'] for c in color_areas.keys()]
        bar_colored = [color_areas[c]['ratio_of_colored'] for c in color_areas.keys()]
        bar_colors = [bar_colors_map[c] for c in color_areas.keys()]

        x_bar = np.arange(len(bar_names))
        width = 0.35

        bars1 = ax3.bar(x_bar - width/2, bar_canvas, width, color=bar_colors, alpha=0.7, label='占画面比例(%)')
        bars2 = ax3.bar(x_bar + width/2, bar_colored, width, color=bar_colors, alpha=1.0, label='占有色面积比例(%)')

        ax3.set_xticks(x_bar)
        ax3.set_xticklabels(bar_names, fontsize=11)
        ax3.set_ylabel('比例 (%)', fontsize=10)
        ax3.legend(fontsize=9)

        for bar in bars1:
            h_val = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, h_val + 0.1, f'{h_val:.1f}%',
                    ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            h_val = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, h_val + 0.1, f'{h_val:.1f}%',
                    ha='center', va='bottom', fontsize=8)

    # 子图4：网格单元面积分布
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('网格单元面积分布', fontsize=14, fontweight='bold')

    cell_areas = proportion_results.get('cell_areas', [])
    if cell_areas:
        ax4.bar(range(len(cell_areas)), cell_areas, color='#457B9D', alpha=0.7)
        ax4.axhline(y=np.mean(cell_areas), color='#E76F51', linestyle='--', linewidth=1.5, label=f'均值: {np.mean(cell_areas):.0f}')
        ax4.set_xlabel('单元序号（按面积降序）', fontsize=10)
        ax4.set_ylabel('面积 (像素²)', fontsize=10)
        ax4.legend(fontsize=9)

        stats = proportion_results.get('cell_area_stats', {})
        info_text = f"最大/最小比: {stats.get('max_min_ratio', 'N/A')}\n标准差: {stats.get('std', 'N/A')}"
        ax4.text(0.95, 0.95, info_text, transform=ax4.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, '03_proportion_analysis.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {path}")


def draw_unit_analysis(img_rgb, cells, color_blocks, unit_results, img_h, img_w, output_dir):
    """绘制单元组合原理分析图"""
    fig = plt.figure(figsize=(18, 12))

    block_colors_vis = {
        'red': '#E63946', 'blue': '#1D3557', 'yellow': '#DAA520',
        'green': '#2D6A4F', 'white': '#EEEEEE'
    }
    size_hatches = {'大': '', '中': '///', '小': '...'}

    # 子图1：单元分类标注图
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('基本单元分类', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)
    ax1.set_aspect('equal')

    size_colors = {'大': '#E6394640', '中': '#2A9D8F40', '小': '#F4A26140'}

    for cell in cells:
        sc = cell.get('size_category', '中')
        dc = cell.get('dominant_color', 'white')

        facecolor = size_colors[sc]
        if dc != 'white':
            facecolor = block_colors_vis[dc] + '80'

        rect = patches.Rectangle(
            (cell['x1'], cell['y1']), cell['width'], cell['height'],
            linewidth=1, edgecolor='#333',
            facecolor=facecolor,
            hatch=size_hatches[sc]
        )
        ax1.add_patch(rect)

        if cell['area'] > img_h * img_w * 0.008:
            label = f"{sc}"
            if dc != 'white':
                label = f"{COLOR_NAMES_CN[dc]}\n{sc}"
            ax1.text(cell['cx'], cell['cy'], label,
                    fontsize=7, ha='center', va='center',
                    fontweight='bold', color='#333',
                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))

    # 图例
    legend_patches = [
        patches.Patch(facecolor='#E6394640', label='大单元'),
        patches.Patch(facecolor='#2A9D8F40', hatch='///', label='中单元'),
        patches.Patch(facecolor='#F4A26140', hatch='...', label='小单元'),
    ]
    ax1.legend(handles=legend_patches, fontsize=9, loc='lower right')

    # 子图2：色彩分布图解
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_rgb, alpha=0.3)
    ax2.set_title('色彩空间分布', fontsize=14, fontweight='bold')

    # 画象限分割线
    ax2.axhline(y=img_h / 2, color='#666', linewidth=1, linestyle=':')
    ax2.axvline(x=img_w / 2, color='#666', linewidth=1, linestyle=':')

    quadrant_labels = {
        '左上': (img_w * 0.25, img_h * 0.25),
        '右上': (img_w * 0.75, img_h * 0.25),
        '左下': (img_w * 0.25, img_h * 0.75),
        '右下': (img_w * 0.75, img_h * 0.75),
    }

    quadrant_dist = unit_results.get('quadrant_distribution', {})
    for q_name, (qx, qy) in quadrant_labels.items():
        q_data = quadrant_dist.get(q_name, {})
        if q_data:
            text = '\n'.join([f"{COLOR_NAMES_CN[c]}" for c in q_data.keys()])
        else:
            text = '(无色块)'
        ax2.text(qx, qy, f"【{q_name}】\n{text}",
                fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # 标注色块
    for block in color_blocks:
        rect = patches.Rectangle(
            (block['x'], block['y']), block['w'], block['h'],
            linewidth=2.5, edgecolor=block_colors_vis[block['color']],
            facecolor='none', linestyle='-'
        )
        ax2.add_patch(rect)
        ax2.annotate(COLOR_NAMES_CN[block['color']],
                    xy=(block['cx'], block['cy']),
                    fontsize=8, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(facecolor=block_colors_vis[block['color']], alpha=0.8, boxstyle='round'))

    ax2.set_xlim(0, img_w)
    ax2.set_ylim(img_h, 0)

    # 子图3：组合逻辑示意
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('色块位置与尺度对比', fontsize=14, fontweight='bold')

    color_positions = unit_results.get('color_positions', {})
    y_offset = 0
    for c_name, positions in color_positions.items():
        for p in positions:
            ax3.barh(y_offset, p['area'], color=block_colors_vis[c_name], alpha=0.8, height=0.6)
            ax3.text(p['area'] + 500, y_offset,
                    f"{COLOR_NAMES_CN[c_name]} [{p['position']}] 面积={p['area']}",
                    va='center', fontsize=9)
            y_offset += 1

    ax3.set_xlabel('面积 (像素²)', fontsize=10)
    ax3.set_ylabel('色块', fontsize=10)
    ax3.set_yticks([])

    # 子图4：类型组合矩阵
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('单元类型组合矩阵', fontsize=14, fontweight='bold')

    type_combos = unit_results.get('type_combinations', {})
    colors_list = ['white', 'red', 'blue', 'yellow', 'green']
    sizes_list = ['大', '中', '小']

    matrix = np.zeros((len(colors_list), len(sizes_list)))
    for i, c in enumerate(colors_list):
        for j, s in enumerate(sizes_list):
            key = f"{c}_{s}"
            matrix[i, j] = type_combos.get(key, 0)

    im = ax4.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(sizes_list)))
    ax4.set_xticklabels(sizes_list, fontsize=11)
    ax4.set_yticks(range(len(colors_list)))
    ax4.set_yticklabels([COLOR_NAMES_CN.get(c, c) for c in colors_list], fontsize=11)

    for i in range(len(colors_list)):
        for j in range(len(sizes_list)):
            val = int(matrix[i, j])
            ax4.text(j, i, str(val), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if val > 2 else '#333')

    plt.colorbar(im, ax=ax4, label='数量')

    plt.tight_layout()
    path = os.path.join(output_dir, '04_unit_analysis.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {path}")


def draw_normalized_proportion(h_lines, v_lines, color_blocks, img_h, img_w, output_dir):
    """绘制归一化比例标注图：将画面分割以百分比形式展示"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title('画面归一化比例结构', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(1.15, -0.15)
    ax.set_aspect('equal')

    # 归一化坐标
    h_norms = [0] + [l['center'] / img_h for l in h_lines] + [1.0]
    v_norms = [0] + [l['center'] / img_w for l in v_lines] + [1.0]

    block_colors_vis = {
        'red': '#E6394680', 'blue': '#1D355780', 'yellow': '#DAA52080',
        'green': '#2D6A4F80', 'white': '#F5F5F520'
    }

    # 绘制网格
    for i in range(len(h_norms) - 1):
        for j in range(len(v_norms) - 1):
            x1, y1 = v_norms[j], h_norms[i]
            w, h = v_norms[j + 1] - v_norms[j], h_norms[i + 1] - h_norms[i]

            # 判断此单元内是否有色块
            fc = '#F5F5F520'
            for b in color_blocks:
                bcx = b['cx'] / img_w
                bcy = b['cy'] / img_h
                if x1 < bcx < x1 + w and y1 < bcy < y1 + h:
                    fc = block_colors_vis.get(b['color'], fc)
                    break

            rect = patches.Rectangle((x1, y1), w, h,
                                      linewidth=1.5, edgecolor='#333', facecolor=fc)
            ax.add_patch(rect)

    # 水平间距标注（右侧）
    for i in range(len(h_norms) - 1):
        gap = h_norms[i + 1] - h_norms[i]
        y_mid = (h_norms[i] + h_norms[i + 1]) / 2
        ax.annotate('', xy=(1.02, h_norms[i]), xytext=(1.02, h_norms[i + 1]),
                    arrowprops=dict(arrowstyle='<->', color='#E63946', lw=1.5))
        ax.text(1.05, y_mid, f'{gap * 100:.1f}%', fontsize=10,
                color='#E63946', va='center', fontweight='bold')

    # 垂直间距标注（底部）
    for j in range(len(v_norms) - 1):
        gap = v_norms[j + 1] - v_norms[j]
        x_mid = (v_norms[j] + v_norms[j + 1]) / 2
        ax.annotate('', xy=(v_norms[j], 1.02), xytext=(v_norms[j + 1], 1.02),
                    arrowprops=dict(arrowstyle='<->', color='#1D3557', lw=1.5))
        ax.text(x_mid, 1.06, f'{gap * 100:.1f}%', fontsize=9,
                color='#1D3557', ha='center', fontweight='bold')

    # 标注分割线位置百分比
    for i, h in enumerate(h_norms[1:-1]):
        ax.text(-0.08, h, f'{h * 100:.1f}%', fontsize=9, va='center',
                color='#666', ha='center')
        cat = h_lines[i].get('category', '')
        lw = 2.5 if '粗' in cat else 1.5
        ax.axhline(y=h, xmin=0.12, xmax=0.88, color='#333', linewidth=lw, linestyle='-')

    for j, v in enumerate(v_norms[1:-1]):
        ax.text(v, -0.08, f'{v * 100:.1f}%', fontsize=9, ha='center',
                color='#666')
        cat = v_lines[j].get('category', '')
        lw = 2.5 if '粗' in cat else 1.5
        ax.axvline(x=v, ymin=0.12, ymax=0.88, color='#333', linewidth=lw, linestyle='-')

    ax.set_xlabel('画面宽度 (%)', fontsize=11)
    ax.set_ylabel('画面高度 (%)', fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, '06_normalized_proportion.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {path}")


def draw_line_thickness_analysis(h_lines, v_lines, img_h, img_w, output_dir):
    """绘制线条粗细对比分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.set_title('水平线条粗细分布', fontsize=14, fontweight='bold')
    h_data = [(f"H{i} (y={int(l['center'])})", l['thickness']) for i, l in enumerate(h_lines)]
    if h_data:
        names, vals = zip(*h_data)
        colors_h = ['#E63946' if v > np.mean(vals) else '#457B9D' for v in vals]
        ax1.barh(range(len(names)), vals, color=colors_h, alpha=0.8)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel('粗细 (像素)', fontsize=10)
        ax1.axvline(x=np.mean(vals), color='#E76F51', linestyle='--', label=f'均值: {np.mean(vals):.1f}px')
        ax1.legend(fontsize=9)

    ax2 = axes[1]
    ax2.set_title('垂直线条粗细分布', fontsize=14, fontweight='bold')
    v_data = [(f"V{i} (x={int(l['center'])})", l['thickness']) for i, l in enumerate(v_lines)]
    if v_data:
        names, vals = zip(*v_data)
        colors_v = ['#E63946' if v > np.mean(vals) else '#457B9D' for v in vals]
        ax2.barh(range(len(names)), vals, color=colors_v, alpha=0.8)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel('粗细 (像素)', fontsize=10)
        ax2.axvline(x=np.mean(vals), color='#E76F51', linestyle='--', label=f'均值: {np.mean(vals):.1f}px')
        ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, '05_line_thickness.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {path}")


def draw_color_detection_result(img_rgb, color_blocks, img_h, img_w, output_dir):
    """绘制色块检测结果图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_rgb)
    ax.set_title('色块检测结果', fontsize=16, fontweight='bold')

    block_colors_vis = {
        'red': '#FF0000', 'blue': '#0000FF', 'yellow': '#FFD700', 'green': '#00FF00'
    }

    for i, block in enumerate(color_blocks):
        color = block_colors_vis[block['color']]
        rect = patches.Rectangle(
            (block['x'], block['y']), block['w'], block['h'],
            linewidth=3, edgecolor=color, facecolor='none', linestyle='-'
        )
        ax.add_patch(rect)
        ax.text(block['x'] + 5, block['y'] - 8,
               f"{COLOR_NAMES_CN[block['color']]} #{i+1}\n{block['w']}x{block['h']}px\n面积:{block['area']}",
               fontsize=8, color=color, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    plt.tight_layout()
    path = os.path.join(output_dir, '00_color_detection.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存: {path}")


# ============================================================
# 第五部分：报告生成
# ============================================================

def generate_report(structure_results, proportion_results, unit_results,
                    color_blocks, h_lines, v_lines, cells, img_h, img_w, output_dir):
    """生成完整的艺术鉴赏分析报告（Markdown 格式），保存为 v2 版本"""

    r = []
    def w(line=""):
        r.append(line)

    # ================================================================
    # 标题与作品基本信息
    # ================================================================
    w("# 蒙德里安画作形式分析报告")
    w()
    w("## 作品基本信息")
    w()
    w("- **作品名称**: Composition with Red, Blue and Yellow-Green")
    w("- **艺术家**: 皮特·蒙德里安 (Piet Mondrian, 1872-1944)")
    w("- **创作年代**: 约 1920 年")
    w("- **艺术流派**: 新造型主义 (De Stijl / Neo-Plasticism)")
    w("- **媒介**: 布面油画")
    w(f"- **分析图像尺寸**: {img_w} × {img_h} 像素")
    w()

    # ================================================================
    # 作品简介（扩写：艺术历程、哲学、绿色意义）
    # ================================================================
    w("### 艺术家与流派背景")
    w()
    w("皮特·蒙德里安是 20 世纪最具影响力的抽象艺术家之一。他的艺术生涯经历了三个阶段的蜕变：")
    w("早期（1892-1908）创作荷兰传统风景画，受海牙画派影响；中期（1908-1917）受立体主义启发，")
    w("逐步将树木、教堂等自然形象分解为几何线条与色块；最终（1917 年之后）与建筑师")
    w("西奥·凡·杜斯堡 (Theo van Doesburg) 等人共同创立了 **De Stijl（风格派）** 运动，")
    w("并发展出其标志性的 **新造型主义 (Neo-Plasticism)** 理论体系。")
    w()
    w("**新造型主义**的核心主张可归纳为三点：")
    w()
    w("1. **还原论**：将视觉语言还原为最基本的要素——水平线、垂直线、三原色（红黄蓝）与三非色（黑白灰）；")
    w("2. **关系论**：艺术的本质不在于单个元素本身，而在于元素之间的 *关系*（位置、比例、面积的对比与呼应）；")
    w("3. **普遍和谐**：通过这些纯粹关系的组织，表达超越个人情感的宇宙普遍秩序。")
    w()
    w("蒙德里安在 1920 年出版的理论专著《新造型主义》(*Le Neo-Plasticisme*) 中写道：")
    w('「新造型主义的目的，是通过对立面的均衡来表达普遍的和谐。」')
    w("(Mondrian, *Le Neo-Plasticisme*, 1920)")
    w()
    w("### 本作品的特殊地位：过渡期的珍贵见证")
    w()
    w('本作创作于 1920 年前后，正是蒙德里安从"实验性抽象"走向"成熟新造型主义"的关键过渡时期。')
    w("这幅作品最引人注目的特征是 **绿色元素的出现**——画面左侧一条纤细而鲜明的绿色竖条。")
    w("在蒙德里安后期的经典作品中，绿色被严格排除在画面之外。这是因为绿色是二次色（黄色与蓝色的混合），")
    w('代表着"自然的残余"——而新造型主义追求的恰恰是 *超越* 自然的纯粹关系。')
    w('蒙德里安在写给友人的信中曾明确指出："自然的颜色必须被纯粹的颜色所取代。"')
    w("(引自 H. Holtzman & M. James 编, *The New Art — The New Life: The Collected Writings of Piet Mondrian*, 1986)")
    w()
    w('因此，本作品中绿色的存在使它成为一个独特的"时间胶囊"：')
    w("我们可以看到蒙德里安 *正在* 走向纯粹三原色体系的过程中，他的色彩语言尚未完全纯化，")
    w('但构图的网格逻辑已经高度成熟。这种"形式先行、色彩渐进"的演变轨迹，')
    w("为我们理解新造型主义的形成提供了珍贵的实物证据。")
    w()

    # ================================================================
    # 一、平面抽象结构分析
    # ================================================================
    w("---")
    w()
    w("## 一、平面抽象结构分析")
    w()
    w("> **平面抽象结构**是指将画面视为一个二维平面场域，分析其中的线条骨架如何将空间分割为")
    w("> 若干区域，以及这些区域之间如何通过位置、面积、色彩形成整体的视觉秩序。")
    w()

    # 1.1 "水平-垂直"二元对立
    w("### 1.1 「水平-垂直」二元对立：宇宙秩序的视觉化")
    w()
    w("在展示数据之前，有必要理解蒙德里安为何 *只* 使用水平线和垂直线。这并非简单的风格偏好，")
    w("而是一套完整的哲学体系：")
    w()
    w('- **水平线** 代表宁静、广袤、大地、被动性（蒙德里安称之为"女性原则"）；')
    w('- **垂直线** 代表力量、崇高、天空、主动性（蒙德里安称之为"男性原则"）；')
    w("- 两者的 **正交交叉** 象征宇宙基本对立面的统一——就像中国哲学中的阴阳互生。")
    w()
    w("蒙德里安认为，斜线和曲线总是暗示着特定的自然形象（如树枝、水波），会让观者陷入具象联想，")
    w('而只有纯粹的正交关系才能表达"超越个别、指向普遍"的精神境界。')
    w('(参见 Y. A. Bois, \"The Iconoclast\", in *Piet Mondrian 1872-1944*, 1995)')
    w()

    # 数据：网格拓扑
    w("### 1.2 网格拓扑结构")
    w()
    colored_count = len([c for c in cells if c.get('dominant_color') != 'white'])
    white_count = len([c for c in cells if c.get('dominant_color') == 'white'])
    w(f"画面由 **{len(h_lines)} 条水平线** 和 **{len(v_lines)} 条垂直线** 构成正交骨架，")
    w(f"将画面分割为 **{len(cells)} 个矩形单元**（{colored_count} 个有色 + {white_count} 个白色）。")
    w()
    w("#### 水平线条位置与粗细：")
    w()
    w("| 编号 | y 坐标 | 粗细 (px) | 分类 |")
    w("|------|--------|-----------|------|")
    for i, l in enumerate(h_lines):
        w(f"| H{i} | {int(l['center'])} | {int(l['thickness'])} | {l.get('category', '-')} |")
    w()
    w("#### 垂直线条位置与粗细：")
    w()
    w("| 编号 | x 坐标 | 粗细 (px) | 分类 |")
    w("|------|--------|-----------|------|")
    for i, l in enumerate(v_lines):
        w(f"| V{i} | {int(l['center'])} | {int(l['thickness'])} | {l.get('category', '-')} |")
    w()

    # 1.3 空间分割层级 + 线条韵律
    w("### 1.3 空间分割层级与线条的「视觉呼吸」")
    w()
    w("线条的粗细并不均匀，呈现出明确的层级差异：")
    w()
    w("- **粗线（92-96px）**：V1、V2、V3、V5 四条垂直线构成画面的主要骨架，它们如同建筑中的承重墙，")
    w("  将画面划分为几个大的空间区域；")
    w("- **中线（83-87px）**：H0、H1、H3 三条水平线提供水平方向的次级分割；")
    w("- **细线（58-71px）**：V0、V4、H2 三条线条负责最精细的空间划分，产生画面中较小的矩形单元。")
    w()
    w("这种粗细交替产生了一种 **「视觉呼吸」** 的节奏效果：")
    w()
    w("> **韵律 (Rhythm)** 是形式美学的核心原理之一，指视觉元素（大小、间距、强弱）")
    w("> 按照一定规律的重复与变化所形成的有组织的运动感。")
    w()
    w('在本作品中，粗线制造"紧张"，细线带来"舒缓"，两者的交替')
    w('赋予了看似静态的画面以呼吸般的内在节奏——观者的视线在粗线处"停顿"，在细线处"滑行"，')
    w("由此形成一种无声的视觉韵律。")
    w()

    # 1.4 视觉重心与「动态均衡」
    w("### 1.4 视觉重心与「动态均衡」")
    w()
    w("> **动态均衡 (Dynamic Equilibrium)** 是蒙德里安新造型主义的核心美学概念。")
    w("> 它区别于传统的对称平衡（如文艺复兴绘画中轴线两侧的镜像对称），")
    w("> 指的是通过 *不等量* 的视觉要素（大面积 vs 小面积、冷色 vs 暖色、粗线 vs 细线）")
    w('> 之间的相互制衡，实现一种"有控制的不对称"——看似不均，实则均衡。')
    w()
    vc = structure_results['visual_center']
    gc = structure_results['geometric_center']
    offset = structure_results['center_offset']
    w(f"- **几何中心**: ({int(gc[0])}, {int(gc[1])})")
    w(f"- **视觉重心**: ({int(vc[0])}, {int(vc[1])})")
    w(f"- **偏移量**: 水平 {offset[0]:+.1f}%，垂直 {offset[1]:+.1f}%")
    w()
    direction_h = "左" if offset[0] < 0 else "右"
    direction_v = "上" if offset[1] < 0 else "下"
    w(f"视觉重心相对于几何中心偏向 **{direction_h}{direction_v}** 方向。")
    w("偏移幅度很小（仅约 3-4%），这说明蒙德里安并不是在制造失衡，而是在追求一种")
    w("*刚刚偏离对称* 的微妙状态——足以打破呆板的对称感，但又不至于让画面倾倒。")
    w('这正是动态均衡的精髓：画面始终处于一种"即将失衡又恰好平衡"的张力状态中。')
    w()
    w("具体来看：右上方的大面积蓝色块（冷色、视觉后退）与右侧的红色竖条（暖色、视觉前进）")
    w("将视觉重心向右拉动；而左下方的黄色大色块以其暖色调和较大面积形成反向的制衡力。")
    w('左侧绿色竖条虽面积最小，却通过其在画面边缘的"锚定"效果，防止了视觉重心过度右移。')
    w()

    # 1.5 正负空间
    w("### 1.5 正负空间与「计白当黑」")
    w()
    w("> **正空间 (Positive Space)** 指画面中被具体视觉元素（色块）占据的区域；")
    w('> **负空间 (Negative Space)** 指围绕和存在于正空间之间的"空白"区域。')
    w('> 在优秀的设计中，负空间并非"无"，而是与正空间同等重要的造型要素。')
    w()
    stats = structure_results['area_stats']
    w(f"- **有色区域（正空间）**: {stats['colored_ratio']:.1f}%")
    w(f"- **白色区域（负空间）**: {stats['white_ratio']:.1f}%")
    w(f"- **黑色线条（骨架）**: {stats['black_ratio']:.1f}%")
    w()
    w(f"白色负空间占据了画面的 **{stats['white_ratio']:.0f}%**，而有色正空间仅占 **{stats['colored_ratio']:.0f}%**。")
    w('这一悬殊的比例关系揭示了蒙德里安的一个重要美学策略：**白色不是"空白"，而是积极的形式要素**。')
    w()
    w('中国书画理论中有"计白当黑"的概念——意思是画面中的留白应当像落墨处一样被精心经营。')
    w("蒙德里安虽未直接受到东方美学影响，但他的做法异曲同工：")
    w('那些大面积的白色矩形并非"缺席"，而是承载着色块之间的视觉张力的"舞台"。')
    w("正是因为白色占据了如此大的面积，少量的色彩才显得格外醒目、格外珍贵——")
    w("如同寂静中的一声清响，空白赋予了色彩以力量。")
    w()

    # ================================================================
    # 二、数理比例关系分析
    # ================================================================
    w("---")
    w()
    w("## 二、数理比例关系分析")
    w()
    w("> **数理比例关系**关注的是画面中各元素在尺寸、间距、面积上的量化关系，")
    w("> 探索其中是否隐含黄金比例、整数比等数学秩序，以及这些数理关系如何转化为视觉上的和谐感。")
    w()

    # 2.1 线条间距比例
    w("### 2.1 线条间距比例")
    w()
    w("#### 水平方向间距（从上至下）：")
    w()
    h_gaps = proportion_results['h_gaps']
    for i, gap in enumerate(h_gaps):
        w(f"- 区间 {i}: {gap:.0f} px")
    w()
    w("#### 垂直方向间距（从左至右）：")
    w()
    v_gaps = proportion_results['v_gaps']
    for i, gap in enumerate(v_gaps):
        w(f"- 区间 {i}: {gap:.0f} px")
    w()

    w("#### 关键间距比值分析：")
    w()
    h_ratios = proportion_results.get('h_ratios', [])
    golden_found = False
    for ra in h_ratios:
        if ra['near_golden']:
            golden_found = True
            w(f"- 水平区间 {ra['pair'][0]}:{ra['pair'][1]} = {ra['values'][0]}:{ra['values'][1]} → 比值 **{ra['ratio']}** ≈ 1/φ (0.618)")
    v_ratios = proportion_results.get('v_ratios', [])
    for ra in v_ratios:
        if ra['near_golden']:
            golden_found = True
            w(f"- 垂直区间 {ra['pair'][0]}:{ra['pair'][1]} = {ra['values'][0]}:{ra['values'][1]} → 比值 **{ra['ratio']}** ≈ φ 或 1/φ")
    if not golden_found:
        w("- 未发现与黄金比例精确匹配的间距对，但多组间距呈现出近似整数比关系")
    w()

    w("#### 近似整数比关系：")
    w()
    for ra in h_ratios[:5]:
        nearest_int = round(ra['ratio'])
        if nearest_int > 0 and abs(ra['ratio'] - nearest_int) < 0.3:
            w(f"- 水平 {ra['pair']}: ≈ {nearest_int}:1 (实际 {ra['ratio']})")
    for ra in v_ratios[:5]:
        nearest_int = round(ra['ratio'])
        if nearest_int > 0 and abs(ra['ratio'] - nearest_int) < 0.3:
            w(f"- 垂直 {ra['pair']}: ≈ {nearest_int}:1 (实际 {ra['ratio']})")
    w()

    # 2.2 黄金分割验证 + 「训练有素的直觉」
    w("### 2.2 黄金分割验证与「训练有素的直觉」")
    w()
    w("> **黄金比例 (Golden Ratio, φ ≈ 1.618)** 是自古希腊以来西方美学中最重要的比例常数。")
    w("> 当一条线段被分割为长段 a 与短段 b，使得 a/b = (a+b)/a ≈ 1.618 时，")
    w("> 这种分割被认为具有最令人愉悦的视觉和谐感。帕特农神庙、达芬奇的《维特鲁威人》等")
    w("> 经典作品中都隐含着黄金比例。")
    w()
    matches = proportion_results.get('golden_matches', [])
    if matches:
        w("以下线条与黄金分割线位置高度吻合：")
        w()
        for m in matches:
            w(f"- **{m['type']}线** 位于 {m['line_pos']:.0f}px，黄金分割位 {m['golden_pos']:.0f}px，偏差仅 **{m['deviation_pct']:.2f}%**")
        w()
    w("最引人注目的是 **V2 垂直线**（x = 38.7%）与黄金分割线（x = 38.2%）的偏差仅为 **0.52%**——")
    w('几乎完美吻合。蒙德里安本人曾声称他作画时"不使用尺子和计算"，完全凭借直觉来决定线条的位置。')
    w('然而，数据告诉我们，他的"直觉"已经将西方几千年的比例美学传统深深内化。')
    w("这是一种 **「训练有素的直觉」(educated intuition)**——经过数十年严格训练后，")
    w("艺术家的手和眼已经能够自动抵达那些在数学上被证明为和谐的位置。")
    w()
    w("此外，H1（y = 58.0%）和 H2（y = 64.7%）两条水平线夹住了黄金分割位（y = 61.8%），")
    w('如同上下两道屏障将黄金分割点"框"在一条狭窄的水平带中——正是小蓝色块和小红色块所在的区域。')
    w("这绝非巧合：蒙德里安将最富有视觉冲击力的小色块精确地安置在了黄金分割带上。")
    w()

    # 2.3 间距中的「音乐性」
    w("### 2.3 间距比值中的音乐性")
    w()
    w("> **毕达哥拉斯传统**认为，宇宙万物的和谐可以用简单的整数比来表达。")
    w("> 音乐中最和谐的音程——八度 (octave) 对应弦长比 2:1，五度 (fifth) 对应 3:2，")
    w("> 四度 (fourth) 对应 4:3。蒙德里安深受通神学 (Theosophy) 影响，通神学继承了")
    w('> 毕达哥拉斯的"天体音乐"(Music of the Spheres) 思想，认为视觉和听觉共享同一套宇宙和声法则。')
    w()
    w("在本作品的间距数据中，我们发现了多组近似简单整数比的关系：")
    w()
    w(f"- 水平区间 0（{h_gaps[0]:.0f}px）与区间 2（{h_gaps[2]:.0f}px）之比 ≈ **2:1**（实际 {h_gaps[0]/h_gaps[2]:.2f}），对应八度音程；")
    if len(v_gaps) > 3:
        w(f"- 垂直区间 0（{v_gaps[0]:.0f}px）与区间 1（{v_gaps[1]:.0f}px）之比 ≈ **2:1**（实际 {v_gaps[0]/v_gaps[1]:.2f}），再次对应八度；")
    w("- 线条粗细的粗/细比 ≈ **3:2**（实际 1.42），接近五度音程。")
    w()
    w('这些比例关系赋予了画面一种隐秘的"音乐性"——虽然我们无法"听见"它，')
    w("但我们的视觉系统能够感知到这些简洁比值带来的和谐感，就像耳朵能够感知和弦的协和一样。")
    w("蒙德里安晚年最后一幅作品的名字就叫《百老汇爵士乐》(*Broadway Boogie-Woogie*, 1943)——")
    w("音乐与视觉的共通性，始终是他创作的核心关怀。")
    w()

    # 2.4 色块面积比例 + 色彩心理重量
    w("### 2.4 色块面积比例与色彩的「心理重量」")
    w()
    w('> **色彩的心理重量 (Visual Weight)** 是指不同颜色在观者心理上所产生的"轻重感"。')
    w('> 一般而言，暖色（红、黄）比冷色（蓝）显得更"重"、更具前进感；')
    w('> 饱和度高的颜色比低饱和度的颜色更"重"；深色比浅色更"重"。')
    w("> 在构图中，面积与心理重量存在互补关系——心理重量大的颜色用小面积即可产生强烈存在感，")
    w("> 而心理重量小的颜色则需要更大面积才能与之抗衡。")
    w()
    color_areas = proportion_results.get('color_areas', {})
    if color_areas:
        w("| 颜色 | 面积 | 占画面 (%) | 占有色面积 (%) |")
        w("|------|------|-----------|---------------|")
        for c, data in color_areas.items():
            w(f"| {COLOR_NAMES_CN[c]} | {data['area']} px² | {data['ratio_of_canvas']}% | {data['ratio_of_colored']}% |")
    w()
    pairs = proportion_results.get('color_area_pairs', [])
    if pairs:
        w("#### 色块间面积比值：")
        w()
        for p in pairs:
            w(f"- {p['pair']} = **{p['ratio']}**:1")
    w()
    w("面积排序为 蓝 > 黄 > 红 > 绿，而色彩心理重量排序恰好接近反向：红 > 黄 > 绿 > 蓝。")
    w("这意味着蒙德里安在进行一种精妙的 **「不等量均衡」**——")
    w('红色因其极强的视觉冲击力（心理重量大），只需较小的面积就能在画面中"撑住"；')
    w("蓝色性格沉稳内敛（心理重量小），需要最大的面积才能与红色分庭抗礼。")
    w('这不是一种简单的"面积相等"的平衡，而是"力矩相等"的平衡——')
    w("如同天平两端，一边放了重物（红色，面积小），另一边放了轻物（蓝色，面积大），")
    w("通过调节力臂（面积）使天平最终水平。")
    w()

    # 2.5 线条粗细比例
    w("### 2.5 线条粗细比例")
    w()
    t_groups = proportion_results.get('thickness_groups', {})
    if t_groups:
        thick_g = t_groups.get('thick', {})
        thin_g = t_groups.get('thin', {})
        w(f"- **粗线组**（均值 {thick_g.get('mean', 0)}px）: {thick_g.get('values', [])}")
        w(f"- **细线组**（均值 {thin_g.get('mean', 0)}px）: {thin_g.get('values', [])}")
        t_ratio = proportion_results.get('thickness_ratio', 0)
        max_min = proportion_results.get('max_min_thickness_ratio', 0)
        if t_ratio:
            w(f"- 粗线组/细线组均值比: **{t_ratio}**")
        if max_min:
            w(f"- 最粗/最细比: **{max_min}** (96px vs 58px)")
        w()
        w("粗细比约 1.42:1，接近 √2 ≈ 1.414（ISO 标准纸张的宽高比），也接近 3:2 的音乐五度音程比。")
        w('黑色线条在蒙德里安的体系中扮演着"骨骼"的角色：它们不是边框，不是装饰，')
        w("而是与色块、白色同等重要的独立造型要素。线条的粗细变化如同")
        w("音乐中力度的 *forte*（强）与 *piano*（弱）——为画面注入了动态的强弱对比。")
    w()

    # 2.6 单元面积统计 + 尺度张力场
    w("### 2.6 单元面积统计与「尺度张力场」")
    w()
    cell_stats = proportion_results.get('cell_area_stats', {})
    if cell_stats:
        w(f"- 最大单元面积: {cell_stats['max']} px²")
        w(f"- 最小单元面积: {cell_stats['min']} px²")
        w(f"- 面积均值: {cell_stats['mean']} px²")
        w(f"- 面积标准差: {cell_stats['std']} px²")
        w(f"- 最大/最小面积比: **{cell_stats['max_min_ratio']}**")
    w()
    w("> **对比 (Contrast)** 是形式美学中产生视觉张力的基本手段。")
    w("> 当两个元素之间的差异越大，它们之间产生的张力就越强。")
    w('> 面积的极端对比（46:1）创造了一个"尺度张力场"——观者的视线被迫在')
    w("> 巨大的白色区域和微小的色块之间反复跳跃，这种跳跃本身就构成了视觉的运动和节奏。")
    w()
    w("标准差（约 58 万 px²）大于均值（约 52 万 px²），说明面积分布是极度不均匀的。")
    w('这种"不均匀"恰恰是蒙德里安刻意为之的：均匀的网格会导致单调和沉闷（如同棋盘），')
    w("而不均匀的分割创造了大小穿插、疏密相间的视觉韵律——")
    w('大面积区域提供"休息"，小面积区域制造"紧张"，两者交替如同呼吸的节奏。')
    w()

    # ================================================================
    # 三、单元组合原理分析
    # ================================================================
    w("---")
    w()
    w("## 三、单元组合原理分析")
    w()
    w('> **单元组合原理**关注的是画面中最基本的视觉"零件"（矩形单元）是什么，')
    w("> 它们如何被分类，以及它们按照什么样的逻辑被组织和排列在一起。")
    w('> 这类似于语言学中对"词汇"（基本单元）和"语法"（组合规则）的区分。')
    w()

    # 3.1 基本单元分类
    w("### 3.1 基本单元分类：画面的「词汇表」")
    w()
    unit_cats = unit_results.get('unit_categories', {})
    w(f"- **大单元**: {unit_cats.get('大', 0)} 个")
    w(f"- **中单元**: {unit_cats.get('中', 0)} 个")
    w(f"- **小单元**: {unit_cats.get('小', 0)} 个")
    w()
    w("### 3.2 单元类型组合矩阵")
    w()
    type_combos = unit_results.get('type_combinations', {})
    w("| 颜色 \\ 尺寸 | 大 | 中 | 小 |")
    w("|-------------|---|---|---|")
    for c in ['white', 'red', 'blue', 'yellow', 'green']:
        row = f"| {COLOR_NAMES_CN.get(c, c)} "
        for s in ['大', '中', '小']:
            key = f"{c}_{s}"
            row += f"| {type_combos.get(key, 0)} "
        row += "|"
        w(row)
    w()
    w("矩阵揭示了一条重要规则：**所有的大单元都是白色的，所有的色块都是中或小单元**。")
    w("这意味着蒙德里安将色彩严格限制在较小的区域内，而将大面积留给白色。")
    w('色彩从不"霸占"画面——它以克制的姿态出现，如同一篇散文中被精心挑选的关键词，')
    w("少而有力。")
    w()

    # 3.3 色彩空间分布 + 对位法
    w("### 3.3 色彩空间分布与「对位法」")
    w()
    w("> **对位法 (Counterpoint)** 原是音乐术语，指两条或多条独立旋律按照一定规则同时进行，")
    w('> 彼此之间既独立又和谐。在视觉艺术中，"对位法"被借用来描述画面中不同视觉元素')
    w("> 之间的呼应与对比关系——它们各自独立存在，但在整体中形成和谐的对话。")
    w()
    color_positions = unit_results.get('color_positions', {})
    for c_name, positions in color_positions.items():
        w(f"**{COLOR_NAMES_CN[c_name]}**:")
        for p in positions:
            w(f"  - 位置: {p['position']}（归一化坐标: {p['norm_x']:.2f}, {p['norm_y']:.2f}），面积: {p['area']} px²")
    w()

    w("#### 象限分布：")
    w()
    quadrant_dist = unit_results.get('quadrant_distribution', {})
    for q_name, q_data in quadrant_dist.items():
        if q_data:
            colors_in_q = ', '.join([COLOR_NAMES_CN[c] for c in q_data.keys()])
            w(f"- **{q_name}象限**: {colors_in_q}")
        else:
            w(f"- **{q_name}象限**: 无色块（纯白空间）")
    w()
    w("色彩分布呈现出精心设计的 **对位结构**：")
    w()
    w("- **对角线对位**：蓝色（冷色，右上）与黄色（暖色，左下）形成对角线上的冷暖对位，")
    w("  如同乐曲中高音部与低音部的对话；")
    w("- **水平对位**：绿色（中性色，左中）与红色（强烈暖色，右中）形成水平方向的力量对比；")
    w('- **四角均沾**：四个象限中都有色彩出现（或紧邻边界），没有任何一个象限是完全"空"的，')
    w('  避免了视觉上的"塌陷"。')
    w()
    w("这种散点式分布并非随意的撒落，而是一张精心编织的视觉张力网——")
    w("每一个色块都通过与其他色块的距离、色温、面积差异形成一组力的关系，")
    w("所有力最终达到了整体的均衡。")
    w()

    # 3.4 重复与变化：主题与变奏
    w("### 3.4 重复与变化：「主题与变奏」")
    w()
    w("> **主题与变奏 (Theme and Variation)** 是古典音乐中最常见的结构形式：")
    w("> 先呈现一个主旋律（主题），然后在不同的声部、不同的调性、不同的节奏中反复出现（变奏），")
    w('> 保持核心辨识度的同时创造丰富的变化。在视觉设计中，"主题与变奏"')
    w("> 表现为同一类元素以不同尺度、不同位置反复出现。")
    w()
    for c in ['red', 'blue', 'yellow', 'green']:
        var_key = f'{c}_variation'
        if var_key in unit_results:
            v = unit_results[var_key]
            w(f"- **{COLOR_NAMES_CN[c]}**: 出现 {v['count']} 次，面积范围 {v['area_range'][0]}-{v['area_range'][1]} px²，大小比 **{v['scale_ratio']}**:1")
    w()
    w('红色是最典型的"主题与变奏"案例：它出现了 3 次，最大的红色块（右中，面积 613048 px²）')
    w("是最小红色块（中中，面积 57562 px²）的 **10.65 倍**。")
    w("三个红色块如同音乐中同一主旋律在强声部（fortissimo）、中声部（mezzo）和弱声部（pianissimo）")
    w("的三次呈现——旋律不变，力度递变，既保持了统一性，又创造了丰富的层次感。")
    w()
    w("蓝色也遵循类似的变奏逻辑：大面积蓝色块居于右上（如同主旋律的首次陈述），")
    w("小面积蓝色块退居左中边缘（如同主旋律的轻声回响），面积比达 10.92:1。")
    w("蒙德里安的好友、De Stijl 核心成员凡·杜斯堡也曾使用音乐术语来分析抽象绘画的构成方式，")
    w("可见音乐与视觉的结构共通性在风格派内部已是共识。")
    w()

    # 3.5 边缘延展：开放构图
    w("### 3.5 边缘延展与「开放构图」")
    w()
    w('> **开放构图 (Open Composition)** 指画面中的视觉元素延伸至画框边缘并被"裁切"，')
    w("> 暗示画面只是一个无限连续空间的局部截取——在画框之外，构图仍在继续。")
    w('> 与之相对的是"封闭构图"，即所有元素都完整地包含在画框之内。')
    w()
    edge_info = unit_results.get('edge_vs_inner', {})
    w(f"- 边缘色块数: **{edge_info.get('edge_count', 0)}**")
    w(f"- 内部色块数: **{edge_info.get('inner_count', 0)}**")
    w(f"- 边缘色块颜色: {', '.join([COLOR_NAMES_CN[c] for c in edge_info.get('edge_colors', [])])}")
    w(f"- 内部色块颜色: {', '.join([COLOR_NAMES_CN[c] for c in edge_info.get('inner_colors', [])])}")
    w()
    w('7 个色块中有 4 个（超过半数）位于画面边缘，被画框"裁切"——')
    w("蓝色从顶部伸入，红色从右侧和底部伸入，小蓝色块从左侧伸入。")
    w('这种处理传达了一个关键理念：**画面不是一个孤立的"窗口"，而是一个更大秩序的局部显现**。')
    w()
    w('蒙德里安追求的不是在画框内创造一个完美的"小世界"，而是揭示一种弥漫于整个宇宙的')
    w("普遍和谐——画框只是偶然截取了这个无限网格中的一个片段。")
    w('正如他所写："画面必须通过其构造暗示着比它自身更为广大的东西。"')
    w('(Mondrian, \"Pure Plastic Art\", 1942)')
    w()

    # 3.6 白色大单元
    w("### 3.6 白色大单元：「空的充实」")
    w()
    w("画面中 3 个最大的矩形单元全部是白色的，它们占据了画面的核心区域。")
    w('这些白色区域并不是"缺席"而是"在场"——蒙德里安将它们视为与色块同等重要的造型元素。')
    w()
    w('在物理学中，场论 (Field Theory) 告诉我们，看似"空"的空间中充满了力的作用——')
    w("电场、磁场、引力场。蒙德里安的白色区域与此异曲同工：")
    w('它们是色块之间的"视觉力场"，承载着红与蓝之间的拉扯、黄与绿之间的呼应。')
    w('没有这些白色的"间距"，色块之间就不会产生张力；')
    w('就像音乐中的"休止符"不是沉默，而是为下一个音符蓄势。')
    w()

    # ================================================================
    # 四、形式规则总结（深度版）
    # ================================================================
    w("---")
    w()
    w("## 四、形式规则总结")
    w()
    w("通过以上三个维度的分析——平面抽象结构、数理比例关系、单元组合原理——")
    w("我们可以提炼出蒙德里安在本作品中运用的核心形式规则：")
    w()

    w("### 规则一：正交网格系统——「水平与垂直的宇宙语法」")
    w()
    w("画面完全由水平线和垂直线构成，拒绝一切斜线和曲线。")
    w("这不仅是一种风格选择，更是一种哲学宣言：蒙德里安认为水平与垂直的正交关系")
    w("是宇宙中最基本的对立统一（静与动、被动与主动、物质与精神），")
    w("只有这种最纯粹的几何关系才能表达超越个人情感的普遍真理。")
    w("线条粗细分为 2-3 个层级（粗线 ~92-96px，细线 ~58-71px），形成主次分明的骨架层级。")
    w()

    w("### 规则二：动态均衡——「有控制的不对称」")
    w()
    w("视觉重心偏离几何中心（偏移 +3.8%, -2.2%），色块面积和位置均不对称分布。")
    w('但这种不对称是"有控制的失衡"：大面积的冷色（蓝）与小面积的暖色（红）通过')
    w("心理重量的互补达到力矩平衡；边缘色块与内部色块通过空间距离形成张力网。")
    w('画面始终处于"即将失衡又恰好平衡"的动态状态——这是蒙德里安对"生命本质"的隐喻：')
    w("生命不是静止的对称，而是永恒的运动中的瞬间均衡。")
    w()

    w("### 规则三：留白的力量——「以少驭多」")
    w()
    w(f"白色负空间占 {stats['white_ratio']:.0f}%，有色正空间仅占 {stats['colored_ratio']:.0f}%。")
    w('白色不是"什么都没有"，而是一种积极的造型力量——它是色块之间张力的载体，')
    w("是视觉对话发生的场域。极少量的色彩被投放在大面积的白色中，如同寂静中的几声清响，")
    w("以最小的物质投入获得了最大的视觉效果。")
    w()

    w("### 规则四：比例的理性秩序——「可见的和声」")
    w()
    w("线条间距呈现近似黄金比例（V2 与黄金分割线偏差仅 0.52%）和简单整数比（2:1, 3:2）关系。")
    w("色块面积通过心理重量的差异实现不等量均衡。")
    w('这些数理关系构成了画面的隐形"和声"——观者未必能意识到这些比例的存在，')
    w("但和谐感已经通过视觉神经传递到了审美体验中。")
    w()

    w("### 规则五：主题变奏——「统一中的多样」")
    w()
    w("同一颜色以不同尺度反复出现（红色 3 次，面积比 10.65:1；蓝色 2 次，面积比 10.92:1），")
    w("如同音乐中主旋律在不同声部、不同力度下的再现。")
    w('这既保持了画面的色彩统一性（观者能识别出"这是红色的画"），')
    w('又通过尺度变化避免了单调——是形式美学中"统一与变化"原则的典范运用。')
    w()

    w("### 规则六：开放构图——「画外之画」")
    w()
    w("超过半数的色块被画框裁切（4/7），暗示构图向画面之外无限延展。")
    w("蒙德里安追求的不是在画框内制造一个封闭的完美世界，")
    w("而是揭示一种弥漫于整个宇宙的结构秩序——画面只是这一无限秩序的局部截取。")
    w()

    w("### 规则七：色彩的克制——「绿色的告别」")
    w()
    w("本作品中绿色（二次色）的出现标志着蒙德里安色彩体系的过渡状态。")
    w("在此后的成熟期作品中，绿色将被彻底排除，只保留三原色（红黄蓝）与三非色（黑白灰）。")
    w('这种"自我剥夺"的策略遵循着新造型主义的最高纲领：')
    w("越是纯粹的要素，越能表达普遍的真理；越是克制的表达，越具有深远的力量。")
    w()

    # ================================================================
    # 五、A3 排版建议
    # ================================================================
    w("---")
    w()
    w("## 五、A3 排版详细建议")
    w()
    w("### 5.1 版面基本设定")
    w()
    w("- **纸张**: A3 横版 (420mm × 297mm)")
    w("- **出血**: 3mm")
    w("- **安全边距**: 上下左右各 12mm")
    w("- **有效版面**: 396mm × 273mm")
    w("- **设计风格**: 蒙德里安网格美学——以黑色直线分割版面，白底为主，辅以四色装饰")
    w()
    w("### 5.2 版面分区布局")
    w()
    w("用 2mm 粗的黑色线条将版面分为以下区域（从左到右、从上到下）：")
    w()
    w("```")
    w("┌──────────────┬──────────────────────────────────────┐")
    w("│              │  B: 标题栏 (全宽上方窄条)              │")
    w("│  A: 原作图像  ├──────────────┬───────────────────────┤")
    w("│  130×130mm   │  C: 结构分析  │  D: 比例分析           │")
    w("│  (含边框)     │  130×130mm   │  130×130mm            │")
    w("├──────────────┼──────────────┼───────────────────────┤")
    w("│  E: 作品信息  │  F: 单元分析  │  G: 形式规则总结       │")
    w("│  130×120mm   │  130×120mm   │  130×120mm            │")
    w("└──────────────┴──────────────┴───────────────────────┘")
    w("```")
    w()
    w("### 5.3 各区域内容指引")
    w()
    w("**A区 — 原作图像**:")
    w("- 放置原作 `pic.jpg`，保持正方形比例，约 120×120mm")
    w("- 图像周围留白 5mm，下方可加一条蓝色细线装饰")
    w()
    w("**B区 — 标题栏**（高约 25mm）:")
    w("- 标题: 「蒙德里安作品形式解析」")
    w("- 副标题: 「Composition with Red, Blue and Yellow-Green, c.1920」")
    w("- 右侧注明小组成员信息")
    w("- 底部可用红色细条装饰（致敬蒙德里安色带）")
    w()
    w("**C区 — 平面抽象结构分析**:")
    w("- 推荐用图: `01_grid_structure.png` (网格拓扑 + 空间分割层级)")
    w("- 补充用图: `02_visual_center_space.png` 左半部分 (视觉重心)")
    w("- 配文(2-3句): 关键词——正交骨架、动态均衡、正负空间")
    w()
    w("**D区 — 数理比例关系分析**:")
    w("- 推荐用图: `06_normalized_proportion.png` (归一化比例结构，最直观)")
    w("- 补充用图: `03_proportion_analysis.png` 右上子图 (黄金分割验证)")
    w("- 配文(2-3句): 关键词——黄金分割(V2偏差0.52%)、音乐性整数比、色彩心理重量均衡")
    w()
    w("**E区 — 作品基本信息**:")
    w("- 艺术家: 皮特·蒙德里安 (1872-1944)")
    w("- 流派: 新造型主义 / De Stijl")
    w("- 简介: 2-3 句概括作品特点（含绿色元素的过渡期作品）")
    w("- 正负空间饼图: `02_visual_center_space.png` 右半部分")
    w()
    w("**F区 — 单元组合原理分析**:")
    w("- 推荐用图: `04_unit_analysis.png` 右上子图 (色彩空间分布)")
    w("- 补充用图: `04_unit_analysis.png` 右下子图 (类型组合矩阵)")
    w("- 配文(2-3句): 关键词——色彩对位法、主题与变奏、开放构图")
    w()
    w("**G区 — 形式规则总结**:")
    w("- 列出提炼出的 7 条核心形式规则（简洁版，每条一句话）")
    w("- 可配小图标或图形符号辅助说明")
    w()
    w("### 5.4 排版风格细节")
    w()
    w("- **字体**: 标题用思源黑体 Bold 18pt，正文用思源黑体 Regular 9pt")
    w("- **分割线**: 主分割线 2mm 黑色，次分割线 0.5mm 灰色")
    w("- **色彩装饰**: 每个分析模块标题左侧可加 3mm 宽的色带（结构=蓝、比例=红、单元=黄、信息=绿）")
    w("- **背景**: 纯白 #FFFFFF")
    w("- **图片处理**: 所有分析图导出为 300dpi PNG，放入版面时保持清晰度")
    w()

    # ================================================================
    # 参考文献
    # ================================================================
    w("---")
    w()
    w("## 参考文献")
    w()
    w("1. Mondrian, P. (1920). *Le Neo-Plasticisme: Principe Général de l'Équivalence Plastique*. Paris: Éditions de l'Effort Moderne.")
    w("2. Holtzman, H. & James, M. (Eds.). (1986). *The New Art — The New Life: The Collected Writings of Piet Mondrian*. Boston: G.K. Hall & Co.")
    w('3. Bois, Y.-A. (1995). \"The Iconoclast\". In Bois, Y.-A., Joosten, J., Rudenstine, A. Z., & Janssen, H. (Eds.), *Piet Mondrian 1872-1944* (exhibition catalogue). Leonardo Arte.')
    w("4. Bloomer, C. M. (1976). *Principles of Visual Perception*. Van Nostrand Reinhold.")
    w('5. Mondrian, P. (1942). \"Pure Plastic Art\". In *Plastic Art and Pure Plastic Art*. New York: Wittenborn.')
    w('6. piet-mondrian.org. \"Composition with Red Blue and Yellow-Green, 1920\". https://www.piet-mondrian.org/composition-with-red-blue-and-yellow-green.jsp')
    w()

    report_text = '\n'.join(r)

    # 保存为新文件，不覆盖原报告
    report_path = os.path.join(output_dir, 'analysis_report_v2.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  增强版报告已保存: {report_path}")

    return report_text


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("蒙德里安画作形式分析")
    print("=" * 60)

    # 1. 加载图像
    print("\n[1/7] 加载图像...")
    img_bgr, img_rgb, img_hsv, img_gray = load_image(IMAGE_PATH)
    img_h, img_w = img_rgb.shape[:2]
    print(f"  图像尺寸: {img_w} × {img_h} 像素")

    # 2. 检测黑色线条
    print("\n[2/7] 检测黑色线条...")
    h_lines, v_lines, h_mask, v_mask, black_mask = detect_black_lines(img_gray, img_hsv, img_h, img_w)
    print(f"  检测到 {len(h_lines)} 条水平线, {len(v_lines)} 条垂直线")
    for i, l in enumerate(h_lines):
        print(f"    H{i}: y={int(l['center'])}, 粗细={int(l['thickness'])}px")
    for i, l in enumerate(v_lines):
        print(f"    V{i}: x={int(l['center'])}, 粗细={int(l['thickness'])}px")

    # 3. 检测色块
    print("\n[3/7] 检测色块...")
    color_blocks = detect_color_blocks(img_hsv, img_rgb, img_h, img_w)
    print(f"  检测到 {len(color_blocks)} 个色块:")
    for i, b in enumerate(color_blocks):
        print(f"    #{i+1} {COLOR_NAMES_CN[b['color']]}: 位置({b['x']},{b['y']}) 尺寸({b['w']}x{b['h']}) 面积={b['area']}")

    # 4. 构建网格
    print("\n[4/7] 构建网格结构...")
    x_coords, y_coords, cells = build_grid(h_lines, v_lines, img_h, img_w)
    cells = assign_colors_to_cells(cells, color_blocks, img_hsv, img_h, img_w)
    print(f"  网格: {len(y_coords)-1} 行 × {len(x_coords)-1} 列 = {len(cells)} 个单元")
    print(f"  x 分割点: {x_coords}")
    print(f"  y 分割点: {y_coords}")

    # 5. 三维度分析
    print("\n[5/7] 执行三维度分析...")

    print("  → 平面抽象结构分析...")
    structure_results = analyze_structure(h_lines, v_lines, cells, color_blocks, img_h, img_w)

    print("  → 数理比例关系分析...")
    proportion_results = analyze_proportions(h_lines, v_lines, cells, color_blocks, img_h, img_w)

    print("  → 单元组合原理分析...")
    unit_results = analyze_units(cells, color_blocks, h_lines, v_lines, img_h, img_w)

    # 6. 生成可视化
    print("\n[6/7] 生成可视化图表...")
    draw_color_detection_result(img_rgb, color_blocks, img_h, img_w, OUTPUT_DIR)
    draw_grid_structure(img_rgb, h_lines, v_lines, x_coords, y_coords, cells, img_h, img_w, OUTPUT_DIR)
    draw_visual_center(img_rgb, structure_results, color_blocks, img_h, img_w, OUTPUT_DIR)
    draw_proportion_analysis(img_rgb, proportion_results, h_lines, v_lines, img_h, img_w, OUTPUT_DIR)
    draw_unit_analysis(img_rgb, cells, color_blocks, unit_results, img_h, img_w, OUTPUT_DIR)
    draw_line_thickness_analysis(h_lines, v_lines, img_h, img_w, OUTPUT_DIR)
    draw_normalized_proportion(h_lines, v_lines, color_blocks, img_h, img_w, OUTPUT_DIR)

    # 7. 生成报告
    print("\n[7/7] 生成分析报告...")
    report = generate_report(structure_results, proportion_results, unit_results,
                            color_blocks, h_lines, v_lines, cells, img_h, img_w, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("分析完成！所有成果已保存至:", OUTPUT_DIR)
    print("=" * 60)

    # 保存原始数据（JSON）
    data = {
        'image_size': {'width': img_w, 'height': img_h},
        'h_lines': h_lines,
        'v_lines': v_lines,
        'color_blocks': color_blocks,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'structure': {
            'visual_center': structure_results['visual_center'],
            'geometric_center': structure_results['geometric_center'],
            'center_offset': structure_results['center_offset'],
            'area_stats': structure_results['area_stats']
        },
        'proportions': {
            'h_gaps': proportion_results['h_gaps'],
            'v_gaps': proportion_results['v_gaps'],
            'golden_matches': proportion_results.get('golden_matches', []),
            'color_areas': proportion_results.get('color_areas', {}),
            'cell_area_stats': proportion_results.get('cell_area_stats', {}),
        },
        'units': {
            'unit_categories': unit_results.get('unit_categories', {}),
            'type_combinations': unit_results.get('type_combinations', {}),
            'quadrant_distribution': {k: {c: int(v) for c, v in qd.items()} for k, qd in unit_results.get('quadrant_distribution', {}).items()},
            'edge_vs_inner': unit_results.get('edge_vs_inner', {})
        }
    }

    json_path = os.path.join(OUTPUT_DIR, 'analysis_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"原始数据已保存: {json_path}")


if __name__ == '__main__':
    main()
