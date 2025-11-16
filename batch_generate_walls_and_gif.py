#!/usr/bin/env python3
"""
批量处理雷达数据，生成墙体检测gif动画
- 读取目标路径所有雷达数据
- 生成图6的gif（最终墙体可视化）
- 为每个文件生成wall JSON文件
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
from tqdm import tqdm
import imageio


# =========================
# 全局可配置参数（重要变量）
# =========================
# 输入目录
INPUT_DIR = "/home/qzl/test_road/test_online_road/extracted_lidar_data_code/extracted_lidar_data/laserscan_json2"
# LaserScan 最大量程（米），None 表示不裁剪
MAX_RANGE = 50.0
# NPY 模式下的 Z 高度过滤（米）
Z_MIN, Z_MAX = -0.04, 0.04
# XY 空间过滤（米）
X_LIMIT = 10.0   # ±10m
Y_LIMIT = 5.0    # ±5m

# ===== 扫描顺序分段参数 =====
JUMP_DIST_THRESH = 0.2      # Jump-distance阈值（米）
MIN_SEGMENT_POINTS = 8       # 最少点数

# ===== RDP算法参数 =====
RDP_EPSILON = 0.05   # 最大允许偏差（米）
MIN_SPLIT_POINTS = 6         # RDP分割后子段最少点数

# ===== PCA拟合过滤参数 =====
WALL_RMSE_THRESH = 0.07      # 线段RMSE阈值（米）
MIN_SEGMENT_LENGTH = 0.5      # 最短线段长度（米）- 允许保留短片段供后续合并

# ===== 参数空间DBSCAN墙体聚类参数 =====
PARAM_DBSCAN_EPS = 0.3       # (θ, ρ)空间的聚类半径（主要控制ρ距离，单位：米）
PARAM_DBSCAN_MIN_SAMPLES = 1 # 最少线段数形成墙体
ALPHA_SCALE = 0.30             # θ的缩放系数（缩小角度以让eps主要控制距离） 0.3米对应约5.7度

# ===== 1D区间合并参数 =====
GAP_THRESH = 2             # 允许的小gap（米），用于合并门洞、遮挡
MIN_WALL_LENGTH = 0.8       # 最短墙体长度（米）
MIN_SEGMENTS_PER_WALL = 1    # 每面墙最少线段数

# ===== 输出目录 =====
OUT_DIR = "/home/qzl/Main/MobiMind/test/test_online_road/extracted_lidar_data_code/extracted_lidar_data/RDP_DBSCAN_scan_batch"
WALL_OUT_DIR = "/home/qzl/Main/MobiMind/test/test_online_road/extracted_lidar_data_code/wall_batch"
GIF_OUTPUT = "/home/qzl/Main/MobiMind/test/test_online_road/extracted_lidar_data_code/walls_animation.gif"

# GIF参数
GIF_FPS = 10  # 帧率
GIF_DURATION = 0.1  # 每帧持续时间（秒）


# =========================================================================
# ===== 复用原始代码中的核心函数 =====
# =========================================================================

def _laserscan_to_xy(laserscan: dict, max_range: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """LaserScan(JSON) → 笛卡尔坐标"""
    angle_min = laserscan['angle_min']
    angle_max = laserscan.get('angle_max', None)
    angle_increment = laserscan['angle_increment']
    ranges = np.asarray(laserscan['ranges'], dtype=np.float64)
    # 有效性与量程过滤
    valid = np.isfinite(ranges) & (ranges > 0.0)
    if max_range is not None:
        valid &= (ranges <= max_range)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64), np.array([], dtype=int)
    valid_indices = np.nonzero(valid)[0]
    ranges = ranges[valid]
    # 角度序列
    if angle_max is not None:
        num = len(laserscan['ranges'])
        thetas = angle_min + np.arange(num) * angle_increment
        thetas = thetas[valid]
    else:
        thetas = angle_min + np.arange(len(ranges)) * angle_increment
    x = ranges * np.cos(thetas)
    y = ranges * np.sin(thetas)
    return np.column_stack([x, y]), valid_indices


def _segment_by_jump(points_xy_ordered: np.ndarray,
                     jump_thresh: float,
                     min_points: int) -> list[np.ndarray]:
    """基于扫描顺序的 1D 跳变分段"""
    if len(points_xy_ordered) == 0:
        return []
    segments = []
    start_idx = 0
    for i in range(1, len(points_xy_ordered)):
        d = np.linalg.norm(points_xy_ordered[i] - points_xy_ordered[i-1])
        if not np.isfinite(d) or d > jump_thresh:
            seg = points_xy_ordered[start_idx:i]
            if len(seg) >= min_points:
                segments.append(seg)
            start_idx = i
    # last segment
    last = points_xy_ordered[start_idx:]
    if len(last) >= min_points:
        segments.append(last)
    return segments


def _fit_line_pca(points: np.ndarray) -> dict | None:
    """PCA 直线拟合，返回法线式 ax+by+c=0 与方向、端点、RMSE"""
    if len(points) < 2:
        return None
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    # 法向量
    a = -direction[1]
    b = direction[0]
    c = -(a * mean[0] + b * mean[1])
    # 投影确定端点
    t_vals = np.dot(points - mean, direction)
    t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))
    p1 = mean + t_min * direction
    p2 = mean + t_max * direction
    # 点到线距离
    denom = max(np.sqrt(a*a + b*b), 1e-12)
    dists = np.abs(a * points[:, 0] + b * points[:, 1] + c) / denom
    rmse = float(np.sqrt(np.mean(dists**2))) if len(dists) > 0 else 0.0
    return {
        'line': (float(a), float(b), float(c)),
        'dir': direction.astype(float),
        'mean': mean.astype(float),
        'p1': p1.astype(float),
        'p2': p2.astype(float),
        'rmse': rmse,
        'length': float(np.linalg.norm(p2 - p1)),
    }


def _points_line_distance_vectorized(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算多个点到线段 ab 的垂直距离（向量化版本）"""
    ab = b - a
    if np.allclose(ab, 0):
        return np.linalg.norm(points - a, axis=1)
    ap = points - a
    ab_squared = np.dot(ab, ab)
    t = np.dot(ap, ab) / ab_squared
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, np.newaxis] * ab
    return np.linalg.norm(points - proj, axis=1)


def _rdp_recursive(points: np.ndarray,
                   start: int,
                   end: int,
                   eps: float,
                   keep_idx: set) -> None:
    """RDP 算法的递归实现（向量化优化版本）"""
    if end <= start + 1:
        return
    a = points[start]
    b = points[end]
    if end > start + 1:
        middle_points = points[start + 1:end]
        distances = _points_line_distance_vectorized(middle_points, a, b)
        max_idx_relative = np.argmax(distances)
        max_dist = distances[max_idx_relative]
        max_idx = start + 1 + max_idx_relative
    else:
        max_dist = -1.0
        max_idx = -1
    if max_dist > eps:
        keep_idx.add(max_idx)
        _rdp_recursive(points, start, max_idx, eps, keep_idx)
        _rdp_recursive(points, max_idx, end, eps, keep_idx)


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker 算法主函数"""
    n = len(points)
    if n <= 2:
        return np.arange(n)
    keep_idx = {0, n - 1}
    _rdp_recursive(points, 0, n - 1, eps, keep_idx)
    idx = np.array(sorted(list(keep_idx)), dtype=int)
    return idx


def _split_by_rdp(points_ordered: np.ndarray,
                  eps: float,
                  min_points: int) -> list[np.ndarray]:
    """使用 RDP 算法对一段有序点进行分割"""
    if len(points_ordered) < 2:
        return []
    key_idx = _rdp(points_ordered, eps)
    segments = []
    for i in range(len(key_idx) - 1):
        s = key_idx[i]
        e = key_idx[i + 1]
        seg_points = points_ordered[s:e+1]
        if len(seg_points) >= min_points:
            segments.append(seg_points)
    return segments


def _segment_to_hessian(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    """将线段转换为Hessian法线形式 (θ, ρ)"""
    m = 0.5 * (p1 + p2)
    d = p2 - p1
    dx, dy = d[0], d[1]
    theta = np.arctan2(dy, dx)
    if theta < 0:
        theta += np.pi
    n = np.array([-np.sin(theta), np.cos(theta)])
    rho = np.dot(n, m)
    return float(theta), float(rho)


def _merge_intervals_1d(intervals: list[tuple[float, float]],
                        gap_thresh: float) -> list[tuple[float, float]]:
    """合并1D区间，容忍小gap"""
    if len(intervals) == 0:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    current_min, current_max = sorted_intervals[0]
    for t_min, t_max in sorted_intervals[1:]:
        if t_min - current_max <= gap_thresh:
            current_max = max(current_max, t_max)
        else:
            merged.append((current_min, current_max))
            current_min, current_max = t_min, t_max
    merged.append((current_min, current_max))
    return merged


def _cluster_segments_in_param_space(fitted_segments: list[dict],
                                     eps: float,
                                     min_samples: int,
                                     alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 (θ, ρ) 参数空间对线段进行DBSCAN聚类"""
    if len(fitted_segments) == 0:
        return np.array([]), np.array([]), np.array([])
    n_segs = len(fitted_segments)
    features = np.zeros((n_segs, 2), dtype=np.float64)
    weights = np.zeros(n_segs, dtype=np.float64)
    for i, seg_dict in enumerate(fitted_segments):
        fit = seg_dict['fit']
        p1 = fit['p1']
        p2 = fit['p2']
        theta, rho = _segment_to_hessian(p1, p2)
        features[i, 0] = theta / alpha
        features[i, 1] = rho
        weights[i] = fit['length']
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = db.labels_
    return features, weights, labels


def _merge_segments_in_cluster(cluster_segments: list[dict],
                                gap_thresh: float) -> list[dict]:
    """对同一簇内的线段，沿墙体方向合并成完整墙体"""
    if len(cluster_segments) == 0:
        return []
    total_weight = 0.0
    weighted_dir = np.zeros(2)
    for seg_dict in cluster_segments:
        fit = seg_dict['fit']
        direction = fit['dir']
        length = fit['length']
        weighted_dir += direction * length
        total_weight += length
    if total_weight > 0:
        wall_dir = weighted_dir / np.linalg.norm(weighted_dir)
    else:
        wall_dir = cluster_segments[0]['fit']['dir']
    intervals = []
    all_points = []
    for seg_dict in cluster_segments:
        fit = seg_dict['fit']
        points = seg_dict['points']
        p1, p2 = fit['p1'], fit['p2']
        t1 = np.dot(p1, wall_dir)
        t2 = np.dot(p2, wall_dir)
        t_min, t_max = min(t1, t2), max(t1, t2)
        intervals.append((t_min, t_max))
        all_points.append(points)
    merged_intervals = _merge_intervals_1d(intervals, gap_thresh)
    walls = []
    all_points_combined = np.vstack(all_points) if len(all_points) > 0 else np.empty((0, 2))
    for t_min, t_max in merged_intervals:
        wall_normal = np.array([-wall_dir[1], wall_dir[0]])
        if len(all_points_combined) > 0:
            t_vals = np.dot(all_points_combined, wall_dir)
            mask = (t_vals >= t_min - 0.01) & (t_vals <= t_max + 0.01)
            interval_points = all_points_combined[mask]
        else:
            interval_points = np.empty((0, 2))
        if len(interval_points) >= 2:
            fit = _fit_line_pca(interval_points)
            if fit is not None:
                walls.append({
                    'fit': fit,
                    'points': interval_points,
                    'is_wall': True
                })
        else:
            rho_avg = 0.0
            if len(all_points_combined) > 0:
                rho_vals = np.dot(all_points_combined, wall_normal)
                rho_avg = np.mean(rho_vals)
            p1 = t_min * wall_dir + rho_avg * wall_normal
            p2 = t_max * wall_dir + rho_avg * wall_normal
            fit = {
                'p1': p1,
                'p2': p2,
                'dir': wall_dir,
                'length': float(t_max - t_min),
                'rmse': 0.0,
                'line': (0.0, 0.0, 0.0),
                'mean': 0.5 * (p1 + p2)
            }
            walls.append({
                'fit': fit,
                'points': np.array([p1, p2]),
                'is_wall': True
            })
    return walls


# =========================================================================
# ===== 处理单个文件的核心函数 =====
# =========================================================================

def process_single_file(input_path: str) -> Tuple[np.ndarray, List[np.ndarray], List[dict]]:
    """
    处理单个雷达文件，返回原始点云、RDP分割段和检测到的墙体

    返回:
        xy: 原始点云 (N, 2)
        split_segments_points: RDP分割后的线段列表
        fitted_segments: 检测到的墙体列表
    """
    # 读取文件
    with open(input_path, "r") as f:
        data = json.load(f)
    laserscan = data["laserscan"] if "laserscan" in data else data

    # LaserScan → XY
    xy, _ = _laserscan_to_xy(laserscan, max_range=MAX_RANGE)

    # XY 空间过滤
    mask = (xy[:,0] >= -X_LIMIT) & (xy[:,0] <= X_LIMIT) & \
           (xy[:,1] >= -Y_LIMIT) & (xy[:,1] <= Y_LIMIT)
    xy = xy[mask]

    # 清理无效点
    xy = xy[np.isfinite(xy).all(axis=1)]

    # Jump分段 + RDP分割
    segments_points = _segment_by_jump(xy, JUMP_DIST_THRESH, MIN_SEGMENT_POINTS)
    split_segments_points = []
    for seg in segments_points:
        split_segments_points.extend(
            _split_by_rdp(seg, eps=RDP_EPSILON, min_points=MIN_SPLIT_POINTS)
        )

    # PCA拟合
    fitted_segments = []
    for seg_pts in split_segments_points:
        fit = _fit_line_pca(seg_pts)
        if fit is None:
            continue
        if fit['rmse'] > WALL_RMSE_THRESH:
            continue
        if fit['length'] < MIN_SEGMENT_LENGTH:
            continue
        fitted_segments.append({'fit': fit, 'points': seg_pts, 'is_wall': False})

    # 参数空间DBSCAN墙体聚类 + 1D区间合并
    if len(fitted_segments) > 0:
        features, weights, labels = _cluster_segments_in_param_space(
            fitted_segments,
            eps=PARAM_DBSCAN_EPS,
            min_samples=PARAM_DBSCAN_MIN_SAMPLES,
            alpha=ALPHA_SCALE
        )

        merged_walls = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_mask = (labels == cluster_id)
            cluster_segments = [fitted_segments[i] for i in np.where(cluster_mask)[0]]
            if len(cluster_segments) < MIN_SEGMENTS_PER_WALL:
                continue
            walls = _merge_segments_in_cluster(cluster_segments, gap_thresh=GAP_THRESH)
            for wall in walls:
                if wall['fit']['length'] >= MIN_WALL_LENGTH:
                    merged_walls.append(wall)

        fitted_segments = merged_walls

    return xy, split_segments_points, fitted_segments


def plot_wall_frame(xy: np.ndarray, split_segments_points: List[np.ndarray],
                    fitted_segments: List[dict], frame_idx: int) -> np.ndarray:
    """
    生成单帧1x2可视化图像（图3+图6）

    返回:
        image: RGB图像数组 (H, W, 3)
    """
    fig, (ax3, ax6) = plt.subplots(1, 2, figsize=(20, 10), dpi=100)

    # ===== 左侧：图3 - RDP分割后的直线段（按段着色） =====
    colors_split = plt.cm.tab20b(np.linspace(0, 1, max(1, len(split_segments_points))))
    for k, seg in enumerate(split_segments_points):
        col = colors_split[k % len(colors_split)]
        ax3.plot(seg[:, 0], seg[:, 1], '.', color=col, markersize=3, alpha=0.9)

    ax3.set_title(f"Frame {frame_idx:04d} - RDP segments (N={len(split_segments_points)}, ε={RDP_EPSILON}m)",
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.axis('equal')
    ax3.grid(True, linestyle=':', alpha=0.4)
    ax3.set_xlim(-8, 8)
    ax3.set_ylim(-8, 8)

    # ===== 右侧：图6 - 最终墙体（红色加粗） =====
    # 绘制原始点
    if len(xy) > 0:
        ax6.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=2, alpha=0.3)

    # 绘制墙体
    for seg in fitted_segments:
        p1, p2 = seg['fit']['p1'], seg['fit']['p2']
        ax6.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='tab:red', linewidth=3.5, alpha=0.95)

    # 绘制原点（自车坐标系）
    ax6.plot(0, 0, 'ko', markersize=10, markeredgewidth=2,
            markerfacecolor='black', label='Robot Origin')

    ax6.set_title(f"Frame {frame_idx:04d} - Walls (N={len(fitted_segments)})",
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Y (m)")
    ax6.axis('equal')
    ax6.grid(True, linestyle=':', alpha=0.4)
    ax6.set_xlim(-8, 8)
    ax6.set_ylim(-8, 8)

    # 转换为图像数组
    plt.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return image


def save_wall_json(fitted_segments: List[dict], input_path: str, output_dir: str):
    """保存墙体数据为JSON"""
    wall_data = {
        'walls': [],
        'metadata': {
            'input_file': str(input_path),
            'num_walls': len(fitted_segments),
            'x_limit': X_LIMIT,
            'y_limit': Y_LIMIT,
            'params': {
                'rdp_epsilon': RDP_EPSILON,
                'wall_rmse_thresh': WALL_RMSE_THRESH,
                'min_wall_length': MIN_WALL_LENGTH,
                'gap_thresh': GAP_THRESH
            }
        }
    }

    for i, seg in enumerate(fitted_segments):
        fit = seg['fit']
        wall_data['walls'].append({
            'id': i,
            'p1': fit['p1'].tolist(),
            'p2': fit['p2'].tolist(),
            'length': fit['length'],
            'direction': fit['dir'].tolist(),
            'rmse': fit['rmse']
        })

    # 保存JSON
    basename = os.path.splitext(os.path.basename(input_path))[0]
    out_json_path = os.path.join(output_dir, f"{basename}_walls.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(wall_data, f, indent=2, ensure_ascii=False)


# =========================================================================
# ===== 主函数：批量处理 =====
# =========================================================================

def main():
    """批量处理所有雷达数据，生成gif和wall文件"""
    print("="*70)
    print(" 批量墙体检测与GIF生成")
    print("="*70)

    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WALL_OUT_DIR, exist_ok=True)

    # 获取所有输入文件
    input_files = sorted(Path(INPUT_DIR).glob("laserscan_*.json"))
    print(f"\n找到 {len(input_files)} 个雷达数据文件")

    if len(input_files) == 0:
        print("错误：未找到任何输入文件！")
        return

    # 批量处理
    frames = []
    total_start = time.time()

    for idx, input_path in enumerate(tqdm(input_files, desc="处理文件")):
        try:
            # 处理单个文件
            xy, split_segments_points, fitted_segments = process_single_file(str(input_path))

            # 生成可视化图像
            image = plot_wall_frame(xy, split_segments_points, fitted_segments, idx)
            frames.append(image)

            # 保存墙体JSON
            save_wall_json(fitted_segments, str(input_path), WALL_OUT_DIR)

        except Exception as e:
            print(f"\n警告：处理文件 {input_path.name} 时出错: {e}")
            continue

    total_time = time.time() - total_start

    # 生成GIF
    if len(frames) > 0:
        print(f"\n生成GIF动画...")
        imageio.mimsave(GIF_OUTPUT, frames, fps=GIF_FPS, loop=0)
        print(f"✅ GIF已保存: {GIF_OUTPUT}")
    else:
        print("错误：没有生成任何帧！")

    # 统计信息
    print("\n" + "="*70)
    print(" 处理完成统计")
    print("="*70)
    print(f"总文件数: {len(input_files)}")
    print(f"成功处理: {len(frames)}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均速度: {len(frames)/total_time:.2f} FPS")
    print(f"\n输出位置:")
    print(f"  - GIF动画: {GIF_OUTPUT}")
    print(f"  - 墙体JSON: {WALL_OUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
