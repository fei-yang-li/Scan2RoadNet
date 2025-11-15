# shiyong scan julei




import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import time
import json



# =========================
# 全局可配置参数（重要变量）
# =========================
# 输入文件（支持 LaserScan JSON 或 NPY）
INPUT_PATH = os.path.join(os.path.dirname(__file__), "extracted_lidar_data_code", "extracted_lidar_data", "laserscan_json", "laserscan_000136.json")
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
OUT_DIR = os.path.join(os.path.dirname(__file__), "extracted_lidar_data_code", "extracted_lidar_data", "RDP_DBSCAN_scan")
WALL_OUT_DIR = os.path.join(os.path.dirname(__file__), "extracted_lidar_data_code", "wall")

# ===== LaserScan(JSON) → 笛卡尔坐标 =====
def _laserscan_to_xy(laserscan: dict, max_range: float | None = None) -> tuple[np.ndarray, np.ndarray]:
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

# --- 基于扫描顺序的 1D 跳变分段 ---
def _segment_by_jump(points_xy_ordered: np.ndarray,
                     jump_thresh: float,
                     min_points: int) -> list[np.ndarray]:
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

# --- PCA 直线拟合，返回法线式 ax+by+c=0 与方向、端点、RMSE ---
def _fit_line_pca(points: np.ndarray) -> dict | None:
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

# ===== RDP (Ramer-Douglas-Peucker) 算法实现 =====
def _point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    计算点 p 到线段 ab 的垂直距离（标量版本，保留用于兼容）

    参数:
        p: 点坐标 [x, y]
        a: 线段起点 [x, y]
        b: 线段终点 [x, y]

    返回:
        距离值（米）
    """
    ab = b - a
    # 如果 a 和 b 重合，返回点到点的距离
    if np.allclose(ab, 0):
        return float(np.linalg.norm(p - a))

    # 计算投影参数 t
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)

    # 投影点
    proj = a + t * ab

    # 返回点到投影点的距离
    return float(np.linalg.norm(p - proj))


def _points_line_distance_vectorized(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算多个点到线段 ab 的垂直距离（向量化版本）

    参数:
        points: N×2 点数组
        a: 线段起点 [x, y]
        b: 线段终点 [x, y]

    返回:
        距离数组（N,）
    """
    ab = b - a

    # 如果 a 和 b 重合，返回点到点的距离
    if np.allclose(ab, 0):
        return np.linalg.norm(points - a, axis=1)

    # 向量化计算投影参数 t
    # t = (p - a) · ab / ||ab||^2
    ap = points - a
    ab_squared = np.dot(ab, ab)
    t = np.dot(ap, ab) / ab_squared
    t = np.clip(t, 0.0, 1.0)

    # 投影点: proj = a + t * ab
    # 利用广播: proj[i] = a + t[i] * ab
    proj = a + t[:, np.newaxis] * ab

    # 返回点到投影点的距离
    return np.linalg.norm(points - proj, axis=1)


def _rdp_recursive(points: np.ndarray,
                   start: int,
                   end: int,
                   eps: float,
                   keep_idx: set) -> None:
    """
    RDP 算法的递归实现（向量化优化版本）

    参数:
        points: N×2 点数组
        start: 起始索引
        end: 结束索引
        eps: 最大允许偏差（米）
        keep_idx: 保留点的索引集合（会被修改）
    """
    if end <= start + 1:
        return

    a = points[start]
    b = points[end]

    # 向量化计算所有中间点到线段的距离
    if end > start + 1:
        middle_points = points[start + 1:end]
        distances = _points_line_distance_vectorized(middle_points, a, b)

        # 找到最大距离及其索引
        max_idx_relative = np.argmax(distances)
        max_dist = distances[max_idx_relative]
        max_idx = start + 1 + max_idx_relative
    else:
        max_dist = -1.0
        max_idx = -1

    # 如果最大偏差超过阈值，在该点处切分
    if max_dist > eps:
        keep_idx.add(max_idx)
        _rdp_recursive(points, start, max_idx, eps, keep_idx)
        _rdp_recursive(points, max_idx, end, eps, keep_idx)


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """
    Ramer-Douglas-Peucker 算法主函数

    在保证最大偏差不超过 eps 的前提下，用尽量少的关键点简化折线

    参数:
        points: N×2 的 numpy 数组，表示有序折线
        eps: 最大允许偏差（米）

    返回:
        关键点的索引数组（从小到大排序）
    """
    n = len(points)
    if n <= 2:
        return np.arange(n)

    # 始终保留首末点
    keep_idx = {0, n - 1}
    _rdp_recursive(points, 0, n - 1, eps, keep_idx)

    # 转为排序数组
    idx = np.array(sorted(list(keep_idx)), dtype=int)
    return idx


def _split_by_rdp(points_ordered: np.ndarray,
                  eps: float,
                  min_points: int) -> list[np.ndarray]:
    """
    使用 RDP 算法对一段有序点进行分割

    参数:
        points_ordered: N×2 有序点数组
        eps: RDP 偏差阈值（米）
        min_points: 每段最少点数

    返回:
        线段列表，每个线段是一个点数组
    """
    if len(points_ordered) < 2:
        return []

    # 运行 RDP 获取关键点索引
    key_idx = _rdp(points_ordered, eps)

    # 将相邻关键点之间的点组成线段
    segments = []
    for i in range(len(key_idx) - 1):
        s = key_idx[i]
        e = key_idx[i + 1]
        seg_points = points_ordered[s:e+1]

        # 过滤掉点数太少的段
        if len(seg_points) >= min_points:
            segments.append(seg_points)

    return segments

# ===== 参数空间DBSCAN墙体聚类方法 =====
def _segment_to_hessian(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    """
    将线段转换为Hessian法线形式 (θ, ρ)

    参数:
        p1, p2: 线段的两个端点

    返回:
        (theta, rho):
            theta ∈ [0, π) - 法向量的角度
            rho - 原点到直线的有符号距离
    """
    # 中点
    m = 0.5 * (p1 + p2)

    # 方向向量
    d = p2 - p1
    dx, dy = d[0], d[1]

    # 计算方向角 θ ∈ [-π, π]
    theta = np.arctan2(dy, dx)

    # 归一化到 [0, π)（因为直线无方向性）
    if theta < 0:
        theta += np.pi

    # 法向量（垂直于方向向量）
    n = np.array([-np.sin(theta), np.cos(theta)])

    # 原点到直线的有符号距离
    rho = np.dot(n, m)

    return float(theta), float(rho)


def _merge_intervals_1d(intervals: list[tuple[float, float]],
                        gap_thresh: float) -> list[tuple[float, float]]:
    """
    合并1D区间，容忍小gap

    参数:
        intervals: [(t_min, t_max), ...] 区间列表
        gap_thresh: 允许的最大gap

    返回:
        合并后的区间列表
    """
    if len(intervals) == 0:
        return []

    # 按 t_min 排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    current_min, current_max = sorted_intervals[0]

    for t_min, t_max in sorted_intervals[1:]:
        # 如果下一个区间的起点距离当前区间终点很近
        if t_min - current_max <= gap_thresh:
            # 合并（扩展当前区间）
            current_max = max(current_max, t_max)
        else:
            # 保存当前区间，开始新区间
            merged.append((current_min, current_max))
            current_min, current_max = t_min, t_max

    # 添加最后一个区间
    merged.append((current_min, current_max))

    return merged


def _cluster_segments_in_param_space(fitted_segments: list[dict],
                                     eps: float,
                                     min_samples: int,
                                     alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在 (θ, ρ) 参数空间对线段进行DBSCAN聚类（优化版）

    参数:
        fitted_segments: 线段列表，每个元素是 {'fit': fit_dict, 'points': points}
        eps: DBSCAN邻域半径（主要控制ρ的距离，单位：米）
        min_samples: DBSCAN最小样本数
        alpha: θ的缩放系数（缩小角度）

    返回:
        (features, weights, labels):
            features: N×2 数组 [[theta/alpha, rho], ...]
            weights: N 数组，线段长度（可用于加权平均）
            labels: DBSCAN聚类标签
    """
    if len(fitted_segments) == 0:
        return np.array([]), np.array([]), np.array([])

    # 预分配数组以避免动态列表增长
    n_segs = len(fitted_segments)
    features = np.zeros((n_segs, 2), dtype=np.float64)
    weights = np.zeros(n_segs, dtype=np.float64)

    # 向量化处理所有线段
    for i, seg_dict in enumerate(fitted_segments):
        fit = seg_dict['fit']
        p1 = fit['p1']
        p2 = fit['p2']

        # 转换到 (θ, ρ) 空间
        theta, rho = _segment_to_hessian(p1, p2)

        # 缩小theta以让eps主要控制rho（距离）
        features[i, 0] = theta / alpha
        features[i, 1] = rho
        weights[i] = fit['length']

    # 在参数空间进行DBSCAN聚类
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = db.labels_

    return features, weights, labels


def _merge_segments_in_cluster(cluster_segments: list[dict],
                                gap_thresh: float) -> list[dict]:
    """
    对同一簇内的线段，沿墙体方向合并成完整墙体

    参数:
        cluster_segments: 属于同一簇的线段列表
        gap_thresh: 允许的gap阈值

    返回:
        合并后的墙体列表，每个元素是 {'fit': fit_dict, 'points': points}
    """
    if len(cluster_segments) == 0:
        return []

    # 1. 计算簇的代表方向（加权平均）
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

    # 2. 将每个线段投影到墙体方向，得到1D区间
    intervals = []
    all_points = []

    for seg_dict in cluster_segments:
        fit = seg_dict['fit']
        points = seg_dict['points']
        p1, p2 = fit['p1'], fit['p2']

        # 投影到墙体方向
        t1 = np.dot(p1, wall_dir)
        t2 = np.dot(p2, wall_dir)

        t_min, t_max = min(t1, t2), max(t1, t2)
        intervals.append((t_min, t_max))
        all_points.append(points)

    # 3. 合并1D区间
    merged_intervals = _merge_intervals_1d(intervals, gap_thresh)

    # 4. 为每个合并后的区间生成墙体线段
    walls = []
    all_points_combined = np.vstack(all_points) if len(all_points) > 0 else np.empty((0, 2))

    for t_min, t_max in merged_intervals:
        # 计算墙体方向的垂直方向（法向量）
        wall_normal = np.array([-wall_dir[1], wall_dir[0]])

        # 找到所有在这个t区间内的点
        if len(all_points_combined) > 0:
            t_vals = np.dot(all_points_combined, wall_dir)
            mask = (t_vals >= t_min - 0.01) & (t_vals <= t_max + 0.01)
            interval_points = all_points_combined[mask]
        else:
            interval_points = np.empty((0, 2))

        # 如果有足够的点，重新拟合
        if len(interval_points) >= 2:
            fit = _fit_line_pca(interval_points)
            if fit is not None:
                walls.append({
                    'fit': fit,
                    'points': interval_points,
                    'is_wall': True  # 标记为墙体
                })
        else:
            # 点太少，直接用端点生成线段
            # 计算这个区间的中心在法向的平均位置
            rho_avg = 0.0
            if len(all_points_combined) > 0:
                rho_vals = np.dot(all_points_combined, wall_normal)
                rho_avg = np.mean(rho_vals)

            # 生成端点
            p1 = t_min * wall_dir + rho_avg * wall_normal
            p2 = t_max * wall_dir + rho_avg * wall_normal

            # 创建虚拟fit
            fit = {
                'p1': p1,
                'p2': p2,
                'dir': wall_dir,
                'length': float(t_max - t_min),
                'rmse': 0.0,
                'line': (0.0, 0.0, 0.0),  # 占位
                'mean': 0.5 * (p1 + p2)
            }
            walls.append({
                'fit': fit,
                'points': np.array([p1, p2]),
                'is_wall': True
            })

    return walls


# 读取单帧（支持 LaserScan JSON 或 NPY）
t_start_all = time.perf_counter()
input_path = INPUT_PATH
is_json = str(input_path).lower().endswith(".json")

if is_json:
    with open(input_path, "r") as f:
        data = json.load(f)
    laserscan = data["laserscan"] if "laserscan" in data else data
    # LaserScan → XY
    xy_from_scan, valid_idx = _laserscan_to_xy(laserscan, max_range=MAX_RANGE)
    # 与 NPY 分支对齐的时间锚点
    z_points = xy_from_scan.copy()  # 用于后续"XY过滤保留比例"的统计分母
    xy = xy_from_scan
else:
    # 兼容 NPY 三维点: 读取并做 Z 过滤
    points = np.load(input_path)
    print(f"原始点数: {points.shape[0]}")
    z_all = points[:, 2]
    z_percentiles = np.percentile(z_all, [1, 25, 50, 75, 99])
    print(f"Z范围: [{np.min(z_all):.3f}, {np.max(z_all):.3f}], 分位(1,25,50,75,99): {np.round(z_percentiles, 3)}")
    z_min, z_max = Z_MIN, Z_MAX
    z_points = points[(points[:, 2] >= Z_MIN) & (points[:, 2] <= Z_MAX)]
    print(f"z_points后点数: {z_points.shape[0]}")
    print(f"Z过滤保留比例: {z_points.shape[0] / max(points.shape[0], 1):.1%} (阈值区间: [{z_min}, {z_max}])")
    xy = z_points[:, :2]
t_after_load = time.perf_counter()
t_after_zfilter = time.perf_counter()
# 再按 X/Y 范围过滤
mask = (xy[:,0] >= -X_LIMIT) & (xy[:,0] <= X_LIMIT) & \
       (xy[:,1] >= -Y_LIMIT) & (xy[:,1] <= Y_LIMIT)
xy = xy[mask]
t_after_xyfilter = time.perf_counter()

# ===== 1. 清理无效点 =====
xy = xy[np.isfinite(xy).all(axis=1)]
t_after_clean = time.perf_counter()

# ===== 2. Jump分段 + RDP分割 =====
t_jump_start = time.perf_counter()
segments_points = _segment_by_jump(xy, JUMP_DIST_THRESH, MIN_SEGMENT_POINTS)
t_jump_end = time.perf_counter()

# 使用 RDP 算法对每个大段做形状内部分段
t_rdp_start = time.perf_counter()
split_segments_points = []
for seg in segments_points:
    split_segments_points.extend(
        _split_by_rdp(seg, eps=RDP_EPSILON, min_points=MIN_SPLIT_POINTS)
    )
t_rdp_end = time.perf_counter()

t_pca_fit_start = time.perf_counter()
fitted_segments = []  # {'fit':fit_dict, 'points':seg_points, 'is_wall':bool}
n_filtered_rmse = 0
n_filtered_length = 0
for idx, seg_pts in enumerate(split_segments_points):
    fit = _fit_line_pca(seg_pts)
    if fit is None:
        continue
    # 基本过滤：RMSE和长度
    if fit['rmse'] > WALL_RMSE_THRESH:
        n_filtered_rmse += 1
        continue
    if fit['length'] < MIN_SEGMENT_LENGTH:
        n_filtered_length += 1
        continue
    fitted_segments.append({'fit': fit, 'points': seg_pts, 'is_wall': False})

t_pca_fit_end = time.perf_counter()

# 保存初步拟合的线段供可视化使用
initial_fitted_segments = fitted_segments.copy()

# ===== 3. 参数空间DBSCAN墙体聚类 + 1D区间合并 =====
t_param_dbscan_start = time.perf_counter()
if len(fitted_segments) > 0:
    # 在(θ, ρ)空间进行DBSCAN聚类
    features, weights, labels = _cluster_segments_in_param_space(
        fitted_segments,
        eps=PARAM_DBSCAN_EPS,
        min_samples=PARAM_DBSCAN_MIN_SAMPLES,
        alpha=ALPHA_SCALE
    )
    t_param_dbscan_end = time.perf_counter()

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(labels == -1)

    # 对每个簇进行1D区间合并
    t_merge_start = time.perf_counter()
    merged_walls = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        # 获取该簇的所有线段
        cluster_mask = (labels == cluster_id)
        cluster_segments = [fitted_segments[i] for i in np.where(cluster_mask)[0]]

        # 检查簇的大小
        if len(cluster_segments) < MIN_SEGMENTS_PER_WALL:
            continue

        # 沿墙体方向合并1D区间
        walls = _merge_segments_in_cluster(cluster_segments, gap_thresh=GAP_THRESH)

        # 过滤太短的墙
        for wall in walls:
            if wall['fit']['length'] >= MIN_WALL_LENGTH:
                merged_walls.append(wall)

    t_merge_end = time.perf_counter()
    fitted_segments = merged_walls
else:
    print("没有找到符合条件的线段")
    features, weights, labels = np.array([]), np.array([]), np.array([])
    num_clusters, num_noise = 0, 0
    t_param_dbscan_end = t_param_dbscan_start
    t_merge_start = t_param_dbscan_end
    t_merge_end = t_merge_start
t_after_cluster_proc = time.perf_counter()

# ===============================================
# ===== 4. 绘制结果（3x2管线总览） =====
# ===============================================

t_plot_start = time.perf_counter()
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

# ===== Ax1: 原始点 =====
if len(xy) > 0:
    ax1.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=3, alpha=0.6)
ax1.set_title(f"Raw points (N={len(xy)})", fontsize=12, fontweight='bold')
ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.axis('equal'); ax1.grid(True, linestyle=':', alpha=0.4)

# ===== Ax2: Jump-distance 分段（按段着色） =====
colors_jump = plt.cm.tab20(np.linspace(0, 1, max(1, len(segments_points))))
for k, seg in enumerate(segments_points):
    col = colors_jump[k % len(colors_jump)]
    ax2.plot(seg[:, 0], seg[:, 1], '.', color=col, markersize=3, alpha=0.9)
ax2.set_title(f"Jump-distance segments (N={len(segments_points)}, Δ={JUMP_DIST_THRESH}m)", fontsize=12, fontweight='bold')
ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.axis('equal'); ax2.grid(True, linestyle=':', alpha=0.4)

# ===== Ax3: RDP 分割后的直线段（按段着色） =====
colors_split = plt.cm.tab20b(np.linspace(0, 1, max(1, len(split_segments_points))))
for k, seg in enumerate(split_segments_points):
    col = colors_split[k % len(colors_split)]
    ax3.plot(seg[:, 0], seg[:, 1], '.', color=col, markersize=3, alpha=0.9)
ax3.set_title(f"RDP segments (N={len(split_segments_points)}, ε={RDP_EPSILON}m)", fontsize=12, fontweight='bold')
ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.axis('equal'); ax3.grid(True, linestyle=':', alpha=0.4)

# ===== Ax4: 参数空间(θ, ρ)聚类可视化 =====
if 'features' in locals() and len(features) > 0:
    # 绘制参数空间中的点（所有初步拟合的线段）
    noise_mask = (labels == -1)
    cluster_mask = (labels != -1)

    # 噪声点（灰色×）
    if np.any(noise_mask):
        ax4.scatter(features[noise_mask, 0], features[noise_mask, 1],
                   c='gray', s=80, alpha=0.6, marker='x', linewidths=2,
                   label=f'Noise ({np.sum(noise_mask)})')

    # 聚类点（按簇着色，大圆点）
    if np.any(cluster_mask):
        unique_labels = sorted(set(labels[cluster_mask]))
        for label in unique_labels:
            mask = (labels == label)
            n_segs = np.sum(mask)
            color = plt.cm.tab10(label % 10)  # 使用cluster_id本身映射颜色，与Ax5对齐
            ax4.scatter(features[mask, 0], features[mask, 1],
                       c=[color], s=100, alpha=0.85,
                       edgecolors='black', linewidths=1.5,
                       label=f'Cluster {label} ({n_segs} segs)')

    ax4.set_title(f"Param-space (θ, ρ) DBSCAN clustering\n"
                  f"Total segments: {len(features)}, Clusters: {num_clusters}, Noise: {num_noise}",
                  fontsize=11, fontweight='bold')
    ax4.set_xlabel(f"θ / {ALPHA_SCALE:.2f} (scaled)", fontsize=10)
    ax4.set_ylabel("ρ (m)", fontsize=10)
    ax4.legend(fontsize=9, loc='best', framealpha=0.9)
    ax4.grid(True, linestyle=':', alpha=0.5)
else:
    ax4.text(0.5, 0.5, 'No segments for clustering',
            ha='center', va='center', transform=ax4.transAxes, fontsize=14, color='red')
    ax4.set_title("Param-space (θ, ρ) DBSCAN", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.4)

# ===== Ax5: 拟合线段叠加原始点（聚类前）=====
if len(xy) > 0:
    ax5.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=2, alpha=0.3)

# 在Ax5中按簇绘制初步拟合的线段（按DBSCAN簇着色）
if 'initial_fitted_segments' in locals() and 'labels' in locals() and len(initial_fitted_segments) > 0:
    for i, seg_dict in enumerate(initial_fitted_segments):
        fit = seg_dict['fit']
        p1, p2 = fit['p1'], fit['p2']

        if i < len(labels):
            if labels[i] == -1:
                color = 'gray'
                lw = 1.5
                alpha = 0.5
            else:
                cluster_id = labels[i]
                color = plt.cm.tab10(cluster_id % 10)
                lw = 2.5
                alpha = 0.85

            ax5.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, linewidth=lw, alpha=alpha)

ax5.set_title(f"Fitted segments colored by DBSCAN cluster\n"
              f"Total: {len(initial_fitted_segments) if 'initial_fitted_segments' in locals() else 0} segments",
              fontsize=11, fontweight='bold')
ax5.set_xlabel("X (m)"); ax5.set_ylabel("Y (m)"); ax5.axis('equal'); ax5.grid(True, linestyle=':', alpha=0.4)

# ===== Ax6: 最终墙体（红色加粗） =====
if len(xy) > 0:
    ax6.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=2, alpha=0.3)

for seg in fitted_segments:
    p1, p2 = seg['fit']['p1'], seg['fit']['p2']
    ax6.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color='tab:red', linewidth=3.5, alpha=0.95)

ax6.set_title(f"Final walls (N={len(fitted_segments)}, gap≤{GAP_THRESH}m, L≥{MIN_WALL_LENGTH}m)",
              fontsize=12, fontweight='bold')
ax6.set_xlabel("X (m)"); ax6.set_ylabel("Y (m)"); ax6.axis('equal'); ax6.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
out_dir = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)
out_multi = os.path.splitext(os.path.basename(input_path))[0] + "_RDP_DBSCAN_pipeline.png"
plt.savefig(os.path.join(out_dir, out_multi), dpi=300, bbox_inches='tight')
plt.close(fig)
t_after_plot = time.perf_counter()

print(f"\n{'='*60}")
print(f"✅ RDP+参数空间DBSCAN流程完成")
print(f"   最终墙体数: {len(fitted_segments)}")
print(f"{'='*60}\n")

# ====== 详细的处理耗时统计 ======
total_time = t_after_plot - t_start_all

print("┌" + "─" * 58 + "┐")
print("│" + " " * 18 + "处理耗时统计（秒）" + " " * 18 + "│")
print("├" + "─" * 58 + "┤")

# 数据加载与预处理
t_load = t_after_load - t_start_all
t_zfilter = t_after_zfilter - t_after_load
t_xyfilter = t_after_xyfilter - t_after_zfilter
t_clean = t_after_clean - t_after_xyfilter
t_preprocess = t_clean + t_zfilter + t_xyfilter + t_load

print(f"│ 【数据加载与预处理】                   {t_preprocess:>8.4f}s │")
print(f"│   ├─ 加载点云                         {t_load:>8.4f}s │")
print(f"│   ├─ Z高度过滤                        {t_zfilter:>8.4f}s │")
print(f"│   ├─ XY范围过滤                       {t_xyfilter:>8.4f}s │")
print(f"│   └─ 清理无效点                       {t_clean:>8.4f}s │")
print("├" + "─" * 58 + "┤")

# 分割阶段
t_jump = t_jump_end - t_jump_start
t_rdp = t_rdp_end - t_rdp_start
t_segmentation = t_jump + t_rdp

print(f"│ 【分割阶段】                           {t_segmentation:>8.4f}s │")
print(f"│   ├─ Jump-distance分段                {t_jump:>8.4f}s │")
print(f"│   └─ RDP几何简化                      {t_rdp:>8.4f}s │")
print("├" + "─" * 58 + "┤")

# 拟合与聚类
t_pca = t_pca_fit_end - t_pca_fit_start
t_param_cluster = t_param_dbscan_end - t_param_dbscan_start
t_merge = t_merge_end - t_merge_start
t_clustering = t_pca + t_param_cluster + t_merge

print(f"│ 【拟合与聚类合并】                     {t_clustering:>8.4f}s │")
print(f"│   ├─ PCA直线拟合                      {t_pca:>8.4f}s │")
print(f"│   ├─ 参数空间DBSCAN                   {t_param_cluster:>8.4f}s │")
print(f"│   └─ 1D区间合并                       {t_merge:>8.4f}s │")
print("├" + "─" * 58 + "┤")

# 可视化
t_plot = t_after_plot - t_plot_start
print(f"│ 【可视化与保存】                       {t_plot:>8.4f}s │")
print("├" + "─" * 58 + "┤")

# 总耗时
print(f"│ 【总耗时】                             {total_time:>8.4f}s │")
print("└" + "─" * 58 + "┘")

# 吞吐量统计
if total_time > 0:
    fps = 1.0 / total_time
    points_per_sec = len(xy) / total_time
    print(f"\n吞吐量: {fps:.2f} FPS | {points_per_sec:.0f} 点/秒")

# ===== 保存墙体数据供后续道路中心线和路口识别使用 =====
wall_data = {
    'walls': [],
    'metadata': {
        'input_file': str(input_path),
        'num_walls': len(fitted_segments),
        'num_points': len(xy),
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

# 保存墙体数据为JSON
os.makedirs(WALL_OUT_DIR, exist_ok=True)
out_json = os.path.splitext(os.path.basename(input_path))[0] + "_walls.json"
out_json_path = os.path.join(WALL_OUT_DIR, out_json)
with open(out_json_path, 'w', encoding='utf-8') as f:
    json.dump(wall_data, f, indent=2, ensure_ascii=False)

print(f"✅ 墙体数据已保存: {out_json_path}")
