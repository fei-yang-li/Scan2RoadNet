"""
道路中心线生成器
读取墙体数据，根据左右墙体配对生成道路中心线
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass


# =========================
# 全局配置参数
# =========================
# 墙体数据输入路径
WALL_JSON_PATH = os.path.join(os.path.dirname(__file__), "extracted_lidar_data_code", "wall", "laserscan_000136_walls.json")

# 输出目录
OUT_DIR = os.path.join(os.path.dirname(__file__), "extracted_lidar_data_code", "road_network")

# ===== 道路配对参数 =====
MAX_ROAD_WIDTH = 3.0        # 最大道路宽度（米）
MIN_ROAD_WIDTH = 0.8        # 最小道路宽度（米）
PARALLEL_ANGLE_THRESH = 10  # 平行判定角度阈值（度）
MIN_OVERLAP_RATIO = 0.3     # 最小重叠比例（墙体投影重叠）


@dataclass
class Wall:
    """墙体数据结构"""
    id: int
    p1: np.ndarray  # 起点
    p2: np.ndarray  # 终点
    length: float
    direction: np.ndarray  # 方向向量
    rmse: float

    @property
    def center(self) -> np.ndarray:
        """墙体中心点"""
        return 0.5 * (self.p1 + self.p2)

    @property
    def angle(self) -> float:
        """墙体角度（弧度）"""
        return np.arctan2(self.direction[1], self.direction[0])

    @property
    def angle_deg(self) -> float:
        """墙体角度（度）"""
        return np.degrees(self.angle)


@dataclass
class Centerline:
    """道路中心线数据结构"""
    id: int
    p1: np.ndarray  # 起点
    p2: np.ndarray  # 终点
    width: float    # 道路宽度
    wall_pair: Tuple[int, int]  # 配对的墙体ID

    @property
    def center(self) -> np.ndarray:
        """中心线中点"""
        return 0.5 * (self.p1 + self.p2)

    @property
    def length(self) -> float:
        """中心线长度"""
        return np.linalg.norm(self.p2 - self.p1)

    @property
    def direction(self) -> np.ndarray:
        """方向向量"""
        vec = self.p2 - self.p1
        return vec / (np.linalg.norm(vec) + 1e-12)


def load_walls(json_path: str) -> List[Wall]:
    """
    从JSON文件加载墙体数据

    参数:
        json_path: 墙体JSON文件路径

    返回:
        墙体列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    walls = []
    for w in data['walls']:
        walls.append(Wall(
            id=w['id'],
            p1=np.array(w['p1']),
            p2=np.array(w['p2']),
            length=w['length'],
            direction=np.array(w['direction']),
            rmse=w['rmse']
        ))

    print(f"✅ 加载了 {len(walls)} 面墙体")
    return walls


def are_parallel(wall1: Wall, wall2: Wall, angle_thresh_deg: float) -> bool:
    """
    判断两面墙是否平行

    参数:
        wall1, wall2: 墙体
        angle_thresh_deg: 角度阈值（度）

    返回:
        是否平行
    """
    # 计算方向向量夹角
    dot = np.abs(np.dot(wall1.direction, wall2.direction))
    angle_diff = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    # 平行: 角度差接近0度或180度
    return angle_diff < angle_thresh_deg or angle_diff > (180 - angle_thresh_deg)


def compute_wall_distance(wall1: Wall, wall2: Wall) -> float:
    """
    计算两面平行墙体之间的平均距离

    参数:
        wall1, wall2: 墙体

    返回:
        平均距离（米）
    """
    # 墙1的法向量
    normal1 = np.array([-wall1.direction[1], wall1.direction[0]])

    # 墙2的两个端点到墙1的有符号距离
    d1 = np.dot(wall2.p1 - wall1.p1, normal1)
    d2 = np.dot(wall2.p2 - wall1.p1, normal1)

    # 返回平均距离的绝对值
    return np.abs(0.5 * (d1 + d2))


def compute_projection_overlap(wall1: Wall, wall2: Wall) -> float:
    """
    计算两面墙体在它们方向上的投影重叠比例

    参数:
        wall1, wall2: 墙体

    返回:
        重叠比例 [0, 1]
    """
    # 使用墙1的方向作为投影轴
    dir_vec = wall1.direction

    # 投影墙1的端点
    t1_start = np.dot(wall1.p1, dir_vec)
    t1_end = np.dot(wall1.p2, dir_vec)
    if t1_start > t1_end:
        t1_start, t1_end = t1_end, t1_start

    # 投影墙2的端点
    t2_start = np.dot(wall2.p1, dir_vec)
    t2_end = np.dot(wall2.p2, dir_vec)
    if t2_start > t2_end:
        t2_start, t2_end = t2_end, t2_start

    # 计算重叠区间
    overlap_start = max(t1_start, t2_start)
    overlap_end = min(t1_end, t2_end)
    overlap_length = max(0, overlap_end - overlap_start)

    # 计算重叠比例（相对于较短的墙）
    min_length = min(t1_end - t1_start, t2_end - t2_start)
    if min_length < 1e-6:
        return 0.0

    return overlap_length / min_length


def find_wall_pairs(walls: List[Wall],
                    max_width: float,
                    min_width: float,
                    angle_thresh: float,
                    min_overlap: float) -> List[Tuple[Wall, Wall]]:
    """
    找到可以配对形成道路的墙体对

    参数:
        walls: 墙体列表
        max_width: 最大道路宽度
        min_width: 最小道路宽度
        angle_thresh: 平行角度阈值
        min_overlap: 最小重叠比例

    返回:
        墙体对列表 [(wall_i, wall_j), ...]
    """
    pairs = []
    used_walls = set()

    for i, wall1 in enumerate(walls):
        if wall1.id in used_walls:
            continue

        best_match = None
        best_score = -1

        for j, wall2 in enumerate(walls):
            if i >= j or wall2.id in used_walls:
                continue

            # 1. 检查是否平行
            if not are_parallel(wall1, wall2, angle_thresh):
                continue

            # 2. 检查距离是否在道路宽度范围内
            dist = compute_wall_distance(wall1, wall2)
            if dist < min_width or dist > max_width:
                continue

            # 3. 检查投影重叠度
            overlap = compute_projection_overlap(wall1, wall2)
            if overlap < min_overlap:
                continue

            # 4. 计算配对得分（重叠度越高越好，距离适中越好）
            # 理想距离设为1.5米
            ideal_width = 1.5
            width_score = 1.0 - abs(dist - ideal_width) / max_width
            score = overlap * 0.7 + width_score * 0.3

            if score > best_score:
                best_score = score
                best_match = (wall2, dist)

        # 如果找到了最佳配对
        if best_match is not None:
            wall2, dist = best_match
            pairs.append((wall1, wall2))
            used_walls.add(wall1.id)
            used_walls.add(wall2.id)
            print(f"  配对: Wall{wall1.id} ↔ Wall{wall2.id}, 宽度={dist:.2f}m, 得分={best_score:.2f}")

    print(f"✅ 找到 {len(pairs)} 对墙体")
    return pairs


def generate_centerline_from_pair(wall1: Wall, wall2: Wall, pair_id: int) -> Centerline:
    """
    根据墙体对生成道路中心线

    参数:
        wall1, wall2: 配对的墙体
        pair_id: 中心线ID

    返回:
        道路中心线
    """
    # 使用墙1的方向作为道路方向
    dir_vec = wall1.direction

    # 投影所有端点到方向轴
    t1 = np.dot(wall1.p1, dir_vec)
    t2 = np.dot(wall1.p2, dir_vec)
    t3 = np.dot(wall2.p1, dir_vec)
    t4 = np.dot(wall2.p2, dir_vec)

    # 找到重叠区间
    t_start = max(min(t1, t2), min(t3, t4))
    t_end = min(max(t1, t2), max(t3, t4))

    # 中心线的起点和终点在重叠区间内
    # 分别取两面墙在该位置的中点
    def get_point_at_t(wall: Wall, t: float) -> np.ndarray:
        """在投影位置t处获取墙上的点"""
        t1 = np.dot(wall.p1, dir_vec)
        t2 = np.dot(wall.p2, dir_vec)
        if abs(t2 - t1) < 1e-6:
            return wall.p1
        ratio = (t - t1) / (t2 - t1)
        return wall.p1 + ratio * (wall.p2 - wall.p1)

    # 起点: 两墙在t_start处的中点
    start1 = get_point_at_t(wall1, t_start)
    start2 = get_point_at_t(wall2, t_start)
    centerline_start = 0.5 * (start1 + start2)

    # 终点: 两墙在t_end处的中点
    end1 = get_point_at_t(wall1, t_end)
    end2 = get_point_at_t(wall2, t_end)
    centerline_end = 0.5 * (end1 + end2)

    # 计算道路宽度
    width = compute_wall_distance(wall1, wall2)

    return Centerline(
        id=pair_id,
        p1=centerline_start,
        p2=centerline_end,
        width=width,
        wall_pair=(wall1.id, wall2.id)
    )


def visualize_results(walls: List[Wall],
                     centerlines: List[Centerline],
                     out_path: str):
    """
    可视化墙体和道路中心线

    参数:
        walls: 墙体列表
        centerlines: 中心线列表
        out_path: 输出图片路径
    """
    fig, ax = plt.subplots(figsize=(14, 14))

    # 绘制墙体（实线，黑色）
    for wall in walls:
        ax.plot([wall.p1[0], wall.p2[0]],
               [wall.p1[1], wall.p2[1]],
               'k-', linewidth=2.5, alpha=0.8, label='Wall' if wall.id == 0 else '')
        # 标注墙体ID
        mid = wall.center
        ax.text(mid[0], mid[1], f'W{wall.id}',
               fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # 绘制道路中心线（虚线，红色）
    for cl in centerlines:
        ax.plot([cl.p1[0], cl.p2[0]],
               [cl.p1[1], cl.p2[1]],
               'r--', linewidth=2.5, alpha=0.9, label='Centerline' if cl.id == 0 else '')
        # 标注中心线ID和宽度
        mid = cl.center
        ax.text(mid[0], mid[1], f'CL{cl.id}\n{cl.width:.2f}m',
               fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    title = f'Road Centerlines - Centerlines: {len(centerlines)}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化结果已保存: {out_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("道路中心线生成器")
    print("=" * 60)

    # 1. 加载墙体数据
    print("\n【步骤1】加载墙体数据")
    walls = load_walls(WALL_JSON_PATH)

    # 2. 墙体配对
    print(f"\n【步骤2】墙体配对 (宽度范围: {MIN_ROAD_WIDTH}-{MAX_ROAD_WIDTH}m)")
    wall_pairs = find_wall_pairs(
        walls,
        max_width=MAX_ROAD_WIDTH,
        min_width=MIN_ROAD_WIDTH,
        angle_thresh=PARALLEL_ANGLE_THRESH,
        min_overlap=MIN_OVERLAP_RATIO
    )

    # 3. 生成道路中心线
    print("\n【步骤3】生成道路中心线")
    centerlines = []
    for i, (wall1, wall2) in enumerate(wall_pairs):
        cl = generate_centerline_from_pair(wall1, wall2, i)
        centerlines.append(cl)
        print(f"  中心线{i}: 长度={cl.length:.2f}m, 宽度={cl.width:.2f}m")

    # 4. 保存数据
    print("\n【步骤4】保存道路中心线数据")
    os.makedirs(OUT_DIR, exist_ok=True)

    road_network_data = {
        'centerlines': [],
        'metadata': {
            'num_centerlines': len(centerlines),
            'wall_json': WALL_JSON_PATH,
            'params': {
                'max_road_width': MAX_ROAD_WIDTH,
                'min_road_width': MIN_ROAD_WIDTH,
                'parallel_angle_thresh': PARALLEL_ANGLE_THRESH,
                'min_overlap_ratio': MIN_OVERLAP_RATIO
            }
        }
    }

    # 保存中心线
    for cl in centerlines:
        road_network_data['centerlines'].append({
            'id': cl.id,
            'p1': cl.p1.tolist(),
            'p2': cl.p2.tolist(),
            'width': cl.width,
            'length': cl.length,
            'wall_pair': cl.wall_pair
        })

    json_filename = os.path.splitext(os.path.basename(WALL_JSON_PATH))[0].replace('_walls', '_centerlines.json')
    json_path = os.path.join(OUT_DIR, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(road_network_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 道路中心线数据已保存: {json_path}")

    # 5. 可视化
    print("\n【步骤5】可视化结果")
    img_filename = os.path.splitext(os.path.basename(WALL_JSON_PATH))[0].replace('_walls', '_centerlines.png')
    img_path = os.path.join(OUT_DIR, img_filename)
    visualize_results(walls, centerlines, img_path)

    print("\n" + "=" * 60)
    print("✅ 道路中心线生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
