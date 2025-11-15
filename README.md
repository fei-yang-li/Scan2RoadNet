# 道路网络提取系统

基于激光雷达（LiDAR）扫描数据，自动提取道路网络拓扑结构，包括墙体检测、道路中心线生成和路口识别。

## 项目概述

本项目实现了从激光雷达点云数据到道路网络拓扑的完整处理流程：

1. **墙体检测**：使用 RDP 算法和参数空间 DBSCAN 聚类从点云中提取墙体线段
2. **道路中心线生成**：基于左右墙体配对生成道路中心线
3. **路口识别**：通过墙体延长线交点法检测路口区域

## 项目结构

```
test_online_road/
├── generate_road_line_scan_mode_RDP_DBSCAN.py  # 墙体检测脚本
├── generate_road_centerline.py                  # 道路中心线生成脚本
├── README.md                                    # 本文件
└── extracted_lidar_data_code/                   # 数据目录
    ├── extracted_lidar_data/
    │   ├── laserscan_json/                     # 输入：激光雷达 JSON 数据
    │   └── RDP_DBSCAN_scan/                    # 输出：墙体检测可视化结果
    ├── wall/                                    # 输出：墙体数据 JSON
    ├── wall_batch/                              # 批量处理的墙体数据
    └── road_network/                            # 输出：道路网络数据（中心线+路口）
```

## 功能模块

### 1. 墙体检测 (`generate_road_line_scan_mode_RDP_DBSCAN.py`)

从激光雷达扫描数据中提取墙体线段。

#### 处理流程

1. **数据加载与预处理**
   - 支持 LaserScan JSON 格式和 NPY 点云格式
   - 量程过滤、Z 高度过滤、XY 空间过滤

2. **点云分段**
   - **Jump-distance 分段**：基于扫描顺序的跳变检测
   - **RDP 几何简化**：使用 Ramer-Douglas-Peucker 算法简化折线

3. **线段拟合**
   - PCA 直线拟合
   - RMSE 和长度过滤

4. **墙体聚类与合并**
   - **参数空间 DBSCAN**：在 (θ, ρ) 空间对线段聚类
   - **1D 区间合并**：沿墙体方向合并相邻线段，容忍小间隙

#### 主要参数

```python
# 输入输出
INPUT_PATH = "extracted_lidar_data_code/extracted_lidar_data/laserscan_json/laserscan_000135.json"
OUT_DIR = "extracted_lidar_data_code/extracted_lidar_data/RDP_DBSCAN_scan"
WALL_OUT_DIR = "extracted_lidar_data_code/wall"

# 数据过滤
MAX_RANGE = 50.0        # 最大量程（米）
X_LIMIT = 10.0          # X 方向范围（米）
Y_LIMIT = 5.0           # Y 方向范围（米）

# RDP 算法
RDP_EPSILON = 0.05      # 最大允许偏差（米）

# 墙体聚类
PARAM_DBSCAN_EPS = 0.3  # 聚类半径
GAP_THRESH = 2         # 允许的间隙（米）
MIN_WALL_LENGTH = 0.8   # 最短墙体长度（米）
```

#### 输出

- **可视化图像**：6 子图展示处理流程（原始点云 → 分段 → 聚类 → 最终墙体）
- **墙体 JSON**：包含墙体端点、方向、长度、RMSE 等信息

### 2. 道路中心线生成 (`generate_road_centerline.py`)

基于墙体数据生成道路中心线并识别路口。

#### 处理流程

1. **墙体配对**
   - 平行判定（角度阈值）
   - 距离检查（道路宽度范围）
   - 投影重叠度检查

2. **中心线生成**
   - 计算配对墙体的重叠区间
   - 生成中心线起点和终点
   - 计算道路宽度

3. **路口识别**
   - **墙体延长线交点法**：适用于扫描数据
   - 检测模式：
     - 模式1：1 横墙（对面）+ 2 竖墙（左右）→ T 字路口
     - 模式2：1 竖墙（对面）+ 2 横墙（上下）→ T 字路口
   - 构建路口多边形

#### 主要参数

```python
# 输入输出
WALL_JSON_PATH = "extracted_lidar_data_code/wall/laserscan_000135_walls.json"
OUT_DIR = "extracted_lidar_data_code/road_network"

# 道路配对
MAX_ROAD_WIDTH = 3.0        # 最大道路宽度（米）
MIN_ROAD_WIDTH = 0.8        # 最小道路宽度（米）
PARALLEL_ANGLE_THRESH = 10  # 平行角度阈值（度）
MIN_OVERLAP_RATIO = 0.3     # 最小重叠比例

# 路口识别
WALL_ANGLE_TOLERANCE = 15       # 墙体方向分类容差（度）
MAX_INTERSECTION_WALL_DISTANCE = 6.0  # 组成路口的两墙最大距离（米）
MAX_INTERSECTION_SIZE = 8.0     # 路口多边形最大尺寸（米）
```

#### 输出

- **可视化图像**：展示墙体、中心线和路口
- **道路网络 JSON**：包含中心线和路口数据

## 使用方法

### 环境要求

```bash
pip install numpy matplotlib scikit-learn
```

### 步骤 1：墙体检测

```bash
python generate_road_line_scan_mode_RDP_DBSCAN.py
```

**修改输入文件**：编辑脚本中的 `INPUT_PATH` 变量，指向你的激光雷达 JSON 文件。

**输出**：
- `extracted_lidar_data_code/extracted_lidar_data/RDP_DBSCAN_scan/` - 可视化结果
- `extracted_lidar_data_code/wall/` - 墙体数据 JSON

### 步骤 2：生成道路中心线

```bash
python generate_road_centerline.py
```

**修改输入文件**：编辑脚本中的 `WALL_JSON_PATH` 变量，指向步骤 1 生成的墙体 JSON 文件。

**输出**：
- `extracted_lidar_data_code/road_network/` - 道路网络数据（JSON + 可视化）

## 数据格式

### 输入：LaserScan JSON

```json
{
  "laserscan": {
    "angle_min": -3.14159,
    "angle_max": 3.14159,
    "angle_increment": 0.0175,
    "ranges": [1.2, 1.3, ...]
  }
}
```

### 输出：墙体 JSON

```json
{
  "walls": [
    {
      "id": 0,
      "p1": [x1, y1],
      "p2": [x2, y2],
      "length": 2.5,
      "direction": [dx, dy],
      "rmse": 0.05
    }
  ],
  "metadata": {...}
}
```

### 输出：道路网络 JSON

```json
{
  "centerlines": [
    {
      "id": 0,
      "p1": [x1, y1],
      "p2": [x2, y2],
      "width": 1.5,
      "length": 3.2,
      "wall_pair": [0, 1]
    }
  ],
  "intersections": [
    {
      "id": 0,
      "center": [x, y],
      "connected_centerlines": [0, 1],
      "polygon": [[x1, y1], [x2, y2], ...]
    }
  ],
  "metadata": {...}
}
```

## 算法说明

### RDP (Ramer-Douglas-Peucker) 算法

用于简化折线，在保证最大偏差不超过阈值的前提下，用尽量少的关键点表示折线。

### 参数空间 DBSCAN

将线段转换为 Hessian 法线形式 (θ, ρ)，在参数空间进行聚类，将属于同一墙体的线段聚为一类。

### 墙体配对

- **平行判定**：计算方向向量夹角
- **距离检查**：计算两面墙之间的平均距离
- **重叠检查**：计算墙体在方向上的投影重叠比例

### 路口识别

基于墙体延长线交点法，检测横墙和竖墙的组合模式，构建路口多边形。

## 性能优化

- 向量化计算：使用 NumPy 向量化操作加速距离计算
- 递归优化：RDP 算法使用向量化版本
- 时间统计：脚本输出详细的处理耗时统计

## 注意事项

1. **路径配置**：所有路径使用相对路径，基于脚本文件所在目录
2. **参数调整**：根据实际场景调整阈值参数（道路宽度、角度容差等）
3. **数据质量**：输入点云质量影响最终结果，建议先进行数据清洗

## 许可证

本项目仅供学习和研究使用。

