# 多轮Chain-of-Thought场景编辑数据集构建流程技术文档

## 摘要

本文档详细描述了一套用于构建多轮Chain-of-Thought（CoT）场景编辑对话数据集的自动化流水线。该流水线采用**逆向工程**方法，从高质量的最终场景出发，通过场景状态逆推、多轮对话生成、视觉渲染、CoT推理生成以及质量评估筛选等关键步骤，构建出适用于训练视觉语言模型进行室内设计场景编辑的高质量对话数据集。

---

## 1. 问题定义与方法论概述

### 1.1 核心挑战

构建多轮场景编辑对话数据集面临以下关键挑战：

1. **场景状态一致性**：多轮编辑过程中，每一轮的场景状态必须与前一轮的编辑操作结果严格对应
2. **操作可逆性**：需要能够从最终状态反推中间状态，保证编辑链的逻辑完整性
3. **CoT推理质量**：生成的思维链需要与实际执行的操作严格对应，不能出现逻辑偏差
4. **数据多样性**：需要在指令风格、操作类型、场景复杂度等维度保持足够的多样性

### 1.2 逆向工程方法论

本流水线采用**逆向工程**（Reverse Engineering）策略，核心思想如下：

$$
S_{final} \xrightarrow{\text{Reverse Operations}} S_{n-1} \xrightarrow{\text{Reverse Operations}} \cdots \xrightarrow{\text{Reverse Operations}} S_0 = \emptyset
$$

其中：
- $S_{final}$ 为经过质量筛选的高质量最终场景
- $S_i$ 为第 $i$ 轮编辑后的场景状态
- $S_0$ 为空场景或初始场景

这种方法的优势在于：
- **保证最终状态质量**：最终场景来自经过筛选的高质量数据
- **逻辑完整性**：每轮操作与场景状态变化严格对应
- **灵活控制**：可以精确控制编辑轮数和操作分布

---

## 2. 流水线架构

### 2.1 整体流程

```
┌─────────────────────┐
│  原始场景数据 (SSR)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 第0步: 场景质量筛选  │  filter_scenes.py
│ (碰撞率/越界率/物体数)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 第1步: 中间数据生成  │  generate_intermediate_data.py
│ (逆向推导+对话模板)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 第2步: 批量场景渲染  │  batch_process_scenes.py
│ (Blender多视角渲染) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 第3步: 编辑链评估筛选│  evaluate_and_filter_chains.py
│ (GPT-4多维度评分)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 第4步: CoT推理生成   │  generate_final_conversations_v3.py
│ (GPT-4o视觉推理)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  最终训练数据集      │
└─────────────────────┘
```

### 2.2 数据格式规范

#### 2.2.1 场景表示格式

系统支持两种场景表示格式，并提供统一的格式转换接口：

**SSR格式**（扁平化表示）：
```json
{
  "objects": [...],
  "bounds_top": [[x1,y1,z1], ...],
  "bounds_bottom": [[x1,y1,z1], ...],
  "room_type": "bedroom",
  "room_id": "scene_001"
}
```

**Groups格式**（分组表示）：
```json
{
  "groups": [{
    "group_name": "...",
    "group_type": "...",
    "objects": [...]
  }],
  "room_envelope": {
    "bounds_top": [...],
    "bounds_bottom": [...]
  }
}
```

#### 2.2.2 物体表示

每个物体包含以下属性：
```json
{
  "jid": "unique_identifier",      // 物体唯一标识符
  "desc": "object description",    // 物体描述
  "pos": [x, y, z],               // 位置坐标
  "rot": [x, y, z, w],            // 四元数旋转
  "size": [width, height, depth], // 尺寸
  "type": "major/minor"           // 物体类型
}
```

---

## 3. 核心技术实现

### 3.1 第0步：场景质量筛选

**实现脚本**: `filter_scenes.py`

#### 3.1.1 筛选策略

采用多条件联合筛选策略，场景满足以下任一条件即被保留：

**条件1（标准条件）**:
$$\text{Collision Rate} < 10\% \land \text{OOB Rate} < 30\% \land |\text{Objects}| > 4$$

**条件2（高复杂度场景）**:
$$|\text{Objects}| \geq 20$$

**条件3（零碰撞场景）**:
$$\text{Collision Rate} = 0\% \land |\text{Objects}| > 5$$

其中：
- $\text{Collision Rate}$ 为物体间碰撞率
- $\text{OOB Rate}$ 为越界率（Out-of-Bounds）
- $|\text{Objects}|$ 为场景物体数量

#### 3.1.2 统计分析

筛选过程同时生成多维度统计信息：
- 房间类型分布（bedroom, livingroom, office等）
- 物体数量分布（5-10, 11-15, 16-20, 21-25, 26-30, 31+）
- 平均物体数量

### 3.2 第1步：中间数据生成

**实现脚本**: `generate_intermediate_data.py`

#### 3.2.1 SceneReverseEditor 类

核心逆向编辑器类，负责从最终场景逆推原始场景和操作序列。

**关键方法**:

```python
class SceneReverseEditor:
    def add_random_variations(
        self, 
        final_scene: Dict,
        current_subscene_name: str = "",
        num_operations: int = None,
        favor_add: bool = False,
        step_progress: float = 0.0
    ) -> Tuple[Dict, List[Dict]]
```

#### 3.2.2 操作类型与概率分布

系统定义7种编辑操作：

| 操作类型 | 单轮模式概率 | 多轮模式概率 | 功能描述 |
|---------|------------|------------|---------|
| `add_object` | 30% | 35% | 添加新物体 |
| `remove_object` | 10% | 10% | 移除物体 |
| `move_object` | 20% | 20% | 移动物体位置 |
| `rotate_object` | 20% | 20% | 旋转物体 |
| `scale_object` | 10% | 5% | 缩放物体 |
| `replace_object` | 10% | 10% | 替换物体 |
| `terminate` | - | - | 终止编辑 |

#### 3.2.3 逆向操作实现

以`add_object`的逆向实现为例：

```python
def _reverse_add_operation(self, initial_scene, tool_id, step_progress):
    """
    逆向添加操作：从原始场景中移除一个物体
    正向执行时等效于添加该物体到场景
    
    采用分阶段选择策略：
    - step_progress < 0.3: 优先选择小物件（体积 < 0.5m³）
    - 0.3 ≤ step_progress < 0.7: 优先选择中等物件（0.5-2.0m³）
    - step_progress ≥ 0.7: 优先选择大物件（≥ 2.0m³）
    """
```

这种分阶段策略确保了：
- 逆向初期（正向末期）处理装饰品
- 逆向中期（正向中期）处理次要家具
- 逆向末期（正向初期）处理核心家具

#### 3.2.4 JID处理机制

对于LLM无法预知的物体标识符（如新增/替换物体），采用特殊标记：

```python
def process_final_scene_jids(self, final_scene, tool_calls):
    """
    处理final_scene中的JID:
    - add_object操作新增的物体: jid = "<NEED_RETRIVEAL>"
    - replace_object操作替换的物体: jid = "<NEED_RETRIVEAL>"
    """
```

#### 3.2.5 用户指令生成

使用GPT-4生成多样化的用户指令，支持两种风格：

**简短指令（50%概率）**:
- Style A: "I want a comfortable bedroom"
- Style B: "Create a workspace with a desk and chair"
- Style C: "I need a modern living room with good lighting"

**详细指令（50%概率）**:
- 高度具体型（25%）：包含具体材质、颜色、数量
- 类别聚焦型（20%）：关注家具类别和功能需求
- 风格审美型（20%）：强调设计风格和美学
- 生活方式型（15%）：聚焦空间使用场景
- 空间布局型（10%）：强调空间组织和流动性
- 混合型（10%）：结合具体需求和通用目标

### 3.3 第2步：批量场景渲染

**实现脚本**: `batch_process_scenes.py`

#### 3.3.1 渲染策略

使用Blender进行场景渲染，生成双视角合并图像：

```
┌─────────────────────────────────────┐
│  Top-down View  │  Diagonal View   │
│   (俯视图)       │   (斜视图)        │
└─────────────────────────────────────┘
```

双视角设计的优势：
- **俯视图**：清晰展示空间布局和物体相对位置
- **斜视图**：展示3D外观和空间感

#### 3.3.2 多进程并行处理

```python
def main():
    # 使用ProcessPoolExecutor进行并行渲染
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_file_wrapper, arg): arg 
                   for arg in task_args}
```

#### 3.3.3 断点续传机制

使用pickle文件保存处理进度：

```python
def load_progress_file(progress_file_path):
    """加载进度文件，返回已处理的文件集合"""
    
def save_progress_file(progress_file_path, processed_files):
    """保存进度文件"""
```

### 3.4 第3步：编辑链评估筛选

**实现脚本**: `evaluate_and_filter_chains.py`

#### 3.4.1 评估维度

采用多维度评分体系：

| 评估维度 | 权重 | 评估要点 |
|---------|-----|---------|
| 编辑连贯性 | 40分 | 操作逻辑性、步骤渐进性、对话一致性 |
| 编辑自然性 | 35分 | 操作直观性、中间状态意义、设计流程真实性 |
| 指令遵循度 | 15分 | 操作准确性、理解清晰度 |
| 视觉过渡质量 | 10分 | 中间状态合理性、过渡平滑度 |

#### 3.4.2 评估流程

```python
def evaluate_single_chain(chain_dir: Path, scene_folder_name: str) -> Dict:
    """
    评估单条编辑链:
    1. 加载编辑链JSON数据
    2. 收集所有渲染图片
    3. 构建评估提示词
    4. 调用GPT-4进行多模态评估
    5. 解析评分结果
    """
```

#### 3.4.3 筛选策略

- 每个场景生成10条编辑链
- 按总分降序排序
- 保留Top-K（默认K=3）条编辑链
- 删除其余编辑链节省存储空间

### 3.5 第4步：CoT推理生成

**实现脚本**: `generate_final_conversations_v3.py`

#### 3.5.1 CoT模板设计

系统设计5种不同风格的CoT模板，等概率随机选择：

| 模板 | 风格 | 特点 |
|-----|------|------|
| Template A | 详细分析型 | 4步法：问题识别→策略规划→参数论证→影响验证 |
| Template B | 策略先行型 | 强调策略性思考和设计优先级 |
| Template C | 问答型 | 以问答形式组织推理过程 |
| Template D | 柔性指令型 | 灵活的链式思维推理 |
| Template E | 零样本CoT型 | 最小化约束的自由推理 |

#### 3.5.2 问题分类框架

所有模板共享统一的问题分类体系：

**1. 物理类问题（Physical Bugs）**:
- 物体重叠/碰撞
- 越界放置
- 悬浮物体

**2. 布局合理性问题（Layout Rationality Bugs）**:
- 核心家具错位（床/沙发未靠墙）
- 缺失必要家具
- 朝向错误

**3. 空间分布问题（Spatial Distribution Bugs）**:
- 家具聚集
- 大面积空白
- 布局不平衡

#### 3.5.3 视觉推理调用

使用GPT-4o的视觉能力进行推理：

```python
def generate_single_turn_cot(client, base64_image, global_user_instruction, 
                            current_scene, tool_calls, metadata, 
                            turn_index, previous_actions):
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }}
            ]}
        ]
    )
```

#### 3.5.4 速率限制器

实现线程安全的API调用速率限制：

```python
class RateLimiter:
    """线程安全的速率限制器，确保API调用间隔至少为指定秒数"""
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.last_call_time = 0
        self.lock = threading.Lock()
```

---

## 4. 对话数据格式

### 4.1 多轮对话结构

```json
{
  "id": "multi_turn_edit_1_abc12345",
  "images": ["step_0.png", "step_1.png", ..., "step_n.png"],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>\n{instruction}\n<current_scene>..."},
    {"role": "assistant", "content": "<think>...</think>\n<tool_calls>..."},
    ...
  ],
  "metadata": {
    "scene_id": "...",
    "conversation_type": "multi_turn",
    "total_turns": n,
    "global_user_instruction": "...",
    "final_scene": {...}
  }
}
```

### 4.2 输出格式规范

Assistant响应采用结构化标签：

```
<think>
{推理过程}
</think>

<conclusion>
{一句话总结}
</conclusion>

<tool_calls>
[
  {
    "id": "tool_1",
    "name": "move_object",
    "arguments": {...}
  }
]
</tool_calls>
```

---

## 5. 技术亮点与创新点

### 5.1 逆向工程保证数据质量

通过从高质量最终场景逆推，确保：
- 最终状态的质量有保障
- 中间状态的物理一致性
- 操作序列的逻辑完整性

### 5.2 分阶段物体选择策略

根据编辑进度动态调整物体选择策略：

$$
P(obj_i | \text{stage}) = \begin{cases}
\text{高}, & \text{if } V(obj_i) \in \text{当前阶段目标体积范围} \\
\text{低}, & \text{otherwise}
\end{cases}
$$

这确保了正向编辑时"先大后小"的自然设计流程。

### 5.3 多样化CoT模板

5种不同风格的CoT模板等概率采样，有效避免模型对特定推理模式的过拟合。

### 5.4 多维度质量评估

从连贯性、自然性、遵循度、视觉质量四个维度综合评估，确保筛选出高质量编辑链。

### 5.5 鲁棒的工程实现

- 多进程并行处理提升效率
- 断点续传支持长时任务
- API速率限制避免请求失败
- 统一的格式转换接口

---

## 6. 使用指南

### 6.1 环境依赖

```bash
# 核心依赖
pip install openai azure-identity tqdm pillow

# 渲染依赖
# 需要安装Blender及相关Python模块
```

### 6.2 执行流程

```bash
# Step 0: 场景筛选
python filter_scenes.py

# Step 1: 生成中间数据
python generate_intermediate_data.py \
    --input-dir /path/to/filtered_scenes \
    --output-dir /path/to/intermediate_data \
    --multi-turn

# Step 2: 批量渲染
python batch_process_scenes.py \
    --input-dir /path/to/intermediate_data \
    --output-dir /path/to/rendered_outputs \
    --max-workers 32

# Step 3: 评估筛选
python evaluate_and_filter_chains.py \
    --input-dir /path/to/rendered_outputs \
    --top-k 3

# Step 4: 生成CoT
python generate_final_conversations_v3.py \
    --input-dir /path/to/rendered_outputs \
    --output-dir /path/to/final_data \
    --num-processes 4
```

---

## 7. 总结

本流水线通过逆向工程方法，系统化地解决了多轮CoT场景编辑数据集构建中的关键技术挑战。从质量筛选、逆向推导、视觉渲染、评估筛选到CoT推理生成，形成了完整的自动化数据生产流程。生成的数据集具有高质量、多样性和逻辑一致性等特点，适用于训练具备视觉推理能力的室内设计场景编辑模型。

---

## 附录：配置参数参考

### A.1 筛选参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `collision_rate_threshold` | 10.0 | 碰撞率阈值 |
| `oob_rate_threshold` | 30.0 | 越界率阈值 |
| `min_objects` | 4 | 最小物体数量 |
| `high_object_threshold` | 20 | 高物体数量阈值 |

### A.2 生成参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `target_turns` | 4-8 | 目标编辑轮数 |
| `chains_per_scene` | 3 | 每场景保留编辑链数 |
| `max_attempts` | 30 | 最大生成尝试次数 |

### A.3 评估参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `top_k` | 3 | 保留最佳编辑链数量 |
| `max_workers` | 10 | 并行评估进程数 |
