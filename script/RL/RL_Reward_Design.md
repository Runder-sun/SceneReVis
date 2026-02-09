# 强化学习场景编辑系统：奖励函数设计与技术细节

## 1. 系统概述

本系统采用强化学习（RL）框架，通过多轮交互优化 3D 室内场景的布局。系统结合了基于规则的物理评估、基于大语言模型（LLM）的格式检查以及基于视觉语言模型（VLM）的语义与审美评估，构建了一个多维度的奖励体系。该体系旨在引导智能体生成既符合物理约束（如无碰撞、无悬空），又满足用户语义需求且布局合理的场景。

## 2. 奖励函数架构

奖励函数的设计区分了**中间轮次（Intermediate Turns）**和**最终轮次（Final Turn）**，以适应不同阶段的优化目标。

### 2.1 最终轮次奖励 (Final Turn Reward)

最终轮次的奖励设计最为全面，采用分层评估体系，总权重为 1.0。主要分为三个层次：**格式层**、**物体层**和**场景层**。

#### A. 格式层 (Format Layer) - 权重 0.10
确保模型输出的结构化指令符合系统要求。
- **Format Reward**: 检查 XML 标签（`<think>`, `<tool_calls>`）的完整性、顺序以及 JSON 格式的正确性。

#### B. 物体层 (Object Layer) - 权重 0.30
关注单个物体的属性及其与需求的匹配度。
- **VLM Key Objects (0.20)**: 利用 VLM 评估场景中是否包含用户需求或房间类型所必需的关键物体（如卧室必须有床）。
- **VLM Size Proportion (0.10)**: 程序化评估物体尺寸是否在合理范围内（基于真实资产尺寸或预定义标准），惩罚过大或过小的物体。

#### C. 场景层 (Scene Layer) - 权重 0.60
关注物体间的空间关系、物理合理性及整体布局质量。

**1. 物理评估 (Physical Evaluation) - 权重 0.30**
使用 Trimesh 物理引擎进行精确计算：
- **Collision Rate (0.08)**: 惩罚物体间的碰撞体积比例。
- **OOB Rate (0.07)**: 惩罚物体超出房间边界的体积比例。
- **Penetration Depth (0.05)**: 惩罚物体间的穿透深度。
- **OOB Volume (0.05)**: 惩罚出界体积的绝对值。
- **Support (0.05)**: 检查物体是否有合理的支撑（如在地板上、桌面上或挂在墙/天花板上）。

**2. VLM 语义与布局评估 (VLM Semantic & Layout) - 权重 0.30**
利用 VLM（如 GPT-4o 或 Qwen-VL）对渲染图进行高层语义分析：
- **Rationality (0.10)**: 评估布局的合理性（如家具分布是否均匀、是否符合生活常识）。
- **Requirement Match (0.10)**: 评估场景是否满足用户的显式和隐式需求。
- **Scene Graph Constraints (0.10)**: 评估物体间的空间关系是否符合预期（如“床头柜在床边”、“电视对着沙发”）。

### 2.2 中间轮次奖励 (Intermediate Turn Reward)

中间轮次的奖励旨在引导智能体逐步改进场景，避免在早期陷入局部最优。

- **Format (0.10)**: 保持格式正确性。
- **Physics (0.40)**: 包含 Collision Rate, OOB Rate, Penetration Depth, OOB Volume 各 0.10。在中间过程中持续施加物理约束，防止生成物理上不可行的中间状态。
- **VLM Scene Improvement (0.25)**: 对比当前轮次与上一轮次的渲染图，评估场景质量是否有提升（Improved/Changed/Worse）。
- **VLM Key Objects (0.25)**: 持续引导智能体添加缺失的关键物体。

## 3. 关键技术组件详解

### 3.1 物理评估模块 (Physics Evaluation)
- **Trimesh Metrics**: 利用 `trimesh` 库计算精确的网格碰撞和包含关系。
- **Support Analysis**: 结合规则（高度、位置）和 VLM 分类（物体是落地、挂墙还是置于桌面），计算无支撑物体的比例。

### 3.2 VLM 评估模块 (VLM Judge)
- **Visual Annotations**: 在输入 VLM 的渲染图中叠加地板网格和物体包围盒（Bounding Boxes），辅助 VLM 理解空间尺度和位置。
- **Dual-View Rendering**: 同时提供顶视图（Top View）和对角视图（Diagonal View），确保 VLM 既能看到全局布局，又能观察垂直方向的堆叠关系。
- **Chain-of-Thought Prompting**: 引导 VLM 先描述场景（Object Inventory, Spatial Layout），再进行评分，提高评估的稳定性和可解释性。

### 3.3 格式与逻辑检查
- **Tool Execution Check**: 验证工具调用参数的有效性（如物体 ID 是否存在），对无效调用施加惩罚。
- **Room Shape & Area**: 在第一轮生成时，检查房间形状（顶点数）和面积是否合理，防止生成过小或畸形的房间。

## 4. 奖励计算流程图

```mermaid
graph TD
    A[Agent Action] --> B{Turn Type?}
    
    B -- Intermediate --> C[Intermediate Reward]
    C --> C1[Format Check (0.1)]
    C --> C2[Physics Check (0.4)]
    C --> C3[VLM Improvement (0.25)]
    C --> C4[VLM Key Objects (0.25)]
    
    B -- Final --> D[Final Reward]
    D --> D1[Format Layer (0.1)]
    D --> D2[Object Layer (0.3)]
    D2 --> D2a[Key Objects (0.2)]
    D2 --> D2b[Size Proportion (0.1)]
    D --> D3[Scene Layer (0.6)]
    D3 --> D3a[Physics Metrics (0.3)]
    D3 --> D3b[VLM Evaluation (0.3)]
    
    C --> E[Total Reward]
    D --> E
```

## 5. 总结

该奖励设计通过**显式规则**（物理引擎）与**隐式语义**（VLM）的结合，解决了传统 3D 生成任务中难以同时兼顾物理真实性和语义一致性的难题。分阶段的奖励机制有效地引导了智能体在长视界（Long-horizon）任务中的探索与利用。
