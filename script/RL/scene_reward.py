"""  
Scene Editing Reward Function for VLM RL Training  
计算基于场景碰撞检测的奖励分数  
"""  

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional  


def compute_score(  
    data_source: str,  
    solution_str: str,  
    ground_truth: Optional[str] = None,  
    extra_info: Optional[Dict[str, Any]] = None,  
) -> float:  
    """  
    单样本奖励计算函数 - 用于 NaiveRewardManager  
      
    Args:  
        data_source: 数据源标识,应该是 "scene_editing"  
        solution_str: 模型生成的响应文本  
        ground_truth: 占位符,可以为 None (不使用)  
        extra_info: 额外信息,包含场景数据和 rollout 奖励  
      
    Returns:  
        float: 奖励分数  
            - 如果 extra_info 中有 rollout_reward_scores,直接返回  
            - 否则尝试从 final_scene 使用 VoxelReward 计算
            - 如果都失败，返回默认值 0.0  
    """  
    # 检查是否是正确的数据源  
    if data_source != "scene_editing":  
        print(f"Warning: Unexpected data_source '{data_source}', expected 'scene_editing'")  
        return 0.0  
    
    # 打印 extra_info 的所有键
    if extra_info:
        print(f"DEBUG [RewardFunction]: extra_info keys = {list(extra_info.keys())}",
              file=sys.stderr, flush=True)
    else:
        raise ValueError("ERROR [RewardFunction]: extra_info is None or empty!")
    
    # 检查 rollout_reward_scores 是否存在且非空
    rollout_rewards = extra_info.get("rollout_reward_scores", {})
    
    if not rollout_rewards:
        # 尝试使用 turn_scores 作为备选
        if "turn_scores" in extra_info:
            print(f"DEBUG [RewardFunction]: Using 'turn_scores' instead of 'rollout_reward_scores'",
                  file=sys.stderr, flush=True)
            turn_scores = extra_info.get("turn_scores", [])
            print(f"DEBUG [RewardFunction]: turn_scores = {turn_scores}",
                  file=sys.stderr, flush=True)
            
            if not turn_scores:
                print(f"WARNING [RewardFunction]: turn_scores is empty",
                      file=sys.stderr, flush=True)
                return 0.0
            
            # turn_scores 是 List[List[float]]，取最后一个列表（最终轮次的分数）
            # 或者计算所有轮次的加权平均
            if isinstance(turn_scores[0], list):
                # List[List[float]] 格式：每个元素是一轮的多个分数
                turn_rewards = [sum(scores) / len(scores) if scores else 0.0 for scores in turn_scores]
            else:
                # List[float] 格式：直接使用
                turn_rewards = turn_scores
            
            print(f"DEBUG [RewardFunction]: Converted turn_rewards = {turn_rewards}",
                  file=sys.stderr, flush=True)
            
            # 使用相同的加权逻辑计算最终分数
            num_turns = len(turn_rewards)
            weights = None
            
            if num_turns <= 2:
                final_score = -1
            else:
                weights = [0.0] * num_turns
                weights[-1] = 1
                other_turns = num_turns - 1
                if other_turns > 0:
                    other_weight = 0.0 / other_turns
                    for i in range(num_turns - 1):
                        weights[i] = other_weight
                final_score = sum(w * r for w, r in zip(weights, turn_rewards))
            
            print(f"DEBUG [RewardFunction]: num_turns = {num_turns}",
                  file=sys.stderr, flush=True)
            if weights is not None:
                print(f"DEBUG [RewardFunction]: weights = {weights}",
                      file=sys.stderr, flush=True)
            print(f"DEBUG [RewardFunction]: final_score = {final_score:.4f}",
                  file=sys.stderr, flush=True)
            
            return float(final_score)
        else:
            raise KeyError(
                f"ERROR [RewardFunction]: Neither 'rollout_reward_scores' nor 'turn_scores' found in extra_info! "
                f"Available keys: {list(extra_info.keys())}"
            )
      
    # 添加调试输出  
    print(f"DEBUG [RewardFunction]: rollout_rewards type = {type(rollout_rewards)}",   
          file=sys.stderr, flush=True)  
    print(f"DEBUG [RewardFunction]: rollout_rewards = {rollout_rewards}",   
          file=sys.stderr, flush=True)  
      
    if isinstance(rollout_rewards, dict):
            # 获取 user_turn_rewards 列表
            turn_rewards = rollout_rewards.get("user_turn_rewards", [])
            
            if not turn_rewards:
                print(f"WARNING [RewardFunction]: user_turn_rewards is empty",
                      file=sys.stderr, flush=True)
                return 0.0
            
            print(f"DEBUG [RewardFunction]: turn_rewards = {turn_rewards}",   
                  file=sys.stderr, flush=True)
            
            # 计算加权平均奖励
            num_turns = len(turn_rewards)
            weights = None  # 初始化 weights 变量
            
            if num_turns <= 2:
                # 只有一轮或两轮，直接返回 -1
                final_score = -1
            else:
                # 三轮及以上：最后一轮权重最高，其他轮平分剩余权重
                # 权重分配：最后一轮 0.6，其他轮平分剩余 0.4
                weights = [0.0] * num_turns
                weights[-1] = 0.6  # 最后一轮权重最高
                
                # 其他轮（包括第一轮和中间轮）平分剩余权重
                other_turns = num_turns - 1
                if other_turns > 0:
                    other_weight = 0.4 / other_turns
                    for i in range(num_turns - 1):
                        weights[i] = other_weight
                
                # 计算加权和
                final_score = sum(w * r for w, r in zip(weights, turn_rewards))
            
            print(f"DEBUG [RewardFunction]: num_turns = {num_turns}",   
                  file=sys.stderr, flush=True)
            if weights is not None:
                print(f"DEBUG [RewardFunction]: weights = {weights}",   
                      file=sys.stderr, flush=True)
            print(f"DEBUG [RewardFunction]: final_score = {final_score:.4f}",   
                  file=sys.stderr, flush=True)
              
            return float(final_score)
    
    # rollout_rewards 不是 dict 类型
    print(f"WARNING [RewardFunction]: rollout_rewards is not a dict, type = {type(rollout_rewards)}",
          file=sys.stderr, flush=True)
    return 0.0
    

  
  
def compute_score_batch(  
    data_sources: List[str],  
    solution_strs: List[str],  
    ground_truths: List[Optional[str]],  
    extra_infos: List[Dict[str, Any]],  
) -> List[float]:  
    """  
    批量奖励计算函数 - 用于 BatchRewardManager  
      
    Args:  
        data_sources: 数据源标识列表  
        solution_strs: 模型生成的响应文本列表  
        ground_truths: 占位符列表,可以为 None  
        extra_infos: 额外信息列表  
      
    Returns:  
        List[float]: 奖励分数列表  
    """  
    scores = []  
    for data_source, solution_str, ground_truth, extra_info in zip(  
        data_sources, solution_strs, ground_truths, extra_infos, strict=True  
    ):  
        score = compute_score(data_source, solution_str, ground_truth, extra_info)  
        scores.append(score)  
      
    return scores