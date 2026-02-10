"""  
Scene Editing Reward Function for VLM RL Training  
Compute reward scores based on scene collision detection  
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
    Single-sample reward computation function - for NaiveRewardManager  
      
    Args:  
        data_source: Data source identifier, should be "scene_editing"  
        solution_str: Model-generated response text  
        ground_truth: Placeholder, can be None (not used)  
        extra_info: Extra info, containing scene data and rollout rewards  
      
    Returns:  
        float: Reward score  
            - If rollout_reward_scores exists in extra_info, return directly  
            - Otherwise try to compute using VoxelReward from final_scene
            - If all fail, return default value 0.0  
    """  
    # Check if the data source is correct  
    if data_source != "scene_editing":  
        print(f"Warning: Unexpected data_source '{data_source}', expected 'scene_editing'")  
        return 0.0  
    
    # Print all keys of extra_info
    if extra_info:
        print(f"DEBUG [RewardFunction]: extra_info keys = {list(extra_info.keys())}",
              file=sys.stderr, flush=True)
    else:
        raise ValueError("ERROR [RewardFunction]: extra_info is None or empty!")
    
    # Check if rollout_reward_scores exists and is non-empty
    rollout_rewards = extra_info.get("rollout_reward_scores", {})
    
    if not rollout_rewards:
        # Try using turn_scores as a fallback
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
            
            # turn_scores is List[List[float]], take the last list (final turn scores)
            # or compute weighted average across all turns
            if isinstance(turn_scores[0], list):
                # List[List[float]] format: each element contains multiple scores from one turn
                turn_rewards = [sum(scores) / len(scores) if scores else 0.0 for scores in turn_scores]
            else:
                # List[float] format: use directly
                turn_rewards = turn_scores
            
            print(f"DEBUG [RewardFunction]: Converted turn_rewards = {turn_rewards}",
                  file=sys.stderr, flush=True)
            
            # Use the same weighting logic to compute the final score
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
      
    # Add debug output  
    print(f"DEBUG [RewardFunction]: rollout_rewards type = {type(rollout_rewards)}",   
          file=sys.stderr, flush=True)  
    print(f"DEBUG [RewardFunction]: rollout_rewards = {rollout_rewards}",   
          file=sys.stderr, flush=True)  
      
    if isinstance(rollout_rewards, dict):
            # Get user_turn_rewards list
            turn_rewards = rollout_rewards.get("user_turn_rewards", [])
            
            if not turn_rewards:
                print(f"WARNING [RewardFunction]: user_turn_rewards is empty",
                      file=sys.stderr, flush=True)
                return 0.0
            
            print(f"DEBUG [RewardFunction]: turn_rewards = {turn_rewards}",   
                  file=sys.stderr, flush=True)
            
            # Compute weighted average reward
            num_turns = len(turn_rewards)
            weights = None  # Initialize weights variable
            
            if num_turns <= 2:
                # Only one or two turns, directly return -1
                final_score = -1
            else:
                # Three or more turns: last turn has highest weight, other turns share remaining weight
                # Weight distribution: last turn 0.6, other turns share remaining 0.4
                weights = [0.0] * num_turns
                weights[-1] = 0.6  # Last turn has highest weight
                
                # Other turns (including first and middle turns) share remaining weight
                other_turns = num_turns - 1
                if other_turns > 0:
                    other_weight = 0.4 / other_turns
                    for i in range(num_turns - 1):
                        weights[i] = other_weight
                
                # Compute weighted sum
                final_score = sum(w * r for w, r in zip(weights, turn_rewards))
            
            print(f"DEBUG [RewardFunction]: num_turns = {num_turns}",   
                  file=sys.stderr, flush=True)
            if weights is not None:
                print(f"DEBUG [RewardFunction]: weights = {weights}",   
                      file=sys.stderr, flush=True)
            print(f"DEBUG [RewardFunction]: final_score = {final_score:.4f}",   
                  file=sys.stderr, flush=True)
              
            return float(final_score)
    
    # rollout_rewards is not a dict type
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
    Batch reward computation function - for BatchRewardManager  
      
    Args:  
        data_sources: List of data source identifiers  
        solution_strs: List of model-generated response texts  
        ground_truths: Placeholder list, can be None  
        extra_infos: List of extra info  
      
    Returns:  
        List[float]: List of reward scores  
    """  
    scores = []  
    for data_source, solution_str, ground_truth, extra_info in zip(  
        data_sources, solution_strs, ground_truths, extra_infos, strict=True  
    ):  
        score = compute_score(data_source, solution_str, ground_truth, extra_info)  
        scores.append(score)  
      
    return scores