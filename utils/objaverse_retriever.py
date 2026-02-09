"""
Objaverse Asset Retriever for llmscene.

This module provides CLIP+SBERT based retrieval for Objaverse/AI2-THOR assets,
adapted from Holodeck's ObjathorRetriever. It automatically adds 'asset_source': 'objaverse'
to retrieved objects for unified handling with 3D-FUTURE assets.

Usage:
    retriever = ObjaverseRetriever()
    scene = retriever.sample_all_assets(scene_data, is_greedy_sampling=True)
"""

import os
import copy
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy imports for heavy dependencies
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_sbert_model = None
_init_lock = threading.Lock()


def _get_objathor_paths():
    """Get paths for objathor assets, features, and annotations.
    
    Uses PathConfig for unified path management.
    Path resolution order:
    1. PathConfig (from YAML config or init_from_config)
    2. Environment variables (OBJATHOR_ASSETS_BASE_DIR, etc.)
    3. Default hardcoded paths (fallback chain)
    """
    try:
        from path_config import PathConfig
        return PathConfig.get_instance().get_objaverse_paths()
    except ImportError:
        # Fallback to legacy logic if PathConfig is not available
        pass
    
    # Legacy fallback (for backward compatibility)
    assets_version = os.environ.get("ASSETS_VERSION", "2023_09_23")
    hd_base_version = os.environ.get("HD_BASE_VERSION", "2023_09_23")
    
    # Determine base directory with fallback chain
    env_base = os.environ.get("OBJATHOR_ASSETS_BASE_DIR")
    local_dev_base = "/path/to/data/datasets/objathor-assets"
    cloud_base = "/path/to/datasets/objathor-assets"
    local_base = os.path.expanduser("~/.objathor-assets")
    
    if env_base and os.path.exists(env_base):
        base_dir = env_base
    elif os.path.exists(local_dev_base):
        base_dir = local_dev_base
    elif os.path.exists(cloud_base):
        base_dir = cloud_base
    else:
        base_dir = local_base
    
    versioned_dir = os.path.join(base_dir, assets_version)
    
    return {
        "base_dir": base_dir,
        "assets_dir": os.path.join(versioned_dir, "assets"),
        "features_dir": os.path.join(versioned_dir, "features"),
        "annotations_path": os.path.join(versioned_dir, "annotations.json.gz"),
        "holodeck_base_dir": os.path.join(base_dir, "holodeck", hd_base_version),
        "thor_features_dir": os.path.join(base_dir, "holodeck", hd_base_version, "thor_object_data"),
        "thor_annotations_path": os.path.join(base_dir, "holodeck", hd_base_version, "thor_object_data", "annotations.json.gz"),
        "glb_cache_dir": os.path.join(base_dir, "glbs"),
    }


def _lazy_load_clip():
    """Lazy load CLIP model (singleton pattern for thread safety)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_tokenizer
    
    with _init_lock:
        if _clip_model is not None:
            return _clip_model, _clip_preprocess, _clip_tokenizer
        
        try:
            import open_clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Use OpenCLIP ViT-L-14 with laion2b_s32b_b82k weights to match Holodeck's pre-computed features
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k", device=device
            )
            _clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
            print(f"[ObjaverseRetriever] OpenCLIP model (ViT-L-14, laion2b) loaded on {device}")
        except ImportError:
            raise ImportError("Please install open_clip_torch: pip install open_clip_torch")
        
        return _clip_model, _clip_preprocess, _clip_tokenizer


def _lazy_load_sbert():
    """Lazy load SBERT model (singleton pattern for thread safety)."""
    global _sbert_model
    
    if _sbert_model is not None:
        return _sbert_model
    
    with _init_lock:
        if _sbert_model is not None:
            return _sbert_model
        
        try:
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer('all-mpnet-base-v2')
            print("[ObjaverseRetriever] SBERT model loaded")
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        
        return _sbert_model


def get_bbox_dims(annotation: Dict) -> Dict[str, float]:
    """Extract bounding box dimensions from annotation.
    
    检查顺序:
    1. objectOrientedBoundingBox (优先级最高)
    2. axisAlignedBoundingBox
    3. thor_metadata.assetMetadata.boundingBox (Holodeck格式)
    4. 默认值 1.0
    """
    if "objectOrientedBoundingBox" in annotation:
        bbox = annotation["objectOrientedBoundingBox"]
        corner_points = bbox.get("cornerPoints", [])
        if corner_points:
            points = np.array(corner_points)
            dims = points.max(axis=0) - points.min(axis=0)
            return {"x": dims[0], "y": dims[1], "z": dims[2]}
    
    # Fallback to axisAlignedBoundingBox
    if "axisAlignedBoundingBox" in annotation:
        bbox = annotation["axisAlignedBoundingBox"]
        return {
            "x": bbox.get("sizeX", 1.0),
            "y": bbox.get("sizeY", 1.0),
            "z": bbox.get("sizeZ", 1.0),
        }
    
    # Fallback to thor_metadata.assetMetadata.boundingBox (Holodeck format)
    if "thor_metadata" in annotation:
        thor_metadata = annotation["thor_metadata"]
        if "assetMetadata" in thor_metadata:
            asset_metadata = thor_metadata["assetMetadata"]
            if "boundingBox" in asset_metadata:
                bbox_info = asset_metadata["boundingBox"]
                # Handle different bbox formats
                if "x" in bbox_info:
                    return bbox_info
                if "size" in bbox_info:
                    return bbox_info["size"]
                # Calculate from min/max
                if "min" in bbox_info and "max" in bbox_info:
                    mins = bbox_info["min"]
                    maxs = bbox_info["max"]
                    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}
    
    return {"x": 1.0, "y": 1.0, "z": 1.0}


class ObjaverseRetriever(nn.Module):
    """
    Objaverse/AI2-THOR asset retriever using CLIP + SBERT hybrid similarity.
    
    This retriever:
    1. Uses CLIP ViT-B/32 for visual-semantic similarity
    2. Uses SBERT all-mpnet-base-v2 for text semantic similarity
    3. Combines both scores for final ranking
    4. Supports size-based filtering
    5. Automatically adds 'asset_source': 'objaverse' to results
    """
    
    def __init__(
        self,
        retrieval_threshold: float = 25.0,  # Aligned with Holodeck (28.0) but slightly more permissive
        use_text: bool = True,
        size_weight: float = 10.0,
        device: Optional[str] = None,
        do_print: bool = False,
    ):
        super().__init__()
        
        self.retrieval_threshold = retrieval_threshold
        self.use_text = use_text
        self.size_weight = size_weight
        self.do_print = do_print
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load paths
        paths = _get_objathor_paths()
        
        # Load annotations (lazy, only when needed)
        self._database = None
        self._paths = paths
        
        # Load features
        self._clip_features = None
        self._sbert_features = None
        self._asset_ids = None
        
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure all data is loaded (lazy initialization)."""
        if self._initialized:
            return
        
        import compress_json
        import compress_pickle
        
        paths = self._paths
        
        # Load annotations
        objathor_annotations = {}
        thor_annotations = {}
        
        if os.path.exists(paths["annotations_path"]):
            objathor_annotations = compress_json.load(paths["annotations_path"])
            print(f"[ObjaverseRetriever] Loaded {len(objathor_annotations)} objathor annotations")
        
        # Load thor annotations for reference only (THOR assets cannot be rendered in Blender)
        if os.path.exists(paths["thor_annotations_path"]):
            thor_annotations = compress_json.load(paths["thor_annotations_path"])
            print(f"[ObjaverseRetriever] Loaded {len(thor_annotations)} thor annotations (for reference only)")
        
        self._database = {**objathor_annotations, **thor_annotations}
        
        # Load features - ONLY Objaverse features (skip THOR as they cannot be rendered in Blender)
        objathor_clip_path = os.path.join(paths["features_dir"], "clip_features.pkl")
        objathor_sbert_path = os.path.join(paths["features_dir"], "sbert_features.pkl")
        
        # Load objathor features only
        objathor_uids = []
        objathor_clip_features = np.array([])
        objathor_sbert_features = np.array([])
        
        if os.path.exists(objathor_clip_path) and os.path.exists(objathor_sbert_path):
            objathor_clip_dict = compress_pickle.load(objathor_clip_path)
            objathor_sbert_dict = compress_pickle.load(objathor_sbert_path)
            
            assert objathor_clip_dict["uids"] == objathor_sbert_dict["uids"]
            
            objathor_uids = objathor_clip_dict["uids"]
            objathor_clip_features = objathor_clip_dict["img_features"].astype(np.float32)
            objathor_sbert_features = objathor_sbert_dict["text_features"].astype(np.float32)
            print(f"[ObjaverseRetriever] Loaded {len(objathor_uids)} objathor features")
        
        # NOTE: Skip THOR features - THOR assets are AI2-THOR Unity built-in assets
        # and cannot be rendered in Blender. Only Objaverse assets have PKL.GZ/GLB files.
        
        # Use only Objaverse features for retrieval
        if len(objathor_clip_features) == 0:
            raise RuntimeError("No Objaverse features found. Please download objathor data first.")
        
        clip_features = objathor_clip_features
        sbert_features = objathor_sbert_features
        
        self._clip_features = torch.from_numpy(clip_features).to(self.device)
        self._clip_features = F.normalize(self._clip_features, p=2, dim=-1)
        
        self._sbert_features = torch.from_numpy(sbert_features).to(self.device)
        
        self._asset_ids = objathor_uids  # Only Objaverse UIDs
        
        print(f"[ObjaverseRetriever] Total {len(self._asset_ids)} Objaverse assets available for retrieval")
        self._initialized = True
    
    @property
    def database(self) -> Dict:
        """Get annotations database."""
        self._ensure_initialized()
        return self._database
    
    def retrieve(
        self, 
        queries: List[str], 
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve assets matching the queries.
        
        Args:
            queries: List of text descriptions
            threshold: CLIP similarity threshold (default: self.retrieval_threshold)
        
        Returns:
            List of (uid, score) tuples sorted by score descending
        """
        self._ensure_initialized()
        
        threshold = threshold or self.retrieval_threshold
        
        # Get models
        clip_model, _, clip_tokenizer = _lazy_load_clip()
        sbert_model = _lazy_load_sbert()
        
        with torch.no_grad():
            # CLIP text encoding
            text_tokens = clip_tokenizer(queries).to(self.device)
            query_feature_clip = clip_model.encode_text(text_tokens)
            query_feature_clip = F.normalize(query_feature_clip.float(), p=2, dim=-1)
        
        # CLIP similarities
        # self._clip_features shape: (num_assets, num_views, feature_dim) e.g. (50092, 3, 768)
        # query_feature_clip shape: (num_queries, feature_dim) e.g. (1, 768)
        # We compute similarity with each view and take the max
        clip_similarities = 100 * torch.einsum(
            "qd, nvd -> qnv", query_feature_clip, self._clip_features
        )  # shape: (num_queries, num_assets, num_views)
        clip_similarities = torch.max(clip_similarities, dim=-1).values  # shape: (num_queries, num_assets)
        
        # SBERT similarities
        query_feature_sbert = sbert_model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        ).to(self.device)
        sbert_similarities = query_feature_sbert @ self._sbert_features.T
        
        # Combine similarities
        if self.use_text:
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities
        
        # Filter by threshold
        threshold_indices = torch.where(clip_similarities > threshold)
        
        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            unsorted_results.append((self._asset_ids[asset_index], score))
        
        # Sort by score descending
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
        
        return results
    
    def compute_size_difference(
        self, 
        target_size: List[float], 
        candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Adjust candidate scores based on aspect ratio similarity (not absolute size).
        
        Compares the shape/proportions of objects by looking at ratios between dimensions.
        This allows finding objects with similar aspect ratios regardless of absolute size,
        since scaling will be applied during import.
        
        Args:
            target_size: Target size [x, y, z] in meters (used for aspect ratio only)
            candidates: List of (uid, score) tuples
        
        Returns:
            List of (uid, adjusted_score) tuples sorted by adjusted score
        """
        # Compute target aspect ratios
        target_x, target_y, target_z = target_size[0], target_size[1], target_size[2]
        
        # Avoid division by zero
        target_x = max(target_x, 0.001)
        target_y = max(target_y, 0.001)
        target_z = max(target_z, 0.001)
        
        # Compute three key aspect ratios for the target
        # Use ratios between all three dimensions to capture full 3D shape
        target_ratios = [
            target_x / target_y,  # width/height ratio
            target_z / target_y,  # depth/height ratio
            target_x / target_z,  # width/depth ratio
        ]
        
        # Compute aspect ratio differences for each candidate
        aspect_ratio_differences = []
        for uid, _ in candidates:
            if uid in self._database:
                size = get_bbox_dims(self._database[uid])
                cand_x = max(size["x"], 0.001)
                cand_y = max(size["y"], 0.001)
                cand_z = max(size["z"], 0.001)
            else:
                # Default 1m cube (ratio 1:1:1)
                cand_x = cand_y = cand_z = 1.0
            
            # Compute candidate aspect ratios
            cand_ratios = [
                cand_x / cand_y,
                cand_z / cand_y,
                cand_x / cand_z,
            ]
            
            # Compute difference in aspect ratios (using log space for scale-invariance)
            # log(a/b) - log(c/d) = log((a/b)/(c/d))
            # This treats 2:1 and 1:2 as equally different from 1:1
            ratio_diffs = []
            for t_ratio, c_ratio in zip(target_ratios, cand_ratios):
                # Use log ratio difference for better scale-invariance
                diff = abs(np.log(t_ratio + 1e-6) - np.log(c_ratio + 1e-6))
                ratio_diffs.append(diff)
            
            # Average difference across all three ratios
            avg_ratio_diff = np.mean(ratio_diffs)
            aspect_ratio_differences.append(avg_ratio_diff)
        
        # Adjust scores based on aspect ratio similarity
        candidates_with_size_diff = []
        for i, (uid, score) in enumerate(candidates):
            # Penalize based on aspect ratio difference
            # Multiply by size_weight to control importance
            adjusted_score = score - aspect_ratio_differences[i] * self.size_weight
            candidates_with_size_diff.append((uid, adjusted_score))
        
        # Sort by adjusted score
        return sorted(candidates_with_size_diff, key=lambda x: x[1], reverse=True)
    
    def retrieve_single(
        self, 
        description: str, 
        target_size: Optional[List[float]] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
        fallback_threshold: float = 20.0
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the best matching asset for a single description.
        
        Args:
            description: Object description
            target_size: Optional target size [x, y, z] in meters
            top_k: Number of candidates to consider
            threshold: CLIP similarity threshold
            fallback_threshold: Lower threshold to use if no results found
        
        Returns:
            Dict with uid, score, size, annotation, or None if no match
        """
        candidates = self.retrieve([description], threshold=threshold)
        
        # If no candidates found with initial threshold, try lower threshold
        if not candidates and fallback_threshold is not None:
            effective_threshold = threshold or self.retrieval_threshold
            if fallback_threshold < effective_threshold:
                candidates = self.retrieve([description], threshold=fallback_threshold)
        
        if not candidates:
            return None
        
        # Limit to top_k
        candidates = candidates[:top_k]
        
        # Adjust by size if provided
        if target_size is not None:
            # Convert to cm
            target_size_cm = [s * 100 for s in target_size]
            candidates = self.compute_size_difference(target_size_cm, candidates)
        
        if not candidates:
            return None
        
        best_uid, best_score = candidates[0]
        annotation = self._database.get(best_uid, {})
        size = get_bbox_dims(annotation)
        
        return {
            "uid": best_uid,
            "score": best_score,
            "size": [size["x"], size["y"], size["z"]],
            "annotation": annotation,
        }
    
    def sample_all_assets(
        self, 
        scene: Dict, 
        is_greedy_sampling: bool = True,
        threshold: Optional[float] = None,
        top_k: int = 20,
        download_glbs: bool = True
    ) -> Dict:
        """
        Sample assets for all objects in a scene that need retrieval.
        
        Supports both formats:
        - Old format: scene["objects"] = [obj1, obj2, ...]
        - New format: scene["groups"] = [{"objects": [obj1, obj2, ...]}, ...]
        
        Each retrieved object will have:
        - 'uid': The Objaverse asset UID
        - 'asset_source': 'objaverse'
        - 'retrieved_size': The actual asset size
        
        Args:
            scene: Scene dict
            is_greedy_sampling: If True, always pick the best match
            threshold: CLIP similarity threshold
            top_k: Number of candidates to consider per object
            download_glbs: If True, download GLBs after retrieval (for rendering)
        
        Returns:
            Updated scene dict with asset UIDs
        """
        self._ensure_initialized()
        
        sampled_scene = copy.deepcopy(scene)
        
        # Extract all objects
        all_objects = []
        object_locations = []
        
        if 'groups' in scene:
            for group_idx, group in enumerate(scene.get('groups', [])):
                for obj_idx, obj in enumerate(group.get('objects', [])):
                    all_objects.append(obj)
                    object_locations.append(('group', group_idx, obj_idx))
        elif 'objects' in scene:
            for obj_idx, obj in enumerate(scene.get('objects', [])):
                all_objects.append(obj)
                object_locations.append(('objects', obj_idx))
        
        if self.do_print:
            print(f"[ObjaverseRetriever] Processing {len(all_objects)} objects")
        
        # Process each object
        sampled_objects = []
        desc_cache = {}  # Cache for same descriptions
        retrieved_uids = set()  # Track all retrieved UIDs
        
        for obj in all_objects:
            new_obj = copy.deepcopy(obj)
            
            # Check if needs retrieval
            needs_retrieval = (
                obj.get('uid') is None and 
                (obj.get('jid') is None or obj.get('jid') in ['<NEED_RETRIEVAL>', ''])
            )
            
            # Skip if already has Objaverse UID
            if obj.get('asset_source') == 'objaverse' and obj.get('uid'):
                retrieved_uids.add(obj['uid'])
                sampled_objects.append(new_obj)
                continue
            
            # Skip if has 3D-FUTURE jid (different source)
            if obj.get('asset_source') == '3d-future' or (obj.get('jid') and obj.get('jid') not in ['<NEED_RETRIEVAL>', '']):
                sampled_objects.append(new_obj)
                continue
            
            if needs_retrieval:
                desc = obj.get('desc', '')
                size = obj.get('size', [1.0, 1.0, 1.0])
                
                # Check cache
                cache_key = (desc, tuple(size))
                if cache_key in desc_cache:
                    cached = desc_cache[cache_key]
                    new_obj['uid'] = cached['uid']
                    new_obj['asset_source'] = 'objaverse'
                    new_obj['retrieved_size'] = cached['size']
                    retrieved_uids.add(cached['uid'])
                    if self.do_print:
                        print(f"  [Cache] {desc[:50]}... -> {cached['uid'][:16]}...")
                else:
                    # Retrieve
                    result = self.retrieve_single(
                        description=desc,
                        target_size=size,
                        top_k=top_k,
                        threshold=threshold
                    )
                    
                    if result:
                        new_obj['uid'] = result['uid']
                        new_obj['asset_source'] = 'objaverse'
                        new_obj['retrieved_size'] = result['size']
                        retrieved_uids.add(result['uid'])
                        desc_cache[cache_key] = {
                            'uid': result['uid'],
                            'size': result['size']
                        }
                        if self.do_print:
                            print(f"  [Retrieved] {desc[:50]}... -> {result['uid'][:16]}... (score: {result['score']:.2f})")
                    else:
                        if self.do_print:
                            print(f"  [No match] {desc[:50]}...")
            
            sampled_objects.append(new_obj)
        
        # Put objects back
        if 'groups' in scene:
            for group in sampled_scene.get('groups', []):
                group['objects'] = []
            
            for i, location in enumerate(object_locations):
                group_idx = location[1]
                sampled_scene['groups'][group_idx]['objects'].append(sampled_objects[i])
        elif 'objects' in scene:
            sampled_scene['objects'] = sampled_objects
        
        # Download GLBs after retrieval
        if download_glbs and retrieved_uids:
            self._download_glbs_for_uids(list(retrieved_uids))
        
        return sampled_scene
    
    def _download_glbs_for_uids(self, uids: List[str]):
        """Download GLBs for UIDs that don't have cached GLB.
        
        Check order:
        1. Cached GLB in cloud storage (/path/to/datasets/objathor-assets/glbs)
        2. Cached GLB in ~/.objaverse (local fallback)
        3. Download GLB if not exists
        
        Note: PKL.GZ format is no longer supported due to rotation issues.
        """
        if not uids:
            return
        
        # Get paths - check multiple locations
        paths = _get_objathor_paths()
        glb_cache_dirs = [
            Path(paths["glb_cache_dir"]),  # Cloud or configured directory
            Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs",  # Local fallback
        ]
        
        # Check which UIDs actually need GLB download
        need_download = []
        
        for uid in uids:
            if not uid or len(uid) < 2:
                continue
            
            # 使用 uid[:2] 直接构建路径（优化：避免遍历所有子目录）
            has_glb = False
            subdir_name = uid[:2]
            
            for cache_dir in glb_cache_dirs:
                if not cache_dir.is_dir():
                    continue
                # 直接查找 uid[:2] 子目录下的 GLB 文件
                cached_glb = cache_dir / subdir_name / f"{uid}.glb"
                if cached_glb.is_file():
                    has_glb = True
                    if self.do_print:
                        print(f"  [GLB cache] {uid[:16]}... ✓ ({cache_dir})")
                    break
            
            if has_glb:
                continue
            
            # Need to download
            need_download.append(uid)
        
        if not need_download:
            if self.do_print:
                print(f"[ObjaverseRetriever] All {len(uids)} assets already cached (GLB)")
            return
        
        # Download missing GLBs
        try:
            from objaverse_glb_manager import get_global_manager
            
            if self.do_print:
                print(f"\n[ObjaverseRetriever] Downloading {len(need_download)}/{len(uids)} missing GLBs...")
            
            manager = get_global_manager(do_print=self.do_print)
            
            # Batch download
            results = manager.get_glb_paths_batch(
                need_download, 
                download_if_missing=True,
                download_processes=4
            )
            
            # Count successful downloads
            downloaded = sum(1 for path in results.values() if path is not None)
            
            if self.do_print:
                print(f"[ObjaverseRetriever] ✓ Downloaded {downloaded}/{len(need_download)} GLBs")
                
                # Report any failures
                failed = [uid for uid, path in results.items() if path is None]
                if failed:
                    print(f"[ObjaverseRetriever] ⚠ {len(failed)} GLBs could not be downloaded:")
                    for uid in failed[:5]:
                        print(f"    - {uid}")
                    if len(failed) > 5:
                        print(f"    ... and {len(failed)-5} more")
        
        except ImportError:
            if self.do_print:
                print("[ObjaverseRetriever] Warning: ObjaverseGLBManager not available, skipping GLB download")
        except Exception as e:
            if self.do_print:
                print(f"[ObjaverseRetriever] Warning: GLB download failed: {e}")


# Convenience function for quick usage
def create_objaverse_retriever(
    threshold: float = 28.0,
    do_print: bool = False
) -> ObjaverseRetriever:
    """Create an ObjaverseRetriever instance."""
    return ObjaverseRetriever(
        retrieval_threshold=threshold,
        do_print=do_print
    )


if __name__ == "__main__":
    # Test the retriever
    import json
    
    retriever = ObjaverseRetriever(do_print=True)
    
    # Test single retrieval
    result = retriever.retrieve_single(
        description="modern wooden desk with drawers",
        target_size=[1.2, 0.75, 0.6]
    )
    print(f"\nSingle retrieval result: {result}")
    
    # Test scene processing
    test_scene = {
        "room_type": "bedroom",
        "objects": [
            {
                "desc": "Modern minimalist white desk with metal frame",
                "size": [1.2, 0.75, 0.6],
                "pos": [0, 0, 0],
                "rot": [0, 0, 0, 1]
            },
            {
                "desc": "Ergonomic office chair with mesh back",
                "size": [0.6, 1.0, 0.6],
                "pos": [1, 0, 0],
                "rot": [0, 0, 0, 1]
            }
        ]
    }
    
    sampled_scene = retriever.sample_all_assets(test_scene)
    print(f"\nSampled scene:")
    print(json.dumps(sampled_scene, indent=2))
