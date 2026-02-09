"""
Objaverse GLB Asset Manager for llmscene.

This module handles downloading, caching, and path resolution for Objaverse GLB assets.
It supports:
1. Pre-downloading all assets for fast RL training
2. Lazy downloading on-demand
3. Fast O(1) path lookup for cached assets

Usage:
    manager = ObjaverseGLBManager()
    
    # Get GLB path (downloads if not cached)
    glb_path = manager.get_glb_path("abc123...")
    
    # Pre-download all assets
    manager.predownload_all(num_processes=8)
"""

import os
import json
import gzip
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import objaverse
try:
    import objaverse
    # Try to import objaverse.xl for custom download_dir support
    try:
        from objaverse.xl.sketchfab import SketchfabDownloader
        import pandas as pd
        _HAVE_OBJAVERSE_XL = True
    except ImportError:
        _HAVE_OBJAVERSE_XL = False
    _HAVE_OBJAVERSE = True
except ImportError:
    _HAVE_OBJAVERSE = False
    _HAVE_OBJAVERSE_XL = False
    print("[ObjaverseGLBManager] Warning: objaverse package not installed")


class ObjaverseGLBManager:
    """
    Manager for Objaverse GLB assets with caching and pre-download support.
    
    GLB files are stored in: {cache_dir}/glbs/{uid[:2]}/{uid}.glb
    
    Path resolution order:
    1. OBJAVERSE_GLB_CACHE_DIR environment variable
    2. /path/to/datasets/objathor-assets/glbs (for cloud deployment)
    3. ~/.objaverse/hf-objaverse-v1/glbs (local fallback)
    
    This manager:
    1. Maintains a cache index for O(1) path lookups
    2. Supports parallel pre-downloading for RL training
    3. Falls back to on-demand downloading if needed
    """
    
    # Default paths - prioritize cloud storage, then local
    @staticmethod
    def _get_default_cache_dir():
        """Get default cache directory with fallback chain.
        
        Uses PathConfig for unified path management.
        """
        try:
            from path_config import PathConfig
            cache_dir = PathConfig.get_instance().objaverse_glb_cache_dir
            if cache_dir:
                return os.path.dirname(cache_dir)  # Return base dir, not glbs subdir
        except ImportError:
            pass
        
        # Legacy fallback
        # 1. Environment variable override
        env_dir = os.environ.get("OBJAVERSE_GLB_CACHE_DIR")
        if env_dir and os.path.exists(env_dir):
            return env_dir
        
        # 2. Local development path (for local machine)
        local_dev_dir = "/path/to/data/datasets/objathor-assets"
        if os.path.exists(local_dev_dir):
            return local_dev_dir
        
        # 3. Cloud storage (blobfuse mount on Azure)
        cloud_dir = "/path/to/datasets/objathor-assets"
        if os.path.exists(cloud_dir):
            return cloud_dir
        
        # 4. Local fallback
        return os.path.expanduser("~/.objaverse")
    
    DEFAULT_CACHE_DIR = None  # Set dynamically in __init__
    GLB_SUBDIR = "glbs"  # Simplified - no longer need hf-objaverse-v1 prefix for cloud storage
    
    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        do_print: bool = False
    ):
        """
        Initialize the GLB manager.
        
        Args:
            cache_dir: Override default cache directory
            do_print: Enable verbose logging
        """
        # Determine cache directory with fallback chain
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = self._get_default_cache_dir()
        
        # Determine GLB subdirectory based on storage location
        # Cloud storage and local dev path use flat 'glbs/' structure
        # Local ~/.objaverse uses 'hf-objaverse-v1/glbs/' structure
        if "/path/to/storage/" in self.cache_dir or "objathor-assets" in self.cache_dir or os.environ.get("OBJAVERSE_GLB_CACHE_DIR"):
            self.glb_base_dir = os.path.join(self.cache_dir, "glbs")
        else:
            # Local fallback with original Objaverse structure
            self.glb_base_dir = os.path.join(self.cache_dir, "hf-objaverse-v1", "glbs")
        
        self.do_print = do_print
        
        # Cache index for fast lookups
        self._cache_index: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._index_loaded = False
        
        # Objathor annotations (for getting all UIDs)
        self._objathor_uids: Optional[Set[str]] = None
        
    def _log(self, msg: str):
        """Log message if verbose mode enabled."""
        if self.do_print:
            print(f"[ObjaverseGLBManager] {msg}")
    
    def _get_glb_path_for_uid(self, uid: str) -> str:
        """
        Get the expected GLB path for a UID.
        
        Objaverse stores GLBs in: glbs/{uid[:2]}/{uid}.glb
        """
        return os.path.join(self.glb_base_dir, uid[:2], f"{uid}.glb")
    
    def _load_cache_index(self):
        """Scan the cache directory and build index."""
        if self._index_loaded:
            return
        
        with self._cache_lock:
            if self._index_loaded:
                return
            
            self._log("Building cache index...")
            
            # Directories to scan for GLBs
            dirs_to_scan = [self.glb_base_dir]
            
            # Also scan default objaverse download location
            default_objaverse_dir = os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs")
            if default_objaverse_dir != self.glb_base_dir and os.path.exists(default_objaverse_dir):
                dirs_to_scan.append(default_objaverse_dir)
            
            for glb_dir in dirs_to_scan:
                if os.path.exists(glb_dir):
                    self._log(f"Scanning {glb_dir}...")
                    for subdir in os.listdir(glb_dir):
                        subdir_path = os.path.join(glb_dir, subdir)
                        if os.path.isdir(subdir_path):
                            for filename in os.listdir(subdir_path):
                                if filename.endswith('.glb'):
                                    uid = filename[:-4]  # Remove .glb
                                    # Only add if not already in index (prefer our cache dir)
                                    if uid not in self._cache_index:
                                        full_path = os.path.join(subdir_path, filename)
                                        self._cache_index[uid] = full_path
            
            self._log(f"Found {len(self._cache_index)} cached GLBs")
            self._index_loaded = True
    
    def is_cached(self, uid: str) -> bool:
        """Check if a GLB is already cached."""
        self._load_cache_index()
        
        # First check index
        if uid in self._cache_index:
            return True
        
        # Double-check filesystem (in case index is stale)
        expected_path = self._get_glb_path_for_uid(uid)
        if os.path.exists(expected_path):
            with self._cache_lock:
                self._cache_index[uid] = expected_path
            return True
        
        return False
    
    def get_glb_path(
        self, 
        uid: str, 
        download_if_missing: bool = True
    ) -> Optional[str]:
        """
        Get the local path to a GLB file.
        
        Args:
            uid: Objaverse asset UID
            download_if_missing: If True, download if not cached
        
        Returns:
            Path to GLB file, or None if not available
        """
        self._load_cache_index()
        
        # Check cache
        if uid in self._cache_index:
            return self._cache_index[uid]
        
        # Check filesystem
        expected_path = self._get_glb_path_for_uid(uid)
        if os.path.exists(expected_path):
            with self._cache_lock:
                self._cache_index[uid] = expected_path
            return expected_path
        
        # Download if requested
        if download_if_missing and _HAVE_OBJAVERSE:
            self._log(f"Downloading GLB for {uid}...")
            try:
                # objaverse.load_objects returns {uid: local_path}
                paths = objaverse.load_objects([uid], download_processes=1)
                if uid in paths:
                    local_path = paths[uid]
                    with self._cache_lock:
                        self._cache_index[uid] = local_path
                    return local_path
            except Exception as e:
                self._log(f"Failed to download {uid}: {e}")
        
        return None
    
    def get_glb_paths_batch(
        self, 
        uids: List[str],
        download_if_missing: bool = True,
        download_processes: int = 4
    ) -> Dict[str, Optional[str]]:
        """
        Get local paths for multiple UIDs.
        
        Args:
            uids: List of Objaverse asset UIDs
            download_if_missing: If True, download missing assets
            download_processes: Number of parallel download processes
        
        Returns:
            Dict mapping uid -> local path (or None if unavailable)
        """
        self._load_cache_index()
        
        results = {}
        to_download = []
        
        # Check cache first
        for uid in uids:
            if uid in self._cache_index:
                results[uid] = self._cache_index[uid]
            else:
                expected_path = self._get_glb_path_for_uid(uid)
                if os.path.exists(expected_path):
                    with self._cache_lock:
                        self._cache_index[uid] = expected_path
                    results[uid] = expected_path
                else:
                    to_download.append(uid)
                    results[uid] = None
        
        # Download missing
        if to_download and download_if_missing and _HAVE_OBJAVERSE:
            self._log(f"Downloading {len(to_download)} GLBs...")
            try:
                paths = objaverse.load_objects(
                    to_download, 
                    download_processes=download_processes
                )
                for uid, path in paths.items():
                    with self._cache_lock:
                        self._cache_index[uid] = path
                    results[uid] = path
            except Exception as e:
                self._log(f"Batch download failed: {e}")
        
        return results
    
    def _load_objathor_uids(self) -> Set[str]:
        """Load all UIDs from objathor annotations.
        
        Uses PathConfig for unified path management.
        """
        if self._objathor_uids is not None:
            return self._objathor_uids
        
        # Try PathConfig first
        try:
            from path_config import PathConfig
            config = PathConfig.get_instance()
            objaverse_paths = config.get_objaverse_paths()
            annotations_path = objaverse_paths.get("annotations_path", "")
            if annotations_path and os.path.exists(annotations_path):
                # Use PathConfig paths
                base_dir = objaverse_paths.get("base_dir", "")
                assets_version = config.objaverse_assets_version
            else:
                # Fall through to legacy logic
                raise ImportError("PathConfig annotations_path not available")
        except ImportError:
            # Legacy fallback
            assets_version = os.environ.get("ASSETS_VERSION", "2023_09_23")
            
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
            
            annotations_path = os.path.join(base_dir, assets_version, "annotations.json.gz")
        
        uids = set()
        
        if os.path.exists(annotations_path):
            self._log(f"Loading UIDs from {annotations_path}...")
            try:
                import compress_json
                annotations = compress_json.load(annotations_path)
                uids.update(annotations.keys())
                self._log(f"Loaded {len(uids)} UIDs from objathor")
            except Exception as e:
                self._log(f"Failed to load annotations: {e}")
        
        # Also load thor annotations if available
        hd_version = os.environ.get("HD_BASE_VERSION", "2023_09_23")
        thor_annotations_path = os.path.join(
            base_dir, "holodeck", hd_version, "thor_object_data", "annotations.json.gz"
        )
        
        if os.path.exists(thor_annotations_path):
            try:
                import compress_json
                thor_annotations = compress_json.load(thor_annotations_path)
                uids.update(thor_annotations.keys())
                self._log(f"Total UIDs with thor: {len(uids)}")
            except Exception as e:
                self._log(f"Failed to load thor annotations: {e}")
        
        self._objathor_uids = uids
        return uids
    
    def predownload_all(
        self,
        num_processes: int = 8,
        batch_size: int = 100,
        progress_callback=None,
        copy_to_cache: bool = True
    ) -> Dict[str, str]:
        """
        Pre-download all GLBs from objathor annotations.
        
        This is useful before RL training to ensure all assets are cached locally.
        
        Args:
            num_processes: Number of parallel download processes
            batch_size: Number of UIDs to download per batch
            progress_callback: Optional callback(downloaded, total) for progress
            copy_to_cache: If True and using basic objaverse API, copy downloaded files to cache directory
        
        Returns:
            Dict mapping uid -> local path
        """
        if not _HAVE_OBJAVERSE:
            raise RuntimeError("objaverse package required for pre-downloading")
        
        all_uids = self._load_objathor_uids()
        
        if not all_uids:
            self._log("No UIDs found to download")
            return {}
        
        self._load_cache_index()
        
        # Filter out already cached
        to_download = [uid for uid in all_uids if uid not in self._cache_index]
        
        self._log(f"Need to download {len(to_download)} / {len(all_uids)} GLBs")
        
        if not to_download:
            self._log("All GLBs already cached!")
            return {uid: self._cache_index[uid] for uid in all_uids if uid in self._cache_index}
        
        all_paths = {}
        downloaded = 0
        
        # Use objaverse.xl API if available (supports custom download_dir)
        if _HAVE_OBJAVERSE_XL:
            self._log(f"Using objaverse.xl API to download directly to {self.glb_base_dir}")
            
            # Get all annotations from Sketchfab (Objaverse v1)
            try:
                annotations_df = SketchfabDownloader.get_annotations()
                
                # Filter to only UIDs we need
                to_download_set = set(to_download)
                filtered_df = annotations_df[annotations_df['uid'].isin(to_download_set)]
                
                self._log(f"Found {len(filtered_df)} objects in Sketchfab annotations")
                
                # Download in batches
                for i in range(0, len(filtered_df), batch_size):
                    batch_df = filtered_df.iloc[i:i + batch_size]
                    
                    try:
                        # Use custom download_dir - files go to {download_dir}/hf-objaverse-v1/glbs/
                        paths = SketchfabDownloader.download_objects(
                            objects=batch_df,
                            download_dir=self.cache_dir,  # This will create glbs/ inside
                            processes=num_processes
                        )
                        
                        # Map fileIdentifier back to uid and update paths
                        for file_id, path in paths.items():
                            # Extract uid from file_id or path
                            uid = os.path.basename(path).replace('.glb', '') if path else None
                            if uid and path:
                                with self._cache_lock:
                                    self._cache_index[uid] = path
                                all_paths[uid] = path
                        
                        downloaded += len(batch_df)
                        
                        if progress_callback:
                            progress_callback(downloaded, len(to_download))
                        else:
                            self._log(f"Progress: {downloaded}/{len(to_download)} ({100*downloaded/len(to_download):.1f}%)")
                            
                    except Exception as e:
                        self._log(f"Batch {i//batch_size} failed: {e}")
                        
            except Exception as e:
                self._log(f"objaverse.xl failed: {e}, falling back to basic API")
                # Fall through to basic API
        
        # Fallback to basic objaverse API (downloads to ~/.objaverse)
        remaining = [uid for uid in to_download if uid not in all_paths]
        if remaining:
            self._log(f"Using basic objaverse API for {len(remaining)} remaining objects")
            import shutil
            
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                
                try:
                    # objaverse.load_objects downloads to ~/.objaverse/hf-objaverse-v1/glbs/
                    paths = objaverse.load_objects(
                        batch,
                        download_processes=num_processes
                    )
                    
                    for uid, src_path in paths.items():
                        # Copy to our cache directory if needed
                        if copy_to_cache and src_path and os.path.exists(src_path):
                            dst_path = self._get_glb_path_for_uid(uid)
                            
                            # Only copy if not already in our cache dir
                            if not dst_path.startswith(os.path.dirname(src_path)):
                                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                                try:
                                    shutil.copy2(src_path, dst_path)
                                    final_path = dst_path
                                except Exception as e:
                                    self._log(f"Failed to copy {uid}: {e}")
                                    final_path = src_path
                            else:
                                final_path = src_path
                        else:
                            final_path = src_path
                        
                        with self._cache_lock:
                            self._cache_index[uid] = final_path
                        all_paths[uid] = final_path
                    
                    downloaded += len(batch)
                    
                    if progress_callback:
                        progress_callback(downloaded, len(to_download))
                    else:
                        self._log(f"Progress: {downloaded}/{len(to_download)} ({100*downloaded/len(to_download):.1f}%)")
                        
                except Exception as e:
                    self._log(f"Batch {i//batch_size} failed: {e}")
        
        self._log(f"Downloaded {len(all_paths)} GLBs")
        
        # Include already cached
        for uid in all_uids:
            if uid in self._cache_index and uid not in all_paths:
                all_paths[uid] = self._cache_index[uid]
        
        return all_paths
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        self._load_cache_index()
        all_uids = self._load_objathor_uids()
        
        cached = len(self._cache_index)
        total = len(all_uids)
        
        # Calculate disk usage
        disk_usage = 0
        for path in self._cache_index.values():
            if os.path.exists(path):
                disk_usage += os.path.getsize(path)
        
        return {
            "cached_count": cached,
            "total_count": total,
            "coverage_percent": 100.0 * cached / total if total > 0 else 0,
            "disk_usage_gb": disk_usage / (1024**3),
            "cache_dir": self.glb_base_dir,
        }


# Global singleton for convenience
_global_manager: Optional[ObjaverseGLBManager] = None
_global_lock = threading.Lock()


def get_global_manager(do_print: bool = False) -> ObjaverseGLBManager:
    """Get the global GLB manager singleton."""
    global _global_manager
    
    if _global_manager is None:
        with _global_lock:
            if _global_manager is None:
                _global_manager = ObjaverseGLBManager(do_print=do_print)
    
    return _global_manager


def get_objaverse_glb_path(uid: str, download_if_missing: bool = True) -> Optional[str]:
    """
    Convenience function to get GLB path for a UID.
    
    Args:
        uid: Objaverse asset UID
        download_if_missing: Download if not cached
    
    Returns:
        Local path to GLB, or None
    """
    return get_global_manager().get_glb_path(uid, download_if_missing)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Objaverse GLB Manager - Download and manage GLB assets")
    parser.add_argument("--predownload", action="store_true", help="Pre-download all GLBs from objathor annotations")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--processes", type=int, default=8, help="Number of parallel download processes (default: 8)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for downloading (default: 100)")
    parser.add_argument("--uid", type=str, help="Get path for specific UID")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory")
    parser.add_argument("--copy-to-cache", action="store_true", default=True, 
                        help="Copy downloaded files to cache directory (default: True)")
    parser.add_argument("--no-copy", action="store_true", help="Don't copy files, just use objaverse default location")
    
    args = parser.parse_args()
    
    manager = ObjaverseGLBManager(cache_dir=args.cache_dir, do_print=True)
    
    if args.stats:
        stats = manager.get_cache_stats()
        print("\n=== Cache Statistics ===")
        print(f"Cached: {stats['cached_count']} / {stats['total_count']} ({stats['coverage_percent']:.1f}%)")
        print(f"Disk usage: {stats['disk_usage_gb']:.2f} GB")
        print(f"Cache dir: {stats['cache_dir']}")
    
    elif args.predownload:
        print("\n=== Pre-downloading all GLBs ===")
        print(f"Processes: {args.processes}, Batch size: {args.batch_size}")
        copy_to_cache = not args.no_copy
        print(f"Copy to cache: {copy_to_cache}")
        print("This may take several hours depending on network speed...\n")
        
        # Show initial stats
        initial_stats = manager.get_cache_stats()
        print(f"Initial cache: {initial_stats['cached_count']} / {initial_stats['total_count']} GLBs")
        print(f"Coverage: {initial_stats['coverage_percent']:.1f}%\n")
        
        # Pre-download
        paths = manager.predownload_all(
            num_processes=args.processes,
            batch_size=args.batch_size,
            copy_to_cache=copy_to_cache
        )
        
        # Show final stats
        final_stats = manager.get_cache_stats()
        print(f"\n=== Download Complete ===")
        print(f"Final cache: {final_stats['cached_count']} / {final_stats['total_count']} GLBs")
        print(f"Coverage: {final_stats['coverage_percent']:.1f}%")
        print(f"Disk usage: {final_stats['disk_usage_gb']:.2f} GB")
        print(f"Cache dir: {final_stats['cache_dir']}")
    
    elif args.uid:
        path = manager.get_glb_path(args.uid)
        if path:
            print(f"GLB path: {path}")
        else:
            print(f"GLB not found for UID: {args.uid}")
    
    else:
        parser.print_help()
