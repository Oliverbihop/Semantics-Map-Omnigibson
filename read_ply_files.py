"""Read and analyze PLY point cloud files.

This script loads three PLY files:
- integrated_slam_map_instance.ply (instance segmentation)
- integrated_slam_map_rgb.ply (RGB colors)
- integrated_slam_map_semantic.ply (semantic segmentation)

Processes them through the same pipeline as replay_saved_frames.py:
- Extract instance IDs and RGB colors
- Compute 3D bounding boxes per instance
- Build occupancy grid
- Detect areas using Hough lines
- Create top-view RGB image
- Save objects and areas to JSON

Provides statistics and visualization options.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Try different PLY reading libraries
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False

# Try rerun for visualization
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

# Try cv2 for image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def read_ply_open3d(file_path: Path) -> Optional[Dict]:
    """Read PLY file using Open3D library.
    
    Args:
        file_path: Path to PLY file
        
    Returns:
        Dictionary with 'points', 'colors', and other available properties
    """
    if not OPEN3D_AVAILABLE:
        return None
        
    try:
        pcd = o3d.io.read_point_cloud(str(file_path))
        
        if len(pcd.points) == 0:
            print(f"  Warning: {file_path.name} appears to be empty")
            return None
            
        data = {
            'points': np.asarray(pcd.points),
            'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
            'normals': np.asarray(pcd.normals) if pcd.has_normals() else None,
        }
        
        return data
        
    except Exception as e:
        print(f"  Error reading {file_path.name} with Open3D: {e}")
        return None


def read_ply_plyfile(file_path: Path) -> Optional[Dict]:
    """Read PLY file using plyfile library.
    
    Args:
        file_path: Path to PLY file
        
    Returns:
        Dictionary with 'points', 'colors', and other available properties
    """
    if not PLYFILE_AVAILABLE:
        return None
        
    try:
        plydata = PlyData.read(str(file_path))
        
        if 'vertex' not in plydata.elements:
            print(f"  Warning: {file_path.name} does not contain vertex data")
            return None
            
        vertex = plydata['vertex']
        num_points = len(vertex)
        
        if num_points == 0:
            print(f"  Warning: {file_path.name} appears to be empty")
            return None
        
        data = {}
        
        if all(prop in vertex.dtype.names for prop in ['x', 'y', 'z']):
            data['points'] = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        else:
            print(f"  Warning: {file_path.name} missing x, y, z coordinates")
            return None
        
        if all(prop in vertex.dtype.names for prop in ['red', 'green', 'blue']):
            colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']])
            if colors.max() > 1.0:
                colors = colors.astype(np.float32) / 255.0
            data['colors'] = colors
        elif all(prop in vertex.dtype.names for prop in ['r', 'g', 'b']):
            colors = np.column_stack([vertex['r'], vertex['g'], vertex['b']])
            if colors.max() > 1.0:
                colors = colors.astype(np.float32) / 255.0
            data['colors'] = colors
        
        if 'label' in vertex.dtype.names:
            data['label'] = vertex['label']
        if 'semantic_id' in vertex.dtype.names:
            data['semantic_id'] = vertex['semantic_id']
        if 'instance_id' in vertex.dtype.names:
            data['instance_id'] = vertex['instance_id']
        if 'nx' in vertex.dtype.names and 'ny' in vertex.dtype.names and 'nz' in vertex.dtype.names:
            data['normals'] = np.column_stack([vertex['nx'], vertex['ny'], vertex['nz']])
        
        print(f"  Available properties: {vertex.dtype.names}")
        
        return data
        
    except Exception as e:
        print(f"  Error reading {file_path.name} with plyfile: {e}")
        return None


def read_ply_file(file_path: Path) -> Optional[Dict]:
    """Read PLY file using available library.
    
    Args:
        file_path: Path to PLY file
        
    Returns:
        Dictionary with point cloud data
    """
    if not file_path.exists():
        print(f"  Error: File not found: {file_path}")
        return None
    
    print(f"\nReading: {file_path.name}")
    
    if OPEN3D_AVAILABLE:
        data = read_ply_open3d(file_path)
        if data is not None:
            print(f"  Successfully read with Open3D")
            return data
    
    if PLYFILE_AVAILABLE:
        data = read_ply_plyfile(file_path)
        if data is not None:
            print(f"  Successfully read with plyfile")
            return data
    
    print(f"  Error: No suitable library available to read PLY files")
    print(f"  Install one of: pip install open3d  or  pip install plyfile")
    return None


def extract_labels_from_colors(colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract label IDs from RGB colors.
    
    In semantic/instance PLY files, labels are often encoded as RGB colors.
    This function converts unique colors to label IDs.
    
    Args:
        colors: (N, 3) array of RGB colors (0-1 or 0-255 range)
        
    Returns:
        label_ids: (N,) array of label IDs for each point
        unique_colors: (M, 3) array of unique colors (label colors)
    """
    if colors.max() <= 1.0:
        colors_uint8 = (colors * 255).astype(np.uint8)
    else:
        colors_uint8 = colors.astype(np.uint8)
    
    unique_colors, label_ids = np.unique(colors_uint8, axis=0, return_inverse=True)
    
    return label_ids, unique_colors


def detect_floor_from_labels(points: np.ndarray, colors: np.ndarray, 
                             method: str = 'largest', min_floor_fraction: float = 0.3) -> Tuple[np.ndarray, int, Dict]:
    """Detect floor points using semantic/instance labels.
    
    The floor is identified as points sharing the same semantic/instance label.
    Since floor is typically the largest connected surface, we use the largest label
    or the label with the lowest Z coordinates.
    
    Args:
        points: (N, 3) array of 3D points
        colors: (N, 3) array of RGB colors (labels encoded as colors)
        method: 'largest' (by point count) or 'lowest' (by Z position)
        min_floor_fraction: Minimum fraction of points required to be considered floor
        
    Returns:
        floor_mask: (N,) boolean array indicating floor points
        floor_label_id: ID of the floor label
        floor_info: Dictionary with floor statistics
    """
    # Extract labels from colors
    label_ids, unique_colors = extract_labels_from_colors(colors)
    unique_label_ids = np.unique(label_ids)
    
    print(f"\n  Found {len(unique_label_ids)} unique labels from colors")
    
    # Count points per label
    label_counts = {lid: np.sum(label_ids == lid) for lid in unique_label_ids}
    
    if method == 'largest':
        # Find largest label by point count
        floor_label_id = max(label_counts, key=label_counts.get)
        floor_count = label_counts[floor_label_id]
        floor_fraction = floor_count / len(points)
        
        print(f"  Using largest label: {floor_label_id} with {floor_count:,} points ({floor_fraction*100:.1f}%)")
        
    elif method == 'lowest':
        # Find label with lowest average Z position (likely floor)
        label_z_means = {}
        for lid in unique_label_ids:
            mask = label_ids == lid
            label_z_means[lid] = np.mean(points[mask, 2])
        
        floor_label_id = min(label_z_means, key=label_z_means.get)
        floor_count = label_counts[floor_label_id]
        floor_fraction = floor_count / len(points)
        
        print(f"  Using lowest Z label: {floor_label_id} with {floor_count:,} points ({floor_fraction*100:.1f}%)")
        print(f"    Average Z: {label_z_means[floor_label_id]:.3f}")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Check if floor fraction is reasonable
    if floor_fraction < min_floor_fraction:
        print(f"  Warning: Floor fraction ({floor_fraction*100:.1f}%) is below threshold ({min_floor_fraction*100:.1f}%)")
    
    # Create floor mask
    floor_mask = label_ids == floor_label_id
    floor_points = points[floor_mask]
    
    floor_info = {
        'label_id': floor_label_id,
        'label_color': unique_colors[floor_label_id],
        'point_count': floor_count,
        'fraction': floor_fraction,
        'z_min': floor_points[:, 2].min(),
        'z_max': floor_points[:, 2].max(),
        'z_mean': floor_points[:, 2].mean(),
        'z_std': floor_points[:, 2].std(),
        'z_range': floor_points[:, 2].max() - floor_points[:, 2].min(),
    }
    
    print(f"\n  Floor statistics:")
    print(f"    Z range: [{floor_info['z_min']:.3f}, {floor_info['z_max']:.3f}]")
    print(f"    Z mean: {floor_info['z_mean']:.3f} ± {floor_info['z_std']:.3f}")
    print(f"    Z spread: {floor_info['z_range']:.3f} (distortion)")
    
    return floor_mask, floor_label_id, floor_info


def detect_floor_plane_ransac(points: np.ndarray, z_percentile: float = 0.40, n_samples: int = 20000, n_iterations: int = 7):
    """Detect floor plane using iterative PCA refinement on points in the lower Z range."""
    if len(points) == 0:
        return None, None, None
    
    z_threshold = np.percentile(points[:, 2], z_percentile * 100)
    floor_candidates = points[points[:, 2] <= z_threshold]
    
    if len(floor_candidates) < 100:
        z_threshold = np.percentile(points[:, 2], 50)
        floor_candidates = points[points[:, 2] <= z_threshold]
    
    print(f"  Using {len(floor_candidates):,} candidate points (Z <= {z_threshold:.3f}) for floor detection")
    
    plane_normal = None
    plane_point = None
    current_candidates = floor_candidates.copy()
    
    for iteration in range(n_iterations):
        if len(current_candidates) < 3:
            break
        
        if len(current_candidates) > n_samples:
            indices = np.random.choice(len(current_candidates), n_samples, replace=False)
            sampled = current_candidates[indices]
        else:
            sampled = current_candidates
        
        centroid = np.mean(sampled, axis=0)
        centered = sampled - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
        if normal[2] < 0:
            normal = -normal
        
        plane_point = centroid
        plane_normal = normal
        
        if iteration < n_iterations - 1:
            candidate_centered = current_candidates - plane_point
            distances = np.abs(np.dot(candidate_centered, plane_normal))
            std_dev = np.std(distances)
            median_dist = np.median(distances)
            threshold = 3.5 * std_dev + median_dist * 1.8
            inliers_mask = distances < threshold
            current_candidates = current_candidates[inliers_mask]
            print(f"  Iteration {iteration + 1}: {len(current_candidates):,} inliers remaining")
    
    if plane_normal is None:
        return None, None, None
    
    candidate_indices = np.where(points[:, 2] <= z_threshold + 0.5)[0]
    if len(candidate_indices) > 0:
        candidate_points = points[candidate_indices]
        candidate_centered = candidate_points - plane_point
        distances = np.abs(np.dot(candidate_centered, plane_normal))
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        median_dist = np.median(distances)
        threshold = max(q3 + 3.0 * iqr, median_dist * 2.5)
        full_inliers = np.zeros(len(points), dtype=bool)
        full_inliers[candidate_indices] = distances < threshold
    else:
        full_inliers = np.zeros(len(points), dtype=bool)
    
    return plane_normal, plane_point, full_inliers


def detect_floor_plane_from_labels(points: np.ndarray, floor_mask: np.ndarray, 
                                  use_robust: bool = True) -> tuple:
    """Fit a plane to floor points using PCA (handles SLAM distortions)."""
    floor_points = points[floor_mask]
    
    if len(floor_points) < 3:
        print("  Error: Not enough floor points to fit plane")
        return None, None, None
    
    print(f"  Fitting plane to {len(floor_points):,} floor points (detected from labels)...")
    
    current_points = floor_points.copy()
    
    if use_robust:
        for iteration in range(3):
            centroid = np.mean(current_points, axis=0)
            centered = current_points - centroid
            
            if len(centered) < 3:
                break
            
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normal = eigenvectors[:, 0]
            if normal[2] < 0:
                normal = -normal
            
            distances = np.abs(np.dot(centered, normal))
            
            if iteration < 2:
                threshold = np.percentile(distances, 95)
                inliers = distances < threshold
                current_points = current_points[inliers]
                print(f"    Iteration {iteration + 1}: {len(current_points):,} inliers remaining")
        
        floor_points_final = current_points
        centroid = np.mean(floor_points_final, axis=0)
        centered = floor_points_final - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
        if normal[2] < 0:
            normal = -normal
    else:
        centroid = np.mean(floor_points, axis=0)
        centered = floor_points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
        if normal[2] < 0:
            normal = -normal
    
    floor_centered = floor_points - centroid
    distances = np.abs(np.dot(floor_centered, normal))
    plane_fit_error = np.sqrt(np.mean(distances**2))
    
    print(f"  Plane fit:")
    print(f"    Normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    print(f"    Point: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"    RMS error: {plane_fit_error:.4f} m")
    print(f"    Max error: {distances.max():.4f} m")
    
    return normal, centroid, floor_mask


def detect_floor_from_color(points: np.ndarray, rgb_colors: np.ndarray, grid: np.ndarray,
                            grid_resolution: float, min_x: float, min_y: float,
                            floor_z_min: float, floor_z_max: float, 
                            color_tolerance: int = 30) -> tuple:
    """Detect floor by finding dominant RGB color in floor grid cells.
    
    Args:
        points: (N, 3) array of 3D points
        rgb_colors: (N, 3) array of RGB colors (0-255)
        grid: (H, W) occupancy grid (0=floor, 1=objects, 2=walls, 3=unknown)
        grid_resolution: Resolution of the grid in meters
        min_x, min_y: Grid origin in world coordinates
        floor_z_min, floor_z_max: Z range for floor detection
        color_tolerance: RGB color tolerance for matching (default 30)
        
    Returns:
        floor_color: (3,) array of dominant floor RGB color, or None if not found
        floor_mask: (N,) boolean array of floor points, or None if not found
    """
    if len(points) == 0 or len(rgb_colors) == 0:
        return None, None
    
    floor_cell_colors = []
    
    grid_height, grid_width = grid.shape
    
    for i, point in enumerate(points):
        x, y, z = point
        if not (floor_z_min <= z <= floor_z_max):
            continue
        
        grid_x = int((x - min_x) / grid_resolution)
        grid_y = int((y - min_y) / grid_resolution)
        
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            if grid[grid_y, grid_x] == 0:  # Floor cell
                floor_cell_colors.append(rgb_colors[i])
    
    if len(floor_cell_colors) == 0:
        print("  No points found in floor grid cells for color-based detection")
        return None, None
    
    floor_cell_colors = np.array(floor_cell_colors)
    
    floor_color = np.median(floor_cell_colors, axis=0).astype(np.uint8)
    
    quantized_colors = (floor_cell_colors // 10) * 10
    unique_colors, counts = np.unique(quantized_colors, axis=0, return_counts=True)
    dominant_idx = np.argmax(counts)
    dominant_quantized = unique_colors[dominant_idx]
    
    matching_mask = np.all((quantized_colors // 10) == (dominant_quantized // 10), axis=1)
    if np.sum(matching_mask) > 0:
        floor_color = np.median(floor_cell_colors[matching_mask], axis=0).astype(np.uint8)
    
    dominant_count = counts[dominant_idx]
    print(f"  Found {len(unique_colors)} unique color clusters in floor grid cells")
    print(f"  Dominant floor color: RGB({floor_color[0]}, {floor_color[1]}, {floor_color[2]}) with {dominant_count:,}/{len(floor_cell_colors):,} points ({dominant_count/len(floor_cell_colors)*100:.1f}%)")
    
    color_diff = np.abs(rgb_colors.astype(int) - floor_color.astype(int))
    color_match = np.all(color_diff <= color_tolerance, axis=1)
    
    z_in_range = (points[:, 2] >= floor_z_min) & (points[:, 2] <= floor_z_max)
    floor_mask = color_match & z_in_range
    
    floor_count = np.sum(floor_mask)
    print(f"  Color-based floor mask: {floor_count:,} points match floor color (tolerance: {color_tolerance})")
    
    return floor_color, floor_mask


def detect_floor_label_from_grid(points: np.ndarray, labels: np.ndarray, grid: np.ndarray,
                                 grid_resolution: float, min_x: float, min_y: float,
                                 floor_z_min: float, floor_z_max: float) -> tuple:
    """Detect floor label by finding dominant label in floor grid cells."""
    if len(points) == 0 or len(labels) == 0:
        return None, None
    
    floor_cell_points = []
    floor_cell_labels = []
    
    grid_height, grid_width = grid.shape
    
    for i, point in enumerate(points):
        x, y, z = point
        if not (floor_z_min <= z <= floor_z_max):
            continue
        
        grid_x = int((x - min_x) / grid_resolution)
        grid_y = int((y - min_y) / grid_resolution)
        
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            if grid[grid_y, grid_x] == 0:  # Floor cell
                floor_cell_points.append(i)
                floor_cell_labels.append(labels[i])
    
    if len(floor_cell_labels) == 0:
        print("  No points found in floor grid cells")
        return None, None
    
    floor_cell_labels = np.array(floor_cell_labels)
    unique_labels, counts = np.unique(floor_cell_labels, return_counts=True)
    dominant_idx = np.argmax(counts)
    floor_label_id = unique_labels[dominant_idx]
    dominant_count = counts[dominant_idx]
    
    print(f"  Found {len(unique_labels)} unique labels in floor grid cells")
    print(f"  Dominant floor label: {floor_label_id} with {dominant_count:,}/{len(floor_cell_labels):,} points ({dominant_count/len(floor_cell_labels)*100:.1f}%)")
    
    floor_mask = labels == floor_label_id
    
    return floor_label_id, floor_mask


def detect_wall_labels_from_semantic(points: np.ndarray, labels: np.ndarray, 
                                     floor_z_min: float, floor_z_max: float,
                                     min_wall_height: float = 0.5, min_points_per_label: int = 100) -> tuple:
    """Detect wall labels by finding semantic labels with consistent colors that form wall-like structures.
    
    Walls typically have:
    - Consistent semantic color (same label across all wall points)
    - Tall vertical structures (high Z-range)
    - Many points forming vertical planes
    
    Args:
        points: (N, 3) array of 3D points
        labels: (N,) array of semantic/instance label IDs for each point
        floor_z_min, floor_z_max: Z range for wall detection
        min_wall_height: Minimum height (Z-range) for a label to be considered a wall (default 0.5m)
        min_points_per_label: Minimum number of points required for a label to be considered (default 100)
        
    Returns:
        wall_label_ids: List of label IDs that are walls
        wall_mask: (N,) boolean array of wall points
        wall_label_stats: Dictionary with statistics for each wall label
    """
    if len(points) == 0 or len(labels) == 0:
        return [], None, {}
    
    z_mask = (points[:, 2] >= floor_z_min) & (points[:, 2] <= floor_z_max)
    points_in_range = points[z_mask]
    labels_in_range = labels[z_mask]
    
    if len(points_in_range) == 0:
        print("  No points in wall detection Z range")
        return [], None, {}
    
    unique_labels, label_indices, label_counts = np.unique(
        labels_in_range, return_inverse=True, return_counts=True
    )
    
    wall_label_ids = []
    wall_label_stats = {}
    
    print(f"  Analyzing {len(unique_labels)} unique semantic labels for wall characteristics...")
    
    for label_id, count in zip(unique_labels, label_counts):
        if count < min_points_per_label:
            continue
        
        label_mask = labels_in_range == label_id
        label_points = points_in_range[label_mask]
        
        if len(label_points) == 0:
            continue
        
        z_min = np.min(label_points[:, 2])
        z_max = np.max(label_points[:, 2])
        z_range = z_max - z_min
        z_mean = np.mean(label_points[:, 2])
        
        x_range = np.max(label_points[:, 0]) - np.min(label_points[:, 0])
        y_range = np.max(label_points[:, 1]) - np.min(label_points[:, 1])
        xy_spread = max(x_range, y_range)
        
        is_tall = z_range >= min_wall_height
        
        has_horizontal_extent = xy_spread >= 0.5
        
        is_very_tall = z_range >= min_wall_height * 1.5
        point_count_ok = (is_very_tall and count >= min_points_per_label) or \
                        (count >= min_points_per_label * 1.5)
        
        is_wall_like = is_tall and has_horizontal_extent and point_count_ok
        
        if is_wall_like:
            wall_label_ids.append(label_id)
            wall_label_stats[label_id] = {
                'count': int(count),
                'z_min': float(z_min),
                'z_max': float(z_max),
                'z_range': float(z_range),
                'z_mean': float(z_mean),
                'x_range': float(x_range),
                'y_range': float(y_range),
                'xy_spread': float(xy_spread)
            }
            print(f"    Wall label {label_id}: {count:,} points, Z-range: {z_range:.2f}m, XY-spread: {xy_spread:.2f}m")
    
    if not wall_label_ids:
        print("  No wall labels found based on semantic color consistency")
        return [], None, {}
    
    wall_mask = np.isin(labels, wall_label_ids)
    
    print(f"\n  Wall detection from semantic colors complete:")
    print(f"    Found {len(wall_label_ids)} wall label(s)")
    print(f"    Total wall points: {np.sum(wall_mask):,}/{len(points):,} ({np.sum(wall_mask)/len(points)*100:.1f}%)")
    
    return wall_label_ids, wall_mask, wall_label_stats


def detect_wall_labels_from_grid(points: np.ndarray, labels: np.ndarray, grid: np.ndarray,
                                grid_resolution: float, min_x: float, min_y: float,
                                floor_z_min: float, floor_z_max: float, min_dominance_ratio: float = 0.5) -> tuple:
    """Detect wall labels by finding dominant labels in wall grid cells."""
    if len(points) == 0 or len(labels) == 0:
        return [], None, {}
    
    wall_cell_points = []
    wall_cell_labels = []
    
    grid_height, grid_width = grid.shape
    
    for i, point in enumerate(points):
        x, y, z = point
        if not (floor_z_min <= z <= floor_z_max):
            continue
        
        grid_x = int((x - min_x) / grid_resolution)
        grid_y = int((y - min_y) / grid_resolution)
        
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            if grid[grid_y, grid_x] == 2:  # Wall cell
                wall_cell_points.append(i)
                wall_cell_labels.append(labels[i])
    
    if len(wall_cell_labels) == 0:
        print("  No points found in wall grid cells")
        return [], None, {}
    
    wall_cell_labels = np.array(wall_cell_labels)
    unique_labels, counts = np.unique(wall_cell_labels, return_counts=True)
    
    sorted_indices = np.argsort(counts)[::-1]
    unique_labels = unique_labels[sorted_indices]
    counts = counts[sorted_indices]
    
    print(f"  Found {len(unique_labels)} unique labels in wall grid cells")
    print(f"  Total points in wall cells: {len(wall_cell_labels):,}")
    
    wall_label_ids = []
    wall_label_stats = {}
    total_wall_points = len(wall_cell_labels)
    
    for label_id, count in zip(unique_labels, counts):
        ratio = count / total_wall_points
        if ratio >= min_dominance_ratio or (len(wall_label_ids) < 5 and ratio >= 0.05):
            wall_label_ids.append(label_id)
            wall_label_stats[label_id] = {
                'count': int(count),
                'ratio': float(ratio),
                'percentage': float(ratio * 100)
            }
            print(f"    Wall label {label_id}: {count:,} points ({ratio*100:.1f}%)")
    
    if not wall_label_ids:
        wall_label_ids = unique_labels[:min(3, len(unique_labels))].tolist()
        for label_id in wall_label_ids:
            idx = np.where(unique_labels == label_id)[0][0]
            count = counts[idx]
            wall_label_stats[label_id] = {
                'count': int(count),
                'ratio': float(count / total_wall_points),
                'percentage': float(count / total_wall_points * 100)
            }
        print(f"  Using top {len(wall_label_ids)} labels as wall labels")
    
    wall_mask = np.isin(labels, wall_label_ids)
    
    print(f"  Total wall points identified: {np.sum(wall_mask):,}/{len(points):,} ({np.sum(wall_mask)/len(points)*100:.1f}%)")
    
    return wall_label_ids, wall_mask, wall_label_stats


def align_floor_to_z0(points: np.ndarray, labels: np.ndarray = None, 
                      floor_mask: np.ndarray = None):
    """Align point cloud so the floor plane is at z=0 and horizontal.
    
    Uses iterative refinement to get better floor alignment.
    If labels and floor_mask are provided, uses label-based floor detection.
    
    Args:
        points: (N, 3) array of 3D points
        labels: Optional (N,) array of label IDs (for label-based floor detection)
        floor_mask: Optional (N,) boolean array of floor points (from label detection)
        
    Returns:
        transformed_points: (N, 3) transformed points with floor at z=0
        rotation_matrix: (3, 3) rotation matrix applied
        translation: (3,) translation applied after rotation
    """
    if len(points) == 0:
        return points, np.eye(3), np.zeros(3)
    
    if floor_mask is not None and labels is not None and np.any(floor_mask):
        print("  Using label-based floor detection")
        plane_normal, plane_point, inliers = detect_floor_plane_from_labels(
            points, floor_mask, use_robust=True
        )
    else:
        plane_normal, plane_point, inliers = detect_floor_plane_ransac(points)
    
    if plane_normal is None:
        print("Warning: Could not detect floor plane, skipping alignment")
        return points, np.eye(3), np.zeros(3)
    
    print(f"\n  Floor plane detected:")
    print(f"    Normal vector: [{plane_normal[0]:.4f}, {plane_normal[1]:.4f}, {plane_normal[2]:.4f}]")
    print(f"    Point on plane: [{plane_point[0]:.4f}, {plane_point[1]:.4f}, {plane_point[2]:.4f}]")
    print(f"    Inlier points: {np.sum(inliers):,}/{len(points):,} ({100*np.sum(inliers)/len(points):.1f}%)")
    
    current_z_align = abs(plane_normal[2])
    print(f"    Current Z-alignment: {current_z_align:.4f} (1.0 = perfect)")
    
    target_normal = np.array([0.0, 0.0, 1.0])
    
    v = np.cross(plane_normal, target_normal)
    s = np.linalg.norm(v)
    c = np.dot(plane_normal, target_normal)
    
    if s < 1e-6:
        rotation_matrix = np.eye(3)
        print("  No rotation needed - already aligned")
    else:
        if s < 1e-10:
            rotation_matrix = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        
        angle_rad = np.arccos(np.clip(c, -1, 1))
        angle_deg = np.degrees(angle_rad)
        print(f"  Rotation angle: {angle_deg:.2f} degrees")
    
    rotated_points = (rotation_matrix @ points.T).T
    
    if np.any(inliers):
        floor_points_z = rotated_points[inliers, 2]
        floor_z_ref = np.percentile(floor_points_z, 15)
        print(f"  Floor Z reference (15th percentile for distorted floors): {floor_z_ref:.4f}")
        print(f"    Floor Z range: min={np.min(floor_points_z):.4f}, median={np.median(floor_points_z):.4f}, max={np.max(floor_points_z):.4f}")
    else:
        floor_z_ref = np.min(rotated_points[:, 2])
        print(f"  Floor Z reference (min, no inliers): {floor_z_ref:.4f}")
    
    translation = np.array([0.0, 0.0, -floor_z_ref])
    transformed_points = rotated_points + translation
    
    if np.any(inliers):
        aligned_floor_z = transformed_points[inliers, 2]
        print(f"  After alignment:")
        print(f"    Floor Z - min: {np.min(aligned_floor_z):.4f}, median: {np.median(aligned_floor_z):.4f}, max: {np.max(aligned_floor_z):.4f}")
        print(f"    Floor Z std dev: {np.std(aligned_floor_z):.4f}")
    
    return transformed_points, rotation_matrix, translation


def print_statistics(data: Dict, name: str):
    """Print statistics about loaded point cloud data.
    
    Args:
        data: Dictionary containing point cloud data
        name: Name of the point cloud (for display)
    """
    print(f"\n{'='*60}")
    print(f"Statistics for {name}")
    print(f"{'='*60}")
    
    if 'points' in data:
        points = data['points']
        print(f"Number of points: {len(points):,}")
        print(f"Point shape: {points.shape}")
        
        print(f"\nPosition statistics:")
        print(f"  X: min={points[:, 0].min():.3f}, max={points[:, 0].max():.3f}, mean={points[:, 0].mean():.3f}")
        print(f"  Y: min={points[:, 1].min():.3f}, max={points[:, 1].max():.3f}, mean={points[:, 1].mean():.3f}")
        print(f"  Z: min={points[:, 2].min():.3f}, max={points[:, 2].max():.3f}, mean={points[:, 2].mean():.3f}")
    
    if 'colors' in data and data['colors'] is not None:
        colors = data['colors']
        print(f"\nColor statistics:")
        print(f"  Color shape: {colors.shape}")
        print(f"  Color range: [{colors.min():.3f}, {colors.max():.3f}]")
        print(f"  Has colors: Yes")
    else:
        print(f"\nColor statistics: No colors available")
    
    if 'label' in data:
        labels = data['label']
        unique_labels = np.unique(labels)
        print(f"\nLabel statistics:")
        print(f"  Unique labels: {len(unique_labels)}")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
    
    if 'semantic_id' in data:
        semantic_ids = data['semantic_id']
        unique_semantic = np.unique(semantic_ids)
        print(f"\nSemantic ID statistics:")
        print(f"  Unique semantic IDs: {len(unique_semantic)}")
        print(f"  ID range: [{semantic_ids.min()}, {semantic_ids.max()}]")
    
    if 'instance_id' in data:
        instance_ids = data['instance_id']
        unique_instances = np.unique(instance_ids)
        print(f"\nInstance ID statistics:")
        print(f"  Unique instance IDs: {len(unique_instances)}")
        print(f"  ID range: [{instance_ids.min()}, {instance_ids.max()}]")
        # Show top 10 most common instances
        if len(unique_instances) > 0:
            counts = {uid: np.sum(instance_ids == uid) for uid in unique_instances[:10]}
            print(f"  Top 10 instance sizes: {dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))}")
    
    if 'normals' in data and data['normals'] is not None:
        normals = data['normals']
        print(f"\nNormal statistics:")
        print(f"  Normal shape: {normals.shape}")
        print(f"  Has normals: Yes")


# ==================== Functions from replay_saved_frames.py ====================

def hash_colors_from_ids(ids: np.ndarray) -> np.ndarray:
    """Generate consistent colors from IDs."""
    h = (ids.astype(np.uint32) * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
    r = (h & np.uint32(0xFF)).astype(np.uint8)
    g = ((h >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
    b = ((h >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
    col = np.stack([r, g, b], axis=1)
    return np.maximum(col, 32)


def compute_3d_bboxes(points: np.ndarray, instance_ids: np.ndarray, semantic_labels: np.ndarray = None, min_points: int = 10) -> dict:
    """Compute axis-aligned 3D bounding boxes per instance."""
    bboxes = {}
    unique_ids = np.unique(instance_ids)
    
    for inst_id in unique_ids:
        if inst_id == 0:  # Skip background
            continue
            
        mask = instance_ids == inst_id
        if np.sum(mask) < min_points:
            continue
            
        inst_points = points[mask]
        if inst_points.shape[0] == 0:
            continue
        
        # Compute axis-aligned bounding box
        min_pt = np.min(inst_points, axis=0)
        max_pt = np.max(inst_points, axis=0)
        center = (min_pt + max_pt) / 2.0
        size = max_pt - min_pt
        
        # Generate consistent color for this instance ID
        color = hash_colors_from_ids(np.array([inst_id]))[0]
        
        bboxes[inst_id] = {
            'center': center,
            'size': size,
            'min_pt': min_pt,
            'max_pt': max_pt,
            'color': color,
            'num_points': inst_points.shape[0]
        }
    
    return bboxes


def visualize_full_bounding_box(min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float, entity_path: str = "full_area"):
    """Draw a wireframe for the overall XYZ bounds of all points."""
    if not RERUN_AVAILABLE:
        return
    
    box_corners = [
        [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z],
    ]
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    edge_lines = []
    for edge in edges:
        edge_lines.append([box_corners[edge[0]], box_corners[edge[1]]])
    
    rr.log(f"{entity_path}/bounding_box", rr.LineStrips3D(
        strips=edge_lines,
        colors=[[255, 255, 0]],
        radii=0.01
    ))


def create_floor_occupancy_grid(bboxes: dict, point_clouds: dict, floor_z_min: float, floor_z_max: float, 
                                min_x: float, max_x: float, min_y: float, max_y: float, grid_resolution: float,
                                wall_mask: np.ndarray = None, all_points: np.ndarray = None) -> np.ndarray:
    """Build a coarse 2D occupancy grid over XY.
    
    Args:
        wall_mask: Optional boolean mask indicating which points are walls (for marking wall cells)
        all_points: Optional full point cloud array (required if wall_mask is provided)
    """
    grid_width = int(np.ceil((max_x - min_x) / grid_resolution)) + 4
    grid_height = int(np.ceil((max_y - min_y) / grid_resolution)) + 4
    
    grid = np.zeros((grid_height, grid_width), dtype=int)
    grid.fill(3)  # Initialize as unknown
    
    has_point_cloud_mask = np.zeros((grid_height, grid_width), dtype=bool)
    
    for inst_id, bbox in bboxes.items():
        if inst_id not in point_clouds:
            continue
            
        pc = point_clouds[inst_id]
        if pc is None or len(pc) == 0:
            continue
        
        for point in pc:
            x, y, z = point
            grid_x = int(np.floor((x - min_x) / grid_resolution))
            grid_y = int(np.floor((y - min_y) / grid_resolution))
            
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                has_point_cloud_mask[grid_y, grid_x] = True
        
        if floor_z_min < 0.05:
            floor_threshold = 0.05
        else:
            floor_threshold = floor_z_min + 0.05
        
        obstacle_mask = (pc[:, 2] >= floor_threshold) & (pc[:, 2] < floor_z_max)
        obstacle_points = pc[obstacle_mask]
        
        if len(obstacle_points) == 0:
            continue
        
        for point in obstacle_points:
            x, y, z = point
            grid_x = int(np.floor((x - min_x) / grid_resolution))
            grid_y = int(np.floor((y - min_y) / grid_resolution))
            
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                grid[grid_y, grid_x] = 1
    
    traversed_mask = has_point_cloud_mask & (grid == 3)
    grid[traversed_mask] = 0
    
    if wall_mask is not None and all_points is not None and np.any(wall_mask):
        print(f"  Applying wall mask to occupancy grid ({np.sum(wall_mask):,} wall points)...")
        wall_points = all_points[wall_mask]
        
        z_valid_walls = (wall_points[:, 2] >= floor_z_min) & (wall_points[:, 2] <= floor_z_max)
        wall_points_valid = wall_points[z_valid_walls]
        
        if len(wall_points_valid) > 0:
            grid_x_all = np.floor((wall_points_valid[:, 0] - min_x) / grid_resolution).astype(np.int32)
            grid_y_all = np.floor((wall_points_valid[:, 1] - min_y) / grid_resolution).astype(np.int32)
            
            valid_mask = (grid_x_all >= 0) & (grid_x_all < grid_width) & (grid_y_all >= 0) & (grid_y_all < grid_height)
            grid_x_valid = grid_x_all[valid_mask]
            grid_y_valid = grid_y_all[valid_mask]
            
            wall_cells_at_boundary = 0
            for gx, gy in zip(grid_x_valid, grid_y_valid):
                is_at_boundary = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < grid_height and 0 <= nx < grid_width:
                        if grid[ny, nx] == 0:
                            is_at_boundary = True
                            break
                
                if is_at_boundary:
                    grid[gy, gx] = 2
                    wall_cells_at_boundary += 1
            
            print(f"  Marked {wall_cells_at_boundary:,} grid cells as walls (only at outline/boundary)")
    
    return grid


def create_top_view_rgb_image(grid: np.ndarray, grid_resolution: float, min_x: float, min_y: float, 
                              bboxes: dict, obj_rgb_colors: dict, areas: list, 
                              all_points: np.ndarray, all_rgb: np.ndarray, all_inst_ids: np.ndarray,
                              save_dir: Path = None, wall_mask: np.ndarray = None):
    """Create a top-view RGB image directly from RGB point cloud data."""
    if not CV2_AVAILABLE:
        print("  cv2 not available, skipping RGB image creation")
        return None
    
    if save_dir is None:
        script_dir = Path(__file__).parent.absolute()
        # save_dir = script_dir / "saved_frames"
        save_dir = script_dir
        
    grid_height, grid_width = grid.shape
    scale_factor = 10
    img_height = grid_height * scale_factor
    img_width = grid_width * scale_factor
    
    img = np.full((img_height, img_width, 3), [240, 240, 240], dtype=np.uint8)
    
    rgb_accum = np.zeros((img_height, img_width, 3), dtype=np.float64)
    count_accum = np.zeros((img_height, img_width), dtype=np.int32)
    
    print(f"  Projecting {len(all_points):,} RGB points onto top-view...")
    
    z_valid = (all_points[:, 2] > -0.5) & (all_points[:, 2] < 3.0)
    valid_points = all_points[z_valid]
    valid_rgb = all_rgb[z_valid]
    
    max_points = 500000
    if len(valid_points) > max_points:
        sample_indices = np.random.choice(len(valid_points), max_points, replace=False)
        valid_points = valid_points[sample_indices]
        valid_rgb = valid_rgb[sample_indices]
        print(f"  Downsampled to {len(valid_points):,} points for rendering")
    
    gx_all = ((valid_points[:, 0] - min_x) / grid_resolution * scale_factor).astype(np.int32) + scale_factor
    gy_all = ((valid_points[:, 1] - min_y) / grid_resolution * scale_factor).astype(np.int32) + scale_factor
    
    valid_mask_coords = (gx_all >= 0) & (gx_all < img_width) & (gy_all >= 0) & (gy_all < img_height)
    gx_valid = gx_all[valid_mask_coords]
    gy_valid = gy_all[valid_mask_coords]
    rgb_valid = valid_rgb[valid_mask_coords]
    
    print(f"  Accumulating RGB values for {len(gx_valid):,} valid points...")
    
    flat_indices = gy_valid * img_width + gx_valid
    pixel_counts = np.bincount(flat_indices, minlength=img_height * img_width)
    count_accum = pixel_counts.reshape(img_height, img_width).astype(np.int32)
    
    for c in range(3):
        rgb_sums = np.bincount(flat_indices, weights=rgb_valid[:, c].astype(np.float64), 
                               minlength=img_height * img_width)
        rgb_accum[:, :, c] = rgb_sums.reshape(img_height, img_width)
    
    valid_mask = count_accum > 0
    if np.any(valid_mask):
        avg_rgb = (rgb_accum[valid_mask] / count_accum[valid_mask, np.newaxis]).astype(np.uint8)
        img[valid_mask] = avg_rgb
        print(f"  Filled {np.sum(valid_mask):,} pixels with RGB data")
    
    for grid_y in range(grid_height):
        for grid_x in range(grid_width):
            start_y = grid_y * scale_factor
            end_y = (grid_y + 1) * scale_factor
            start_x = grid_x * scale_factor
            end_x = (grid_x + 1) * scale_factor
            
            cell_region = valid_mask[start_y:end_y, start_x:end_x] if start_y < img_height and start_x < img_width else np.array([])
            if len(cell_region) == 0 or not np.any(cell_region):
                cell_value = grid[grid_y, grid_x]
                if cell_value == 0:
                    color = [200, 200, 200]
                elif cell_value == 2:
                    color = [50, 50, 50]
                elif cell_value == 3:
                    color = [240, 240, 240]
                else:
                    color = [150, 150, 150]
                
                if start_y < img_height and start_x < img_width:
                    img[start_y:end_y, start_x:end_x] = color
    
    for inst_id, bbox in bboxes.items():
        center = bbox['center']
        gx = int((center[0] - min_x) / grid_resolution) * scale_factor + scale_factor
        gy = int((center[1] - min_y) / grid_resolution) * scale_factor + scale_factor
        
        if 0 <= gx < img_width and 0 <= gy < img_height:
            label = f"{inst_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(img, (gx, gy - text_height - 2), (gx + text_width, gy), (255, 255, 255), -1)
            cv2.putText(img, label, (gx, gy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    print(f"  Drawing {len(areas)} area outlines...")
    for area_idx, area in enumerate(areas):
        bbox = area.get('bbox', {})
        points = bbox.get('points', [])
        
        if len(points) == 0:
            bounds = area.get('bounds', [])
            if len(bounds) == 4:
                min_x_a, min_y_a, max_x_a, max_y_a = bounds
                points = [[min_x_a, min_y_a], [max_x_a, min_y_a], [max_x_a, max_y_a], [min_x_a, max_y_a]]
        
        if len(points) < 4:
            continue
            
        pts = []
        for p in points:
            if len(p) >= 2:
                gx = int((p[0] - min_x) / grid_resolution) * scale_factor + scale_factor
                gy = int((p[1] - min_y) / grid_resolution) * scale_factor + scale_factor
                gx = max(0, min(gx, img_width - 1))
                gy = max(0, min(gy, img_height - 1))
                pts.append([gx, gy])
        
        if len(pts) >= 4:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 6)
            
            if pts.shape[0] > 0:
                center = pts.mean(axis=0).astype(int)
                center[0] = max(0, min(center[0], img_width - 1))
                center[1] = max(0, min(center[1], img_height - 1))
                area_id = area.get('id', area_idx)
                label = f"A{area_id}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, 
                            (center[0] - text_width//2 - 8, center[1] - text_height - baseline - 8), 
                            (center[0] + text_width//2 + 8, center[1] + baseline + 8), 
                            (255, 255, 255), -1)
                cv2.putText(img, label, 
                           (center[0] - text_width//2, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if wall_mask is not None and np.any(wall_mask):
        print(f"  Drawing {np.sum(wall_mask):,} wall points on top-view...")
        wall_points = all_points[wall_mask]
        
        z_valid_walls = (wall_points[:, 2] > -0.5) & (wall_points[:, 2] < 3.0)
        wall_points_valid = wall_points[z_valid_walls]
        
        if len(wall_points_valid) > 0:
            max_wall_points = 50000
            if len(wall_points_valid) > max_wall_points:
                sample_indices = np.random.choice(len(wall_points_valid), max_wall_points, replace=False)
                wall_points_valid = wall_points_valid[sample_indices]
            
            gx_walls = ((wall_points_valid[:, 0] - min_x) / grid_resolution * scale_factor).astype(np.int32) + scale_factor
            gy_walls = ((wall_points_valid[:, 1] - min_y) / grid_resolution * scale_factor).astype(np.int32) + scale_factor
            
            valid_wall_mask = (gx_walls >= 0) & (gx_walls < img_width) & (gy_walls >= 0) & (gy_walls < img_height)
            gx_walls_valid = gx_walls[valid_wall_mask]
            gy_walls_valid = gy_walls[valid_wall_mask]
            
            wall_color = [200, 50, 50]
            overlay = img.copy()
            
            wall_mask_img = np.zeros((img_height, img_width), dtype=np.uint8)
            for gx, gy in zip(gx_walls_valid, gy_walls_valid):
                if 0 <= gx < img_width and 0 <= gy < img_height:
                    cv2.circle(wall_mask_img, (gx, gy), 3, 255, -1)
            
            wall_pixels = wall_mask_img > 0
            overlay[wall_pixels] = wall_color
            
            alpha = 0.6
            img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
            
            print(f"  Drew {len(gx_walls_valid):,} wall points on top-view map")
    
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb_map_path = save_dir / "top_view_rgb_map.png"
        cv2.imwrite(str(rgb_map_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved top-view RGB map image: {rgb_map_path}")
    except Exception as e:
        print(f"Warning: Could not save top-view RGB map image: {e}")
    
    return img


def visualize_traversable_map_2d(grid: np.ndarray, grid_resolution: float, min_x: float, min_y: float):
    """Create and visualize a 2D traversable map from the occupancy grid."""
    if not RERUN_AVAILABLE:
        return
    
    grid_height, grid_width = grid.shape
    scale_factor = 10
    img_height = grid_height * scale_factor
    img_width = grid_width * scale_factor
    
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    colors = {
        0: [0, 255, 0],
        1: [255, 0, 0],
        2: [128, 0, 0],
        3: [128, 128, 128]
    }
    
    for grid_y in range(grid_height):
        for grid_x in range(grid_width):
            cell_value = grid[grid_y, grid_x]
            color = colors.get(cell_value, [128, 128, 128])
            
            start_y = grid_y * scale_factor
            end_y = (grid_y + 1) * scale_factor
            start_x = grid_x * scale_factor
            end_x = (grid_x + 1) * scale_factor
            
            img[start_y:end_y, start_x:end_x] = color
    
    if CV2_AVAILABLE:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    rr.log("traversable_map/2d", rr.Image(img))
    print("  ✓ 2D traversable map visualized in Rerun")
    
    if CV2_AVAILABLE:
        try:
            script_dir = Path(__file__).parent.absolute()
            # save_dir = script_dir / "saved_frames"
            save_dir = script_dir
            save_dir.mkdir(exist_ok=True)
            map_image_path = save_dir / "traversable_map_2d.png"
            cv2.imwrite(str(map_image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  Saved 2D traversable map: {map_image_path}")
        except Exception as e:
            print(f"  Warning: Could not save traversable map image: {e}")


def draw_areas_on_image(img: np.ndarray, areas: list, scale_factor: int):
    """Draw detected areas on the image with different colors and labels."""
    if not CV2_AVAILABLE:
        return img
        
    img_with_areas = img.copy()
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], 
              [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255]]
    
    for i, area in enumerate(areas):
        color = colors[i % len(colors)]
        
        contour = area.get('contour')
        if contour is not None:
            cv2.drawContours(img_with_areas, [contour], -1, color, 3)
            x, y, w, h = cv2.boundingRect(contour)
        else:
            x, y, w, h = area.get('image_bounds', [0, 0, 0, 0])
            if w > 0 and h > 0:
                cv2.rectangle(img_with_areas, (x, y), (x + w, y + h), color, 2)
        
        if w > 0 and h > 0:
            label = f"Area {area['id']}"
            cv2.putText(img_with_areas, label, (x, max(10, y - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_with_areas


def detect_areas_from_traversable_map(grid: np.ndarray, min_x: float, min_y: float, grid_resolution: float):
    """Detect areas directly from the traversable map using connected components."""
    if not CV2_AVAILABLE:
        return []
    
    grid_height, grid_width = grid.shape
    
    traversable_mask = (grid == 0).astype(np.uint8)
    
    if np.sum(traversable_mask) == 0:
        print("  No traversable cells found in grid")
        return []
    
    num_labels, labels = cv2.connectedComponents(traversable_mask, connectivity=4)
    
    if num_labels <= 1:
        print("  No separate areas found")
        return []
    
    print(f"  Found {num_labels - 1} connected traversable areas")
    
    areas = []
    min_area_cells = 3
    min_area_world = 0.5
    
    filtered_small = 0
    
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(bool)
        
        area_cells = np.sum(component_mask)
        if area_cells < min_area_cells:
            filtered_small += 1
            continue
        
        rows = np.where(component_mask)[0]
        cols = np.where(component_mask)[1]
        
        if len(rows) == 0 or len(cols) == 0:
            continue
        
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        world_x = min_x + min_col * grid_resolution
        world_y = min_y + min_row * grid_resolution
        world_w = (max_col - min_col + 1) * grid_resolution
        world_h = (max_row - min_row + 1) * grid_resolution
        
        center_x = world_x + world_w / 2.0
        center_y = world_y + world_h / 2.0
        
        area_world = area_cells * (grid_resolution * grid_resolution)
        
        if area_world < min_area_world:
            filtered_small += 1
            continue
        
        component_mask_uint8 = component_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_world = []
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            for point in largest_contour:
                px, py = point[0]
                wx = min_x + px * grid_resolution
                wy = min_y + py * grid_resolution
                contour_world.append([wx, wy])
        else:
            contour_world = [
                [world_x, world_y],
                [world_x + world_w, world_y],
                [world_x + world_w, world_y + world_h],
                [world_x, world_y + world_h],
                [world_x, world_y]
            ]
        
        areas.append({
            'id': len(areas),
            'center': [center_x, center_y],
            'bounds': [world_x, world_y, world_x + world_w, world_y + world_h],
            'size': [world_w, world_h],
            'area': area_world,
            'points': contour_world
        })
    
    areas = sorted(areas, key=lambda a: a['area'], reverse=True)
    for i, area in enumerate(areas):
        area['id'] = i
    
    object_cells = np.where(grid == 1)
    num_objects_assigned = 0
    num_inside = 0
    num_nearby = 0
    max_nearby_distance = 2.0
    
    def point_in_polygon(point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    for obj_y, obj_x in zip(object_cells[0], object_cells[1]):
        obj_world_x = min_x + obj_x * grid_resolution
        obj_world_y = min_y + obj_y * grid_resolution
        obj_point = [obj_world_x, obj_world_y]
        
        assigned = False
        min_dist = float('inf')
        nearest_area_id = None
        
        for area in areas:
            area_polygon = area['points']
            if len(area_polygon) >= 3:
                if point_in_polygon(obj_point, area_polygon):
                    if 'object_cells' not in area:
                        area['object_cells'] = []
                    if [obj_x, obj_y] not in area['object_cells']:
                        area['object_cells'].append([obj_x, obj_y])
                        num_objects_assigned += 1
                        num_inside += 1
                        assigned = True
                        break
        
        if not assigned:
            for area in areas:
                area_center = area['center']
                dist = np.sqrt((obj_world_x - area_center[0])**2 + (obj_world_y - area_center[1])**2)
                
                if dist < max_nearby_distance and dist < min_dist:
                    min_dist = dist
                    nearest_area_id = area['id']
            
            if nearest_area_id is not None:
                if 'object_cells' not in areas[nearest_area_id]:
                    areas[nearest_area_id]['object_cells'] = []
                if [obj_x, obj_y] not in areas[nearest_area_id]['object_cells']:
                    areas[nearest_area_id]['object_cells'].append([obj_x, obj_y])
                    num_objects_assigned += 1
                    num_nearby += 1
    
    for i, area in enumerate(areas):
        num_objects = len(area.get('object_cells', []))
        print(f"    Area {i}: {area['area']:.2f} m² at ({area['center'][0]:.2f}, {area['center'][1]:.2f}), {num_objects} objects")
    
    print(f"  Filtered out {filtered_small} areas smaller than {min_area_world} m²")
    print(f"  Assigned {num_objects_assigned} object cells to areas ({num_inside} inside, {num_nearby} nearby within {max_nearby_distance}m)")
    print(f"  Final areas detected: {len(areas)}")
    return areas


def visualize_object_area_assignments(areas: list, bboxes: dict, grid: np.ndarray, 
                                     grid_resolution: float, min_x: float, min_y: float,
                                     entity_path: str = "object_area_assignments"):
    """Visualize which objects are assigned to which areas."""
    if not RERUN_AVAILABLE or not areas or not bboxes:
        return
    
    grid_to_instance = {}
    grid_height, grid_width = grid.shape
    
    for inst_id, bbox in bboxes.items():
        center = bbox['center']
        size = bbox['size']
        
        min_x_obj = center[0] - size[0] / 2.0
        max_x_obj = center[0] + size[0] / 2.0
        min_y_obj = center[1] - size[1] / 2.0
        max_y_obj = center[1] + size[1] / 2.0
        
        grid_x_min = int((min_x_obj - min_x) / grid_resolution)
        grid_x_max = int((max_x_obj - min_x) / grid_resolution) + 1
        grid_y_min = int((min_y_obj - min_y) / grid_resolution)
        grid_y_max = int((max_y_obj - min_y) / grid_resolution) + 1
        
        for gy in range(max(0, grid_y_min), min(grid_height, grid_y_max)):
            for gx in range(max(0, grid_x_min), min(grid_width, grid_x_max)):
                if grid[gy, gx] == 1:
                    grid_to_instance[(gx, gy)] = inst_id
    
    area_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], 
                   [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255]]
    
    for area in areas:
        area_id = area['id']
        area_color = area_colors[area_id % len(area_colors)]
        object_cells = area.get('object_cells', [])
        
        if not object_cells:
            continue
        
        assigned_inst_ids = set()
        for obj_x, obj_y in object_cells:
            if (obj_x, obj_y) in grid_to_instance:
                inst_id = grid_to_instance[(obj_x, obj_y)]
                assigned_inst_ids.add(inst_id)
        
        if not assigned_inst_ids:
            continue
        
        assigned_centers = []
        assigned_half_sizes = []
        assigned_colors = []
        assigned_labels = []
        
        for inst_id in assigned_inst_ids:
            if inst_id in bboxes:
                bbox = bboxes[inst_id]
                assigned_centers.append(bbox['center'])
                assigned_half_sizes.append(bbox['size'] / 2.0)
                assigned_colors.append([c // 2 + 128 for c in area_color])
                assigned_labels.append(f"Area {area_id} - Obj {inst_id}")
        
        if assigned_centers:
            rr.log(f"{entity_path}/area_{area_id}/objects", rr.Boxes3D(
                centers=np.array(assigned_centers),
                half_sizes=np.array(assigned_half_sizes),
                colors=np.array(assigned_colors),
                labels=assigned_labels,
                radii=0.03
            ))
            
            rr.log(f"{entity_path}/area_{area_id}/summary", rr.TextLog(
                f"Area {area_id}: {len(assigned_inst_ids)} objects assigned\n"
                f"Object IDs: {sorted(assigned_inst_ids)}"
            ))
            
            print(f"  Area {area_id}: {len(assigned_inst_ids)} objects assigned - {sorted(assigned_inst_ids)}")


def visualize_image_detected_areas(areas: list, entity_path: str):
    """Log 2D areas as line strips following the actual contour boundaries in Rerun."""
    if not RERUN_AVAILABLE or not areas:
        return
    
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], 
              [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 0, 255]]
    
    for i, area in enumerate(areas):
        color = colors[i % len(colors)]
        area_id = area['id']
        
        points = area.get('points', [])
        
        if len(points) < 3:
            bounds = area.get('bounds', [])
            if len(bounds) == 4:
                min_x_a, min_y_a, max_x_a, max_y_a = bounds
                points = [[min_x_a, min_y_a], [max_x_a, min_y_a], 
                         [max_x_a, max_y_a], [min_x_a, max_y_a], [min_x_a, min_y_a]]
            else:
                center = area['center']
                size = area['size']
                cx, cy = center[0], center[1]
                w, h = size[0], size[1]
                points = [[cx - w/2, cy - h/2], [cx + w/2, cy - h/2],
                         [cx + w/2, cy + h/2], [cx - w/2, cy + h/2], [cx - w/2, cy - h/2]]
        
        if len(points) > 0:
            points_2d = np.array(points)
            if len(points_2d.shape) == 2 and points_2d.shape[1] >= 2:
                points_2d = points_2d[:, :2].tolist()
                
                if len(points_2d) > 0 and points_2d[0] != points_2d[-1]:
                    points_2d.append(points_2d[0])
                
                rr.log(f"{entity_path}/area_{area_id}", rr.LineStrips2D(
                    strips=[points_2d],
                    colors=[color],
                    labels=[f"Area {area_id}"]
                ))
                
                points_3d = [[p[0], p[1], 0.0] for p in points_2d]
                rr.log(f"{entity_path}/area_{area_id}_3d", rr.LineStrips3D(
                    strips=[points_3d],
                    colors=[color],
                    labels=[f"Area {area_id}"]
                ))
                
                center = area['center']
                rr.log(f"{entity_path}/area_{area_id}/center", rr.Points2D(
                    positions=[[center[0], center[1]]],
                    colors=[color],
                    radii=0.1,
                    labels=[f"Area {area_id}"]
                ))


def visualize_floor_grid(floor_grid: np.ndarray, grid_resolution: float, min_x: float, min_y: float, 
                        floor_num: int, entity_path: str = "floor_grid"):
    """Log grid cells and return areas detected from traversable map."""
    detected_areas = detect_areas_from_traversable_map(floor_grid, min_x, min_y, grid_resolution)
    
    if not RERUN_AVAILABLE:
        return detected_areas
    
    grid_height, grid_width = floor_grid.shape
    
    floor_cells = []
    object_cells = []
    wall_cells = []
    unknown_cells = []
    
    for grid_y in range(grid_height):
        for grid_x in range(grid_width):
            cell_value = floor_grid[grid_y, grid_x]
            cell_center_x = min_x + (grid_x - 0.5) * grid_resolution
            cell_center_y = min_y + (grid_y - 0.5) * grid_resolution
            
            if cell_value == 0:
                floor_cells.append([cell_center_x, cell_center_y, 0.01])
            elif cell_value == 1:
                object_cells.append([cell_center_x, cell_center_y, 0.02])
            elif cell_value == 2:
                wall_cells.append([cell_center_x, cell_center_y, 0.03])
            elif cell_value == 3:
                unknown_cells.append([cell_center_x, cell_center_y, 0.015])
    
    if floor_cells:
        rr.log(f"{entity_path}/floor", rr.Points3D(
            positions=floor_cells,
            colors=[[200, 200, 200]],
            radii=0.08
        ))
    
    if object_cells:
        rr.log(f"{entity_path}/objects", rr.Points3D(
            positions=object_cells,
            colors=[[100, 100, 100]],
            radii=0.06
        ))
    
    if wall_cells:
        rr.log(f"{entity_path}/walls", rr.Points3D(
            positions=wall_cells,
            colors=[[50, 50, 50]],
            radii=0.08
        ))
    
    if unknown_cells:
        rr.log(f"{entity_path}/unknown", rr.Points3D(
            positions=unknown_cells,
            colors=[[240, 240, 240]],
            radii=0.05
        ))
    
    # Visualize detected areas if available
    if detected_areas:
        visualize_image_detected_areas(detected_areas, f"{entity_path}/areas")
    
    return detected_areas


def separate_into_floors_by_height(bboxes: dict, point_clouds: dict, all_points: np.ndarray = None, 
                                   room_names: list = None, wall_mask: np.ndarray = None) -> dict:
    """Create a single comprehensive occupancy analysis over the full Z-range.
    
    Args:
        bboxes: Dictionary of bounding boxes
        point_clouds: Dictionary of point clouds per instance
        all_points: Optional full point cloud array to ensure all points are included in bounds
        wall_mask: Optional boolean mask indicating which points are walls
    """
    if not bboxes:
        return {}
    
    all_points_list = []
    if all_points is not None and len(all_points) > 0:
        all_points_list.append(all_points)
    
    for pc in point_clouds.values():
        if pc is not None and len(pc) > 0:
            all_points_list.append(pc)
    
    if len(all_points_list) == 0:
        all_centers = [bbox['center'] for bbox in bboxes.values()]
        all_centers = np.array(all_centers)
        min_x = np.min(all_centers[:, 0])
        max_x = np.max(all_centers[:, 0])
        min_y = np.min(all_centers[:, 1])
        max_y = np.max(all_centers[:, 1])
        min_z = np.min(all_centers[:, 2])
        max_z = np.max(all_centers[:, 2])
    else:
        all_points_combined = np.concatenate(all_points_list, axis=0)
        min_x = np.min(all_points_combined[:, 0])
        max_x = np.max(all_points_combined[:, 0])
        min_y = np.min(all_points_combined[:, 1])
        max_y = np.max(all_points_combined[:, 1])
        min_z = np.min(all_points_combined[:, 2])
        max_z = np.max(all_points_combined[:, 2])
    
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding_x = max(x_range * 0.15, 1.0)
    padding_y = max(y_range * 0.15, 1.0)
    
    min_x = min_x - padding_x
    max_x = max_x + padding_x
    min_y = min_y - padding_y
    max_y = max_y + padding_y
    
    print(f"Grid bounds computed from {len(all_points_list)} point cloud(s):")
    print(f"  X: [{min_x:.2f}, {max_x:.2f}] (range: {max_x - min_x:.2f}m)")
    print(f"  Y: [{min_y:.2f}, {max_y:.2f}] (range: {max_y - min_y:.2f}m)")
    print(f"  Z: [{min_z:.2f}, {max_z:.2f}] (range: {max_z - min_z:.2f}m)")
    
    if RERUN_AVAILABLE:
        visualize_full_bounding_box(min_x, max_x, min_y, max_y, min_z, max_z, "full_area")
    
    floor_z_min = max(min_z, -0.2)
    floor_z_max = min(max_z, floor_z_min + 3.0)
    
    print(f"Using Z range for occupancy: [{floor_z_min:.2f}, {floor_z_max:.2f}]")
    
    grid_resolution = 0.05
    x_range_final = max_x - min_x
    y_range_final = max_y - min_y
    max_range = max(x_range_final, y_range_final)
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    half_size = max_range / 2.0
    min_x = center_x - half_size
    max_x = center_x + half_size
    min_y = center_y - half_size
    max_y = center_y + half_size
    
    grid_size = int(np.ceil(max_range / grid_resolution)) + 4
    grid_width = grid_size
    grid_height = grid_size
    
    print(f"  Aligned to square grid: {max_range:.2f}m x {max_range:.2f}m")
    print(f"Grid size: {grid_width} x {grid_height} cells ({grid_width * grid_resolution:.2f}m x {grid_height * grid_resolution:.2f}m)")
    
    floor_grid = create_floor_occupancy_grid(
        bboxes, point_clouds, floor_z_min, floor_z_max,
        min_x, max_x, min_y, max_y, grid_resolution,
        wall_mask=wall_mask, all_points=all_points
    )
    
    detected_areas = visualize_floor_grid(floor_grid, grid_resolution, min_x, min_y, 1, "comprehensive_occupancy")
    
    floor_objects = {}
    for inst_id, bbox in bboxes.items():
        if floor_z_min <= bbox['center'][2] <= floor_z_max:
            floor_objects[inst_id] = bbox
    
    floor_center_z = (floor_z_min + floor_z_max) / 2.0
    occupied_cells = np.sum(floor_grid)
    total_cells = grid_width * grid_height
    occupancy_rate = occupied_cells / total_cells if total_cells > 0 else 0
    
    floor_name = "Comprehensive_Occupancy"
    
    floors = {
        floor_name: {
            'floor_index': 1,
            'z_min': floor_z_min,
            'z_max': floor_z_max,
            'z_center': floor_center_z,
            'bboxes': floor_objects,
            'object_count': len(floor_objects),
            'occupancy_grid': floor_grid,
            'grid_resolution': grid_resolution,
            'grid_width': grid_width,
            'grid_height': grid_height,
            'occupied_cells': occupied_cells,
            'total_cells': total_cells,
            'occupancy_rate': occupancy_rate,
            'bounds': {
                'min_x': min_x,
                'max_x': max_x,
                'min_y': min_y,
                'max_y': max_y,
                'min_z': floor_z_min,
                'max_z': floor_z_max
            }
        }
    }
    
    if detected_areas:
        if RERUN_AVAILABLE:
            visualize_image_detected_areas(detected_areas, "comprehensive_occupancy/image_areas")
        floors[floor_name]['image_areas'] = detected_areas
        floors[floor_name]['num_areas'] = len(detected_areas)
    else:
        floors[floor_name]['image_areas'] = []
        floors[floor_name]['num_areas'] = 0
    
    return floors


def visualize_3d_bboxes(bboxes: dict, entity_path: str = "bboxes/3d"):
    """Visualize 3D bounding boxes using Rerun's Boxes3D component."""
    if not RERUN_AVAILABLE or not bboxes:
        return
    
    all_centers = []
    all_half_sizes = []
    all_colors = []
    all_labels = []
    
    individual_centers = []
    individual_colors = []
    individual_labels = []
    individual_ids = []
    
    for inst_id, bbox in bboxes.items():
        all_centers.append(bbox['center'])
        all_half_sizes.append(bbox['size'] / 2.0)
        all_colors.append(bbox['color'])
        all_labels.append(f"ID:{inst_id}")
        
        individual_centers.append(bbox['center'])
        individual_colors.append(bbox['color'])
        individual_labels.append(f"Object ID: {inst_id}\nPoints: {bbox.get('num_points', 0)}")
        individual_ids.append(inst_id)
    
    if all_centers:
        rr.log(entity_path, rr.Boxes3D(
            centers=np.array(all_centers),
            half_sizes=np.array(all_half_sizes),
            colors=np.array(all_colors),
            radii=0.025,
            labels=all_labels,
        ))
        
        objects_category = f"{entity_path}/objects"
        
        rr.log(f"{objects_category}/labels", rr.Points3D(
            positions=np.array(individual_centers),
            colors=np.array(individual_colors),
            labels=individual_labels,
            radii=0.05
        ))
        
        id_list = sorted(individual_ids)
        rr.log(f"{objects_category}/summary", rr.TextLog(
            f"Total objects: {len(bboxes)}\nObject IDs: {', '.join(map(str, id_list))}"
        ))
        
        for inst_id, bbox in bboxes.items():
            center = bbox['center']
            size = bbox['size']
            num_points = bbox.get('num_points', 0)
            object_path = f"{objects_category}/object_{inst_id}"
            
            # Log the bounding box directly to the object path to avoid empty parent entity warning
            rr.log(object_path, rr.Boxes3D(
                centers=[center],
                half_sizes=[size / 2.0],
                colors=[bbox['color']],
                labels=[f"Object {inst_id}"],
                radii=0.025
            ))
            
            rr.log(f"{object_path}/info", rr.TextLog(
                f"Object ID: {inst_id}\n"
                f"Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]\n"
                f"Size: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]\n"
                f"Points: {num_points}"
            ))
            
            rr.log(f"{object_path}/center", rr.Points3D(
                positions=[center],
                colors=[bbox['color']],
                labels=[f"ID: {inst_id}"],
                radii=0.08
            ))


def save_objects_and_areas_to_json(save_dir: Path, bboxes: dict, floors: dict, filename: str = "analysis.json") -> Path:
    """Write objects and areas to JSON with polygon points and object-area mapping."""
    def to_list(val):
        if hasattr(val, "tolist"):
            return val.tolist()
        if isinstance(val, (list, tuple)):
            return [to_list(v) for v in val]
        if isinstance(val, (int, float, str)):
            return val
        return val
    
    areas = []
    areas_info = []
    for floor_name, floor_data in (floors or {}).items():
        image_areas = floor_data.get("image_areas", [])
        for area in image_areas:
            area_id_int = int(area.get("id", 0))
            
            polygon_points = area.get("points", [])
            
            if polygon_points and len(polygon_points) >= 3:
                points_2d = []
                for pt in polygon_points:
                    if len(pt) >= 2:
                        points_2d.append([float(pt[0]), float(pt[1])])
                
                if points_2d:
                    x_coords = [p[0] for p in points_2d]
                    y_coords = [p[1] for p in points_2d]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    bounds = (min_x, min_y, max_x, max_y)
                else:
                    bounds = (0, 0, 0, 0)
            else:
                b = area.get("bounds")
                if b and len(b) == 4:
                    min_x, min_y, max_x, max_y = [float(v) for v in b]
                    points_2d = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                    bounds = (min_x, min_y, max_x, max_y)
                else:
                    cx, cy = float(area["center"][0]), float(area["center"][1])
                    w, h = float(area["size"][0]), float(area["size"][1])
                    min_x, max_x = cx - w / 2.0, cx + w / 2.0
                    min_y, max_y = cy - h / 2.0, cy + h / 2.0
                    points_2d = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                    bounds = (min_x, min_y, max_x, max_y)
            
            area_dict = {
                "id": area_id_int,
                "polygon": {"points": points_2d},
                "label": "",
                "bounds": bounds,
            }
            areas.append(area_dict)
            areas_info.append({"id": area_id_int, "bounds": bounds, "points": points_2d})
    
    def point_in_bounds(x: float, y: float, bounds: tuple) -> bool:
        mnx, mny, mxx, mxy = bounds
        return (mnx <= x <= mxx) and (mny <= y <= mxy)
    
    def point_in_polygon(x: float, y: float, polygon: list) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm."""
        if len(polygon) < 3:
            return False
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    objects = []
    for inst_id, bbox in (bboxes or {}).items():
        min_pt = bbox.get("min_pt")
        max_pt = bbox.get("max_pt")
        points_3d = []
        if min_pt is not None and max_pt is not None:
            xmin, ymin, zmin = [float(v) for v in to_list(min_pt)]
            xmax, ymax, zmax = [float(v) for v in to_list(max_pt)]
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            points_3d = [
                [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
                [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
            ]
        else:
            center = [float(x) for x in to_list(bbox.get("center", [0, 0, 0]))]
            size = [float(x) for x in to_list(bbox.get("size", [0, 0, 0]))]
            cx, cy, cz = center
            sx, sy, sz = size
            hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
            xmin, xmax = cx - hx, cx + hx
            ymin, ymax = cy - hy, cy + hy
            zmin, zmax = cz - hz, cz + hz
            points_3d = [
                [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
                [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
            ]
        
        area_id_match = -1
        grid_resolution = None
        min_x = None
        min_y = None
        
        for floor_name, floor_data in (floors or {}).items():
            grid_resolution = floor_data.get("grid_resolution", 0.1)
            bounds = floor_data.get("bounds", {})
            min_x = bounds.get("min_x", 0)
            min_y = bounds.get("min_y", 0)
            break
        
        if grid_resolution is not None and min_x is not None and min_y is not None:
            xmin, ymin, zmin = [float(v) for v in to_list(bbox.get("min_pt", [cx, cy, 0]))]
            xmax, ymax, zmax = [float(v) for v in to_list(bbox.get("max_pt", [cx, cy, 0]))]
            
            obj_grid_x_min = int((xmin - min_x) / grid_resolution)
            obj_grid_x_max = int((xmax - min_x) / grid_resolution) + 1
            obj_grid_y_min = int((ymin - min_y) / grid_resolution)
            obj_grid_y_max = int((ymax - min_y) / grid_resolution) + 1
            
            for floor_name, floor_data in (floors or {}).items():
                image_areas = floor_data.get("image_areas", [])
                for area in image_areas:
                    object_cells = area.get("object_cells", [])
                    for obj_cell_x, obj_cell_y in object_cells:
                        if (obj_grid_x_min <= obj_cell_x <= obj_grid_x_max and 
                            obj_grid_y_min <= obj_cell_y <= obj_grid_y_max):
                            area_id_match = area.get("id", -1)
                            break
                    if area_id_match != -1:
                        break
                if area_id_match != -1:
                    break
        
        if area_id_match == -1:
            for a in areas_info:
                if "points" in a and len(a["points"]) >= 3:
                    if point_in_polygon(cx, cy, a["points"]):
                        area_id_match = a["id"]
                        break
                elif point_in_bounds(cx, cy, a["bounds"]):
                    area_id_match = a["id"]
                    break
        
        objects.append({
            "id": int(inst_id),
            "bbox": {"points": points_3d},
            "num_points": int(bbox.get("num_points", 0)),
            "area_id": int(area_id_match),
            "label": "",
        })
    
    output = {"objects": objects, "areas": areas}
    
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved objects and areas JSON: {out_path}")
    return out_path


def main():
    """Main function to process PLY files through the full pipeline."""
    script_dir = Path(__file__).parent.absolute()
    
    # ply_dir = script_dir / "PLY_output" / "Beechwood_0_int"
    ply_dir = script_dir
    
    ply_files = {
        'instance': ply_dir / "integrated_slam_map_instance.ply",
        'rgb': ply_dir / "integrated_slam_map_rgb.ply",
        'semantic': ply_dir / "integrated_slam_map_semantic.ply",
    }
    
    print("="*60)
    print("PLY File Processor (Same Pipeline as replay_saved_frames.py)")
    print("="*60)
    print(f"Reading PLY files from: {ply_dir}")
    
    # Check library availability
    if not OPEN3D_AVAILABLE and not PLYFILE_AVAILABLE:
        print("\nERROR: No PLY reading library available!")
        print("Please install one of:")
        print("  pip install open3d")
        print("  pip install plyfile")
        return
    
    print(f"\nLibrary availability:")
    print(f"  Open3D: {'Available' if OPEN3D_AVAILABLE else 'Not installed'}")
    print(f"  plyfile: {'Available' if PLYFILE_AVAILABLE else 'Not installed'}")
    print(f"  Rerun: {'Available' if RERUN_AVAILABLE else 'Not installed'}")
    print(f"  OpenCV: {'Available' if CV2_AVAILABLE else 'Not installed'}")
    
    # Initialize Rerun if available
    if RERUN_AVAILABLE:
        rr.init("ply_processor", spawn=True)
        print("\nRerun viewer initialized")
    
    # Read all PLY files
    print("\n" + "="*60)
    print("Stage: Reading PLY files")
    print("="*60)
    data_dict = {}
    for file_type, file_path in ply_files.items():
        data = read_ply_file(file_path)
        data_dict[file_type] = data
        if data is not None:
            print_statistics(data, file_type)
    
    # Determine which point cloud to use (prefer instance, fall back to semantic, then RGB)
    base_points = None
    instance_ids = None
    rgb_colors = None
    semantic_labels = None
    semantic_colors = None
    
    # Extract instance IDs from instance PLY
    if 'instance' in data_dict and data_dict['instance'] is not None:
        instance_data = data_dict['instance']
        if 'points' in instance_data:
            base_points = instance_data['points']
            print(f"\nUsing instance PLY as base: {len(base_points):,} points")
            
            # Extract instance IDs from colors
            if 'colors' in instance_data and instance_data['colors'] is not None:
                label_ids, unique_colors = extract_labels_from_colors(instance_data['colors'])
                # Map label IDs to instance IDs (use label_id + 1 to avoid 0)
                instance_ids = label_ids + 1  # Shift by 1 to avoid 0 (background)
                print(f"  Extracted {len(np.unique(instance_ids))} unique instance IDs from colors")
            elif 'instance_id' in instance_data:
                instance_ids = instance_data['instance_id']
                print(f"  Using instance_id field: {len(np.unique(instance_ids))} unique IDs")
    
    # Extract RGB colors from RGB PLY
    if 'rgb' in data_dict and data_dict['rgb'] is not None:
        rgb_data = data_dict['rgb']
        if 'points' in rgb_data and 'colors' in rgb_data and rgb_data['colors'] is not None:
            if base_points is None:
                base_points = rgb_data['points']
            # Normalize RGB colors to 0-255 range
            rgb_colors_raw = rgb_data['colors']
            if rgb_colors_raw.max() <= 1.0:
                rgb_colors = (rgb_colors_raw * 255).astype(np.uint8)
            else:
                rgb_colors = rgb_colors_raw.astype(np.uint8)
            print(f"\nRGB colors loaded: {len(rgb_colors):,} points")
    
    # Extract semantic labels if available
    if 'semantic' in data_dict and data_dict['semantic'] is not None:
        semantic_data = data_dict['semantic']
        if 'points' in semantic_data:
            if base_points is None:
                base_points = semantic_data['points']
            # Extract semantic IDs from colors
            if 'colors' in semantic_data and semantic_data['colors'] is not None:
                semantic_colors_raw = semantic_data['colors']
                # Normalize semantic colors to 0-255 range
                if semantic_colors_raw.max() <= 1.0:
                    semantic_colors = (semantic_colors_raw * 255).astype(np.uint8)
                else:
                    semantic_colors = semantic_colors_raw.astype(np.uint8)
                semantic_label_ids, _ = extract_labels_from_colors(semantic_data['colors'])
                semantic_labels = semantic_label_ids
                print(f"\nSemantic labels loaded: {len(np.unique(semantic_labels))} unique labels")
            elif 'semantic_id' in semantic_data:
                semantic_labels = semantic_data['semantic_id']
    
    if base_points is None:
        print("\nERROR: No valid point cloud data found!")
        return
    
    # Ensure all arrays have the same length (align point clouds)
    print(f"\nAligning point clouds...")
    min_len = len(base_points)
    if instance_ids is not None:
        min_len = min(min_len, len(instance_ids))
    if rgb_colors is not None:
        min_len = min(min_len, len(rgb_colors))
    if semantic_labels is not None:
        min_len = min(min_len, len(semantic_labels))
    if semantic_colors is not None:
        min_len = min(min_len, len(semantic_colors))
    
    base_points = base_points[:min_len]
    if instance_ids is not None:
        instance_ids = instance_ids[:min_len]
    if rgb_colors is not None:
        rgb_colors = rgb_colors[:min_len]
    if semantic_labels is not None:
        semantic_labels = semantic_labels[:min_len]
    if semantic_colors is not None:
        semantic_colors = semantic_colors[:min_len]
    
    print(f"  Aligned to {min_len:,} points")
    
    if instance_ids is None:
        print("\nERROR: Could not extract instance IDs!")
        return
    
    # Keep original points for transformation
    base_points_original = base_points.copy()
    
    # Step 1: Initial floor alignment using Z-based method
    print("\n" + "="*60)
    print("Stage 1: Initial floor alignment (Z-based)")
    print("="*60)
    base_points_prelim, rotation_matrix_prelim, translation_prelim = align_floor_to_z0(base_points_original)
    
    # Step 2: Build preliminary occupancy grid to identify floor cells
    print("\n" + "="*60)
    print("Stage 2: Building preliminary occupancy grid")
    print("="*60)
    
    # Compute bounding box from all actual points
    min_x = np.min(base_points_prelim[:, 0])
    max_x = np.max(base_points_prelim[:, 0])
    min_y = np.min(base_points_prelim[:, 1])
    max_y = np.max(base_points_prelim[:, 1])
    min_z = np.min(base_points_prelim[:, 2])
    max_z = np.max(base_points_prelim[:, 2])
    
    # Add padding/margin to ensure all points are included (increased padding)
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding_x = max(x_range * 0.15, 1.0)  # 15% margin or minimum 1.0m (increased)
    padding_y = max(y_range * 0.15, 1.0)
    
    min_x = min_x - padding_x
    max_x = max_x + padding_x
    min_y = min_y - padding_y
    max_y = max_y + padding_y
    
    print(f"Grid bounds with padding:")
    print(f"  X: [{min_x:.2f}, {max_x:.2f}] (range: {max_x - min_x:.2f}m)")
    print(f"  Y: [{min_y:.2f}, {max_y:.2f}] (range: {max_y - min_y:.2f}m)")
    
    floor_z_min = max(min_z, -0.2)
    floor_z_max = min(max_z, floor_z_min + 3.0)
    grid_resolution = 0.05  # 5cm resolution (smaller cells for higher detail)
    
    # Create point cloud dictionary for preliminary grid
    pc_inst_dict_prelim = {}
    for inst_id in np.unique(instance_ids):
        if inst_id == 0:
            continue
        inst_mask = instance_ids == inst_id
        if np.sum(inst_mask) >= 50:
            pc_inst_dict_prelim[inst_id] = base_points_prelim[inst_mask]
    
    # Compute preliminary bboxes
    bboxes_prelim = compute_3d_bboxes(base_points_prelim, instance_ids, semantic_labels, min_points=50)
    
    # Build preliminary grid
    floor_grid_prelim = create_floor_occupancy_grid(
        bboxes_prelim, pc_inst_dict_prelim, floor_z_min, floor_z_max,
        min_x, max_x, min_y, max_y, grid_resolution
    )
    
    # Step 3: Extract labels and detect dominant floor label from grid
    print("\n" + "="*60)
    print("Stage 3: Detecting floor from labels and/or RGB colors")
    print("="*60)
    floor_label_id = None
    floor_mask = None
    floor_color = None
    labels_to_use = instance_ids  # Default to instance IDs
    use_semantic = False
    use_color = False
    
    # Try RGB color-based floor detection first (if RGB colors available)
    if rgb_colors is not None and len(rgb_colors) == len(base_points_prelim):
        print("  Trying RGB color-based floor detection...")
        floor_color, floor_mask_color = detect_floor_from_color(
            base_points_prelim, rgb_colors, floor_grid_prelim,
            grid_resolution, min_x, min_y, floor_z_min, floor_z_max
        )
        if floor_color is not None and floor_mask_color is not None and np.sum(floor_mask_color) > 0:
            floor_mask = floor_mask_color
            use_color = True
            print("  ✓ Using RGB color-based floor detection")
    
    # Try semantic labels (from colors) if color-based detection didn't work or as additional check
    if (floor_mask is None or not use_color) and semantic_labels is not None and len(semantic_labels) == len(base_points_prelim):
        print("  Trying semantic labels from extracted colors...")
        floor_label_id, floor_mask_label = detect_floor_label_from_grid(
            base_points_prelim, semantic_labels, floor_grid_prelim,
            grid_resolution, min_x, min_y, floor_z_min, floor_z_max
        )
        if floor_label_id is not None:
            # Combine with color-based mask if available
            if floor_mask is not None:
                # Use intersection: point must match both color and label
                floor_mask = floor_mask & floor_mask_label
                print("  ✓ Combined RGB color and semantic label floor detection")
            else:
                floor_mask = floor_mask_label
            labels_to_use = semantic_labels
            use_semantic = True
    
    # If semantic failed, try instance labels directly
    if floor_label_id is None and len(instance_ids) == len(base_points_prelim):
        print("  Trying instance labels...")
        floor_label_id, floor_mask_label = detect_floor_label_from_grid(
            base_points_prelim, instance_ids, floor_grid_prelim,
            grid_resolution, min_x, min_y, floor_z_min, floor_z_max
        )
        if floor_label_id is not None:
            # Combine with color-based mask if available
            if floor_mask is not None:
                # Use intersection: point must match both color and label
                floor_mask = floor_mask & floor_mask_label
                print("  ✓ Combined RGB color and instance label floor detection")
            else:
                floor_mask = floor_mask_label
            labels_to_use = instance_ids
    
    # Step 4: Re-align using label-based floor detection if available
    if floor_mask is not None and np.any(floor_mask):
        print("\n" + "="*60)
        print("Stage 4: Re-aligning floor using label-based detection")
        print("="*60)
        print(f"  Using {'semantic' if use_semantic else 'instance'} labels with floor label ID: {floor_label_id}")
        base_points, rotation_matrix, translation = align_floor_to_z0(
            base_points_original, labels=labels_to_use, floor_mask=floor_mask
        )
    else:
        print("\n" + "="*60)
        print("Stage 4: Using initial Z-based alignment (label detection failed)")
        print("="*60)
        base_points = base_points_prelim
        rotation_matrix = rotation_matrix_prelim
        translation = translation_prelim
    
    # Step 5: Detect walls using semantic color consistency
    print("\n" + "="*60)
    print("Stage 5: Detecting walls from semantic/instance labels")
    print("="*60)
    wall_label_ids = []
    wall_mask = None
    wall_label_stats = {}
    
    # First try: Detect walls directly from semantic color consistency
    # Since walls have consistent semantic colors, we can identify them by looking for
    # labels that form tall vertical structures
    if len(labels_to_use) == len(base_points):
        print("  Method 1: Detecting walls from semantic color consistency...")
        min_z = np.min(base_points[:, 2])
        max_z = np.max(base_points[:, 2])
        floor_z_min = max(min_z, -0.2)
        floor_z_max = min(max_z, floor_z_min + 3.0)
        
        wall_label_ids, wall_mask, wall_label_stats = detect_wall_labels_from_semantic(
            base_points, labels_to_use, floor_z_min, floor_z_max,
            min_wall_height=0.6,  # Minimum 60cm height for walls (more conservative)
            min_points_per_label=150  # Minimum 150 points per label (more conservative)
        )
    
    # Fallback: Use grid-based detection if semantic-based detection failed
    if wall_mask is None or not np.any(wall_mask):
        print("\n  Method 2: Falling back to grid-based wall detection...")
        
        # Rebuild grid with final aligned points for wall detection
        pc_inst_dict_final = {}
        for inst_id in np.unique(instance_ids):
            if inst_id == 0:
                continue
            inst_mask = instance_ids == inst_id
            if np.sum(inst_mask) >= 50:
                pc_inst_dict_final[inst_id] = base_points[inst_mask]
        
        bboxes_final = compute_3d_bboxes(base_points, instance_ids, semantic_labels, min_points=50)
        
        # Recompute bounds after final alignment
        min_x = np.min(base_points[:, 0])
        max_x = np.max(base_points[:, 0])
        min_y = np.min(base_points[:, 1])
        max_y = np.max(base_points[:, 1])
        min_z = np.min(base_points[:, 2])
        max_z = np.max(base_points[:, 2])
        
        x_range = max_x - min_x
        y_range = max_y - min_y
        padding_x = max(x_range * 0.15, 1.0)
        padding_y = max(y_range * 0.15, 1.0)
        
        min_x = min_x - padding_x
        max_x = max_x + padding_x
        min_y = min_y - padding_y
        max_y = max_y + padding_y
        
        floor_z_min = max(min_z, -0.2)
        floor_z_max = min(max_z, floor_z_min + 3.0)
        grid_resolution = 0.05  # 5cm resolution (smaller cells for higher detail)
        
        floor_grid_final = create_floor_occupancy_grid(
            bboxes_final, pc_inst_dict_final, floor_z_min, floor_z_max,
            min_x, max_x, min_y, max_y, grid_resolution
        )
        
        # Detect wall labels from the final grid
        if len(labels_to_use) == len(base_points):
            wall_label_ids, wall_mask, wall_label_stats = detect_wall_labels_from_grid(
                base_points, labels_to_use, floor_grid_final,
                grid_resolution, min_x, min_y, floor_z_min, floor_z_max,
                min_dominance_ratio=0.3
            )
    else:
        # If semantic-based detection succeeded, still need to compute bboxes and grid for visualization
        pc_inst_dict_final = {}
        for inst_id in np.unique(instance_ids):
            if inst_id == 0:
                continue
            inst_mask = instance_ids == inst_id
            if np.sum(inst_mask) >= 50:
                pc_inst_dict_final[inst_id] = base_points[inst_mask]
        
        bboxes_final = compute_3d_bboxes(base_points, instance_ids, semantic_labels, min_points=50)
    
    if wall_mask is not None and np.any(wall_mask):
        print(f"\n  Wall detection complete:")
        print(f"    Found {len(wall_label_ids)} wall label(s)")
        print(f"    Total wall points: {np.sum(wall_mask):,}")
        
        if RERUN_AVAILABLE:
            # Visualize wall points
            wall_points = base_points[wall_mask]
            wall_colors_viz = np.full((len(wall_points), 3), [200, 50, 50], dtype=np.uint8)
            rr.log("pc/walls_label_detected", rr.Points3D(
                positions=wall_points.astype(np.float32),
                colors=wall_colors_viz,
                radii=0.015
            ))
            print("  Wall points visualized in Rerun (red)")
    else:
        print("\n  Wall detection failed or no walls found")
    
    # Visualize point clouds in Rerun
    if RERUN_AVAILABLE:
        print("\n" + "="*60)
        print("Stage: Visualizing point clouds")
        print("="*60)
        
        # Visualize RGB point cloud if available
        if rgb_colors is not None:
            print(f"  Logging RGB point cloud ({len(base_points):,} points)...")
            # Downsample if too many points for performance
            if len(base_points) > 1000000:
                sample_indices = np.random.choice(len(base_points), 1000000, replace=False)
                pc_sample = base_points[sample_indices].astype(np.float32)
                rgb_sample = rgb_colors[sample_indices]
                print(f"  Downsampled to {len(pc_sample):,} points for visualization")
            else:
                pc_sample = base_points.astype(np.float32)
                rgb_sample = rgb_colors
            
            rr.log("pc/world_rgb", rr.Points3D(
                positions=pc_sample,
                colors=rgb_sample,
                radii=0.01
            ))
            print("  ✓ RGB point cloud visualized")
        
        # Visualize instance-colored point cloud
        print(f"  Logging instance-colored point cloud ({len(base_points):,} points)...")
        if len(base_points) > 1000000:
            sample_indices = np.random.choice(len(base_points), 1000000, replace=False)
            pc_sample = base_points[sample_indices].astype(np.float32)
            inst_ids_sample = instance_ids[sample_indices]
        else:
            pc_sample = base_points.astype(np.float32)
            inst_ids_sample = instance_ids
        
        # Generate colors from instance IDs
        inst_colors = hash_colors_from_ids(inst_ids_sample)
        rr.log("pc/world_instance", rr.Points3D(
            positions=pc_sample,
            colors=inst_colors,
            radii=0.01
        ))
        print("  ✓ Instance-colored point cloud visualized")
        
        # Visualize original semantic point cloud from PLY file (aligned with base_points)
        if semantic_colors is not None:
            print(f"  Logging original semantic point cloud ({len(base_points):,} points)...")
            # Downsample if too many points for performance
            if len(base_points) > 1000000:
                sample_indices = np.random.choice(len(base_points), 1000000, replace=False)
                pc_sample = base_points[sample_indices].astype(np.float32)
                colors_sample = semantic_colors[sample_indices]
                print(f"  Downsampled to {len(pc_sample):,} points for visualization")
            else:
                pc_sample = base_points.astype(np.float32)
                colors_sample = semantic_colors
            
            rr.log("pc/world_semantic_original", rr.Points3D(
                positions=pc_sample,
                colors=colors_sample,
                radii=0.01
            ))
            print("  ✓ Original semantic point cloud visualized (aligned)")
    
    # Use final bboxes computed during wall detection
    bboxes = bboxes_final
    
    print("\n" + "="*60)
    print("Stage: Computing 3D bounding boxes")
    print("="*60)
    print(f"Computed {len(bboxes)} bounding boxes")
    
    if RERUN_AVAILABLE:
        visualize_3d_bboxes(bboxes, "bboxes/3d")
    
    # Store RGB colors per object for visualization
    obj_rgb_colors = {}
    if rgb_colors is not None:
        print("\nComputing RGB colors per object...")
        for inst_id, bbox in bboxes.items():
            mask = instance_ids == inst_id
            if np.any(mask):
                inst_rgb = rgb_colors[mask]
                avg_rgb = np.median(inst_rgb, axis=0).astype(np.uint8)
                obj_rgb_colors[inst_id] = avg_rgb.tolist()
    
    # Build point cloud dictionary for final occupancy grid
    print("\n" + "="*60)
    print("Stage: Building final occupancy grid and detecting areas")
    print("="*60)
    # Use pc_inst_dict_final which was already computed
    pc_inst_dict = pc_inst_dict_final
    
    # Separate into floors and create occupancy grid (uses final grid with proper floor/wall classification)
    # Pass base_points to ensure all points are included in bounds computation
    # Pass wall_mask to mark wall cells in the occupancy grid
    floors = separate_into_floors_by_height(bboxes, pc_inst_dict, all_points=base_points, wall_mask=wall_mask)
    
    if not floors:
        print("\nERROR: Failed to create occupancy grid!")
        return
    
    # Create RGB top-view image if RGB data is available
    floor_data = list(floors.values())[0]
    floor_grid = floor_data.get('occupancy_grid')
    
    # Visualize object-to-area assignments if areas are available
    if RERUN_AVAILABLE and 'image_areas' in floor_data and floor_data['image_areas']:
        print("\n" + "="*60)
        print("Stage: Visualizing object-to-area assignments")
        print("="*60)
        bounds = floor_data.get('bounds', {})
        min_x = bounds.get('min_x', 0)
        min_y = bounds.get('min_y', 0)
        grid_resolution = floor_data.get('grid_resolution', 0.1)
        visualize_object_area_assignments(
            floor_data['image_areas'], bboxes, floor_grid,
            grid_resolution, min_x, min_y, "object_area_assignments"
        )
    
    # Visualize 2D traversable map
    if floor_grid is not None:
        bounds = floor_data.get('bounds', {})
        min_x = bounds.get('min_x', 0)
        min_y = bounds.get('min_y', 0)
        grid_resolution = floor_data.get('grid_resolution', 0.1)
        
        print("\n" + "="*60)
        print("Stage: Creating 2D traversable map")
        print("="*60)
        visualize_traversable_map_2d(floor_grid, grid_resolution, min_x, min_y)
    
    if floor_grid is not None and rgb_colors is not None:
        print("\n" + "="*60)
        print("Stage: Creating RGB top-view image")
        print("="*60)
        bounds = floor_data.get('bounds', {})
        min_x = bounds.get('min_x', 0)
        min_y = bounds.get('min_y', 0)
        grid_resolution = floor_data.get('grid_resolution', 0.1)
        image_areas = floor_data.get('image_areas', [])
        
        script_dir = Path(__file__).parent.absolute()
        # save_dir = script_dir / "saved_frames"
        save_dir = script_dir
        create_top_view_rgb_image(
            floor_grid, grid_resolution, min_x, min_y,
            bboxes, obj_rgb_colors, image_areas,
            base_points, rgb_colors, instance_ids, save_dir,
            wall_mask=wall_mask
        )
    
    print("\n" + "="*60)
    print("Stage: Exporting JSON")
    print("="*60)
    try:
        script_dir = Path(__file__).parent.absolute()
        # save_dir = script_dir / "saved_frames"
        save_dir = script_dir
        analysis_json_path = save_objects_and_areas_to_json(save_dir, bboxes, floors, filename="analysis.json")
        print(f"\nTo label objects and areas, run:")
        print(f"  python label_objects_and_areas.py {analysis_json_path}")
        print(f"Or use PyQt5 version:")
        print(f"  python label_objects_and_areas.py {analysis_json_path} --pyqt")
    except Exception as e:
        print(f"JSON save failure: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Stage: Done!")
    print("="*60)


if __name__ == "__main__":
    main()

