"""GUI application to label objects and areas with names.

Allows editing labels for objects and areas detected in the SLAM map analysis.
Saves labels to JSON file that can be merged with the analysis.json output.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    import cv2
    from PIL import Image, ImageTk
    CV2_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    PIL_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
                                 QLabel, QLineEdit, QFileDialog, QMessageBox, QSplitter)
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap, QImage
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


def find_default_analysis_json() -> Optional[Path]:
    """Find the default analysis.json file relative to the script directory."""
    script_dir = Path(__file__).parent.absolute()
    
    possible_paths = [
        script_dir / "analysis.json",
        script_dir / "saved_frames" / "analysis.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def load_saved_map_image(analysis_json_path: Path) -> Optional[np.ndarray]:
    """Try to load a saved top-view map image if it exists.
    
    Looks for the map image in the same directory as the analysis JSON.
    Prefers RGB version if available, falls back to regular map.
    """
    if not CV2_AVAILABLE:
        return None
    
    script_dir = Path(__file__).parent.absolute()
    
    # Try locations relative to analysis JSON first, then script directory
    possible_paths = [
        analysis_json_path.parent / "top_view_rgb_map.png",  # RGB version first
        analysis_json_path.parent / "top_view_map.png",      # Regular version
        script_dir / "top_view_rgb_map.png",
        script_dir / "top_view_map.png",
    ]
    
    for img_path in possible_paths:
        if img_path.exists():
            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    print(f"Loaded saved map image: {img_path}")
                    return img_rgb
            except Exception as e:
                print(f"Warning: Could not load map image from {img_path}: {e}")
    
    return None


def generate_map_image_from_analysis(analysis: Dict, grid_resolution: float = 0.1, highlighted_object_id: Optional[int] = None) -> Optional[np.ndarray]:
    """Generate a top-view map image from analysis JSON data."""
    if not CV2_AVAILABLE:
        return None
    
    try:
        all_x = []
        all_y = []
        
        for obj in analysis.get('objects', []):
            bbox = obj.get('bbox', {})
            points = bbox.get('points', [])
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                all_x.extend(xs)
                all_y.extend(ys)
        
        for area in analysis.get('areas', []):
            # Support both new polygon format and old bbox format
            polygon = area.get('polygon', {})
            bbox = area.get('bbox', {})
            points = polygon.get('points', []) if polygon else bbox.get('points', [])
            if points:
                xs = [p[0] for p in points if len(p) >= 2]
                ys = [p[1] for p in points if len(p) >= 2]
                all_x.extend(xs)
                all_y.extend(ys)
        
        if not all_x or not all_y:
            return None
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        padding = max((max_x - min_x) * 0.1, (max_y - min_y) * 0.1, 1.0)
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        grid_width = int(np.ceil((max_x - min_x) / grid_resolution)) + 2
        grid_height = int(np.ceil((max_y - min_y) / grid_resolution)) + 2
        grid = np.full((grid_height, grid_width), 3, dtype=np.int32)
        
        for area in analysis.get('areas', []):
            polygon = area.get('polygon', {})
            bbox = area.get('bbox', {})
            points = polygon.get('points', []) if polygon else bbox.get('points', [])
            if len(points) >= 3:
                xs = [p[0] for p in points if len(p) >= 2]
                ys = [p[1] for p in points if len(p) >= 2]
                if xs and ys:
                    area_min_x, area_max_x = min(xs), max(xs)
                    area_min_y, area_max_y = min(ys), max(ys)
                    
                    gx_min = int((area_min_x - min_x) / grid_resolution)
                    gx_max = int((area_max_x - min_x) / grid_resolution) + 1
                    gy_min = int((area_min_y - min_y) / grid_resolution)
                    gy_max = int((area_max_y - min_y) / grid_resolution) + 1
                    
                    gx_min = max(0, min(gx_min, grid_width - 1))
                    gx_max = max(0, min(gx_max, grid_width))
                    gy_min = max(0, min(gy_min, grid_height - 1))
                    gy_max = max(0, min(gy_max, grid_height))
                    
                    grid[gy_min:gy_max, gx_min:gx_max] = 0
        
        for obj in analysis.get('objects', []):
            bbox = obj.get('bbox', {})
            points = bbox.get('points', [])
            if points:
                xs = [p[0] for p in points if len(p) >= 2]
                ys = [p[1] for p in points if len(p) >= 2]
                if xs and ys:
                    obj_x = sum(xs) / len(xs)
                    obj_y = sum(ys) / len(ys)
                    
                    gx = int((obj_x - min_x) / grid_resolution)
                    gy = int((obj_y - min_y) / grid_resolution)
                    
                    if 0 <= gx < grid_width and 0 <= gy < grid_height:
                        grid[gy, gx] = 1
        
        scale_factor = 10
        img_height = grid_height * scale_factor
        img_width = grid_width * scale_factor
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        colors = {
            0: [200, 200, 200],
            1: [100, 100, 100],
            2: [50, 50, 50],
            3: [240, 240, 240]
        }
        
        for gy in range(grid_height):
            for gx in range(grid_width):
                cell_value = grid[gy, gx]
                color = colors.get(cell_value, [128, 128, 128])
                
                start_y = gy * scale_factor
                end_y = (gy + 1) * scale_factor
                start_x = gx * scale_factor
                end_x = (gx + 1) * scale_factor
                
                img[start_y:end_y, start_x:end_x] = color
        
        for area in analysis.get('areas', []):
            polygon = area.get('polygon', {})
            bbox = area.get('bbox', {})
            points = polygon.get('points', []) if polygon else bbox.get('points', [])
            if len(points) >= 3:
                pts = []
                for p in points:
                    if len(p) >= 2:
                        gx = int((p[0] - min_x) / grid_resolution) * scale_factor
                        gy = int((p[1] - min_y) / grid_resolution) * scale_factor
                        pts.append([gx, gy])
                
                if len(pts) >= 3:
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                    
                    if pts.shape[0] > 0:
                        center = pts.mean(axis=0).astype(int)
                        area_id = area.get('id', '')
                        cv2.putText(img, f"A{area_id}", tuple(center), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for obj in analysis.get('objects', []):
            bbox = obj.get('bbox', {})
            points = bbox.get('points', [])
            obj_id = obj.get('id', '')
            is_highlighted = (highlighted_object_id is not None and obj_id == highlighted_object_id)
            
            if points:
                xs = [p[0] for p in points if len(p) >= 2]
                ys = [p[1] for p in points if len(p) >= 2]
                if xs and ys:
                    obj_x = sum(xs) / len(xs)
                    obj_y = sum(ys) / len(ys)
                    obj_min_x, obj_max_x = min(xs), max(xs)
                    obj_min_y, obj_max_y = min(ys), max(ys)
                    
                    gx_center = int((obj_x - min_x) / grid_resolution) * scale_factor
                    gy_center = int((obj_y - min_y) / grid_resolution) * scale_factor
                    gx_min = int((obj_min_x - min_x) / grid_resolution) * scale_factor
                    gx_max = int((obj_max_x - min_x) / grid_resolution) * scale_factor
                    gy_min = int((obj_min_y - min_y) / grid_resolution) * scale_factor
                    gy_max = int((obj_max_y - min_y) / grid_resolution) * scale_factor
                    
                    if (0 <= gx_min < img_width and 0 <= gx_max < img_width and
                        0 <= gy_min < img_height and 0 <= gy_max < img_height):
                        if is_highlighted:
                            cv2.rectangle(img, (gx_min - 3, gy_min - 3), (gx_max + 3, gy_max + 3), (0, 255, 255), 4)
                            cv2.rectangle(img, (gx_min, gy_min), (gx_max, gy_max), (255, 255, 0), 3)
                        else:
                            cv2.rectangle(img, (gx_min, gy_min), (gx_max, gy_max), (255, 0, 0), 2)
                        
                        if 0 <= gx_center < img_width and 0 <= gy_center < img_height:
                            center_color = (0, 255, 255) if is_highlighted else (255, 0, 0)
                            center_size = 6 if is_highlighted else 4
                            cv2.circle(img, (gx_center, gy_center), center_size, center_color, -1)
                        
                        area_id = obj.get('area_id', -1)
                        label_y = max(15, gy_min - 5)
                        label_text = f"O{obj_id}"
                        if area_id >= 0:
                            label_text += f" (A{area_id})"
                        
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        
                        text_bg_color = (0, 255, 255) if is_highlighted else (255, 255, 255)
                        cv2.rectangle(img, 
                                     (gx_min, label_y - text_height - 5),
                                     (gx_min + text_width + 4, label_y + baseline),
                                     text_bg_color, -1)
                        
                        text_color = (0, 0, 0) if is_highlighted else (255, 0, 0)
                        cv2.putText(img, label_text, (gx_min + 2, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        return img
    
    except Exception as e:
        print(f"Error generating map image: {e}")
        return None


def load_analysis_json(json_path: Path) -> Optional[Dict]:
    """Load the analysis JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None


def merge_labels_with_analysis(analysis_json_path: Path, labels_json_path: Path, output_json_path: Optional[Path] = None) -> Path:
    """Merge labels into the analysis JSON file and save the result.
    
    Args:
        analysis_json_path: Path to the original analysis.json
        labels_json_path: Path to the labels.json file
        output_json_path: Optional path for output (default: overwrites analysis.json)
        
    Returns:
        Path to the merged JSON file
    """
    # Load analysis
    analysis = load_analysis_json(analysis_json_path)
    if analysis is None:
        raise ValueError(f"Could not load analysis JSON: {analysis_json_path}")
    
    # Load labels
    labels = load_analysis_json(labels_json_path)
    if labels is None:
        raise ValueError(f"Could not load labels JSON: {labels_json_path}")
    
    # Merge labels into objects
    if 'objects' in analysis:
        object_labels = labels.get('objects', {})
        default_label = labels.get('default_object_label', '')
        for obj in analysis['objects']:
            obj_id = obj.get('id')
            if obj_id is not None:
                obj['label'] = object_labels.get(str(obj_id), default_label)
    
    # Merge labels into areas
    if 'areas' in analysis:
        area_labels = labels.get('areas', {})
        default_label = labels.get('default_area_label', '')
        for area in analysis['areas']:
            area_id = area.get('id')
            if area_id is not None:
                area['label'] = area_labels.get(str(area_id), default_label)
    
    # Save merged result
    if output_json_path is None:
        output_json_path = analysis_json_path
    
    with open(output_json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Merged labels saved to: {output_json_path}")
    return output_json_path


def label_with_tkinter(analysis_json_path: Path, labels_json_path: Optional[Path] = None, show_map: bool = True) -> Dict:
    """Create a tkinter GUI to label objects and areas."""
    if not TKINTER_AVAILABLE:
        raise ImportError("tkinter is not available. Please install it or use PyQt version.")
    
    # Load analysis data
    analysis = load_analysis_json(analysis_json_path)
    if analysis is None:
        return {}
    
    # Load existing labels from analysis.json (prefer labels from analysis.json over labels.json)
    labels = {
        'objects': {},
        'areas': {},
        'default_object_label': '',
        'default_area_label': ''
    }
    
    # First, load labels from analysis.json (if they exist)
    for obj in analysis.get('objects', []):
        obj_id = str(obj.get('id', ''))
        existing_label = obj.get('label', '')
        labels['objects'][obj_id] = existing_label
    
    for area in analysis.get('areas', []):
        area_id = str(area.get('id', ''))
        existing_label = area.get('label', '')
        labels['areas'][area_id] = existing_label
    
    # Optionally, also check labels.json as a fallback (for backward compatibility)
    if labels_json_path and labels_json_path.exists():
        try:
            with open(labels_json_path, 'r') as f:
                existing_labels = json.load(f)
                # Only use labels.json if label is empty in analysis.json
                for obj_id, label in existing_labels.get('objects', {}).items():
                    if obj_id in labels['objects'] and not labels['objects'][obj_id]:
                        labels['objects'][obj_id] = label
                for area_id, label in existing_labels.get('areas', {}).items():
                    if area_id in labels['areas'] and not labels['areas'][area_id]:
                        labels['areas'][area_id] = label
        except Exception:
            pass
    
    map_image = None
    map_image_original = None
    if show_map and CV2_AVAILABLE and PIL_AVAILABLE:
        map_image_original = load_saved_map_image(analysis_json_path)
        if map_image_original is None:
            map_image_original = generate_map_image_from_analysis(analysis)
            if map_image_original is not None:
                print("Generated map image from JSON data (saved map not found)")
        map_image = map_image_original
    
    # Create main window
    root = tk.Tk()
    root.title("Label Objects and Areas")
    if map_image is not None:
        root.geometry("1400x800")
    else:
        root.geometry("1000x700")
    
    if map_image is not None:
        main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        map_frame = ttk.Frame(main_paned)
        map_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(map_frame, text="Map View", font=("Arial", 12, "bold")).pack(pady=5)
        
        map_img_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        map_img_pil = Image.fromarray(map_img_rgb)
        
        max_display_size = 600
        if map_img_pil.width > max_display_size or map_img_pil.height > max_display_size:
            ratio = min(max_display_size / map_img_pil.width, max_display_size / map_img_pil.height)
            new_width = int(map_img_pil.width * ratio)
            new_height = int(map_img_pil.height * ratio)
            map_img_pil = map_img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        map_img_tk = ImageTk.PhotoImage(map_img_pil)
        map_label = ttk.Label(map_frame, image=map_img_tk)
        map_label.image = map_img_tk
        map_label.pack(padx=5, pady=5)
        
        ttk.Label(map_frame, text="Red: Objects (O#), Green: Areas (A#), Yellow: Selected", font=("Arial", 9)).pack(pady=2)
        main_paned.add(map_frame, weight=1)
        
        def update_map_highlight(obj_id_to_highlight):
            if map_image_original is None:
                return
            if isinstance(obj_id_to_highlight, str):
                try:
                    obj_id_to_highlight = int(obj_id_to_highlight)
                except ValueError:
                    obj_id_to_highlight = None
            
            highlighted_img = generate_map_image_from_analysis(analysis, highlighted_object_id=obj_id_to_highlight)
            if highlighted_img is not None:
                map_img_rgb = cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB)
                map_img_pil = Image.fromarray(map_img_rgb)
                
                max_display_size = 600
                if map_img_pil.width > max_display_size or map_img_pil.height > max_display_size:
                    ratio = min(max_display_size / map_img_pil.width, max_display_size / map_img_pil.height)
                    new_width = int(map_img_pil.width * ratio)
                    new_height = int(map_img_pil.height * ratio)
                    map_img_pil = map_img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                map_img_tk_new = ImageTk.PhotoImage(map_img_pil)
                map_label.configure(image=map_img_tk_new)
                map_label.image = map_img_tk_new
        
        tables_frame = ttk.Frame(main_paned)
        tables_frame.pack(fill=tk.BOTH, expand=True)
        main_paned.add(tables_frame, weight=1)
    else:
        tables_frame = root
        update_map_highlight = None
    
    notebook = ttk.Notebook(tables_frame if map_image is not None else root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    obj_frame = ttk.Frame(notebook)
    notebook.add(obj_frame, text="Objects")
    
    obj_tree_frame = ttk.Frame(obj_frame)
    obj_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    obj_tree = ttk.Treeview(obj_tree_frame, columns=('ID', 'Label', 'Points', 'Area ID'), show='headings', height=20)
    obj_tree.heading('ID', text='Object ID')
    obj_tree.heading('Label', text='Label')
    obj_tree.heading('Points', text='Points')
    obj_tree.heading('Area ID', text='Area ID')
    obj_tree.column('ID', width=100)
    obj_tree.column('Label', width=300)
    obj_tree.column('Points', width=100)
    obj_tree.column('Area ID', width=100)
    
    obj_scrollbar = ttk.Scrollbar(obj_tree_frame, orient=tk.VERTICAL, command=obj_tree.yview)
    obj_tree.configure(yscrollcommand=obj_scrollbar.set)
    obj_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    obj_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    for obj in analysis.get('objects', []):
        obj_id = obj.get('id', '')
        label = labels['objects'].get(str(obj_id), '')
        num_points = obj.get('num_points', 0)
        area_id = obj.get('area_id', -1)
        obj_tree.insert('', tk.END, values=(obj_id, label, num_points, area_id))
    
    obj_edit_frame = ttk.Frame(obj_frame)
    obj_edit_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(obj_edit_frame, text="Selected Object ID:").pack(side=tk.LEFT, padx=5)
    obj_id_var = tk.StringVar()
    obj_id_entry = ttk.Entry(obj_edit_frame, textvariable=obj_id_var, state='readonly', width=10)
    obj_id_entry.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(obj_edit_frame, text="Label:").pack(side=tk.LEFT, padx=5)
    obj_label_var = tk.StringVar()
    obj_label_entry = ttk.Entry(obj_edit_frame, textvariable=obj_label_var, width=30)
    obj_label_entry.pack(side=tk.LEFT, padx=5)
    
    def update_obj_label():
        selection = obj_tree.selection()
        if not selection:
            return
        item = obj_tree.item(selection[0])
        obj_id = item['values'][0]
        new_label = obj_label_var.get().strip()
        if new_label:
            labels['objects'][str(obj_id)] = new_label
            obj_tree.item(selection[0], values=(obj_id, new_label, item['values'][2], item['values'][3]))
    
    obj_update_btn = ttk.Button(obj_edit_frame, text="Update Label", command=update_obj_label)
    obj_update_btn.pack(side=tk.LEFT, padx=5)
    
    def on_obj_select(event):
        selection = obj_tree.selection()
        if selection:
            item = obj_tree.item(selection[0])
            obj_id = item['values'][0]
            obj_id_var.set(obj_id)
            obj_label_var.set(item['values'][1])
            
            if update_map_highlight is not None:
                update_map_highlight(obj_id)
    
    obj_tree.bind('<<TreeviewSelect>>', on_obj_select)
    
    area_frame = ttk.Frame(notebook)
    notebook.add(area_frame, text="Areas")
    
    area_tree_frame = ttk.Frame(area_frame)
    area_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    area_tree = ttk.Treeview(area_tree_frame, columns=('ID', 'Label', 'Size'), show='headings', height=20)
    area_tree.heading('ID', text='Area ID')
    area_tree.heading('Label', text='Label')
    area_tree.heading('Size', text='Size (W x H)')
    area_tree.column('ID', width=100)
    area_tree.column('Label', width=300)
    area_tree.column('Size', width=200)
    
    area_scrollbar = ttk.Scrollbar(area_tree_frame, orient=tk.VERTICAL, command=area_tree.yview)
    area_tree.configure(yscrollcommand=area_scrollbar.set)
    area_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    area_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    for area in analysis.get('areas', []):
        area_id = area.get('id', '')
        label = labels['areas'].get(str(area_id), '')
        polygon = area.get('polygon', {})
        bbox = area.get('bbox', {})
        points = polygon.get('points', []) if polygon else bbox.get('points', [])
        if len(points) >= 3:
            xs = [p[0] for p in points if len(p) >= 2]
            ys = [p[1] for p in points if len(p) >= 2]
            if xs and ys:
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                size_str = f"{width:.2f} x {height:.2f}"
            else:
                size_str = "N/A"
        else:
            size_str = "N/A"
        area_tree.insert('', tk.END, values=(area_id, label, size_str))
    
    area_edit_frame = ttk.Frame(area_frame)
    area_edit_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(area_edit_frame, text="Selected Area ID:").pack(side=tk.LEFT, padx=5)
    area_id_var = tk.StringVar()
    area_id_entry = ttk.Entry(area_edit_frame, textvariable=area_id_var, state='readonly', width=10)
    area_id_entry.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(area_edit_frame, text="Label:").pack(side=tk.LEFT, padx=5)
    area_label_var = tk.StringVar()
    area_label_entry = ttk.Entry(area_edit_frame, textvariable=area_label_var, width=30)
    area_label_entry.pack(side=tk.LEFT, padx=5)
    
    def update_area_label():
        selection = area_tree.selection()
        if not selection:
            return
        item = area_tree.item(selection[0])
        area_id = item['values'][0]
        new_label = area_label_var.get().strip()
        if new_label:
            labels['areas'][str(area_id)] = new_label
            area_tree.item(selection[0], values=(area_id, new_label, item['values'][2]))
    
    area_update_btn = ttk.Button(area_edit_frame, text="Update Label", command=update_area_label)
    area_update_btn.pack(side=tk.LEFT, padx=5)
    
    def on_area_select(event):
        selection = area_tree.selection()
        if selection:
            item = area_tree.item(selection[0])
            area_id_var.set(item['values'][0])
            area_label_var.set(item['values'][1])
    
    area_tree.bind('<<TreeviewSelect>>', on_area_select)
    
    def save_labels():
        for obj in analysis.get('objects', []):
            obj_id = str(obj.get('id', ''))
            if obj_id in labels['objects']:
                obj['label'] = labels['objects'][obj_id]
        
        for area in analysis.get('areas', []):
            area_id = str(area.get('id', ''))
            if area_id in labels['areas']:
                area['label'] = labels['areas'][area_id]
        
        # Save directly to analysis.json
        with open(analysis_json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        messagebox.showinfo("Success", f"Labels saved directly to:\n{analysis_json_path}")
    
    save_btn = ttk.Button(root if map_image is None else tables_frame, text="Save Labels", command=save_labels)
    save_btn.pack(pady=5)
    
    root.mainloop()
    
    return labels


def label_with_pyqt5(analysis_json_path: Path, labels_json_path: Optional[Path] = None, show_map: bool = True) -> Dict:
    """Create a PyQt5 GUI to label objects and areas."""
    if not PYQT_AVAILABLE:
        raise ImportError("PyQt5 is not available. Please install it with: pip install PyQt5")
    
    app = QApplication([])
    
    # Load analysis data
    analysis = load_analysis_json(analysis_json_path)
    if analysis is None:
        return {}
    
    labels = {
        'objects': {},
        'areas': {}
    }
    
    for obj in analysis.get('objects', []):
        obj_id = str(obj.get('id', ''))
        labels['objects'][obj_id] = obj.get('label', '')
    
    for area in analysis.get('areas', []):
        area_id = str(area.get('id', ''))
        labels['areas'][area_id] = area.get('label', '')
    
    if labels_json_path and labels_json_path.exists():
        try:
            with open(labels_json_path, 'r') as f:
                existing_labels = json.load(f)
                for obj_id, label in existing_labels.get('objects', {}).items():
                    if obj_id in labels['objects'] and not labels['objects'][obj_id]:
                        labels['objects'][obj_id] = label
                for area_id, label in existing_labels.get('areas', {}).items():
                    if area_id in labels['areas'] and not labels['areas'][area_id]:
                        labels['areas'][area_id] = label
        except Exception:
            pass
    
    map_image = None
    map_image_original = None
    if show_map and CV2_AVAILABLE:
        map_image_original = load_saved_map_image(analysis_json_path)
        if map_image_original is None:
            map_image_original = generate_map_image_from_analysis(analysis)
            if map_image_original is not None:
                print("Generated map image from JSON data (saved map not found)")
        map_image = map_image_original
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Label Objects and Areas")
    if map_image is not None:
        window.setGeometry(100, 100, 1600, 900)
    else:
        window.setGeometry(100, 100, 1200, 800)
    
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout(central_widget)
    
    main_splitter = QSplitter(Qt.Horizontal)
    layout.addWidget(main_splitter)
    
    if map_image is not None:
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)
        map_label_title = QLabel("<h2>Map View</h2>")
        map_layout.addWidget(map_label_title)
        
        map_img_rgb = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        height, width, channel = map_img_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(map_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        max_display_size = 600
        if width > max_display_size or height > max_display_size:
            ratio = min(max_display_size / width, max_display_size / height)
            q_image = q_image.scaled(int(width * ratio), int(height * ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        map_pixmap = QPixmap.fromImage(q_image)
        map_pixmap_label = QLabel()
        map_pixmap_label.setPixmap(map_pixmap)
        map_pixmap_label.setAlignment(Qt.AlignCenter)
        map_layout.addWidget(map_pixmap_label)
        
        legend_label = QLabel("Red: Objects (O#), Green: Areas (A#), Yellow: Selected")
        legend_label.setAlignment(Qt.AlignCenter)
        map_layout.addWidget(legend_label)
        
        main_splitter.addWidget(map_widget)
        
        def update_map_highlight(obj_id_to_highlight):
            if map_image_original is None:
                return
            if isinstance(obj_id_to_highlight, str):
                try:
                    obj_id_to_highlight = int(obj_id_to_highlight)
                except ValueError:
                    obj_id_to_highlight = None
            
            highlighted_img = generate_map_image_from_analysis(analysis, highlighted_object_id=obj_id_to_highlight)
            if highlighted_img is not None:
                map_img_rgb = cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB)
                height, width, channel = map_img_rgb.shape
                bytes_per_line = 3 * width
                q_image = QImage(map_img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                max_display_size = 600
                if width > max_display_size or height > max_display_size:
                    ratio = min(max_display_size / width, max_display_size / height)
                    q_image = q_image.scaled(int(width * ratio), int(height * ratio), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                map_pixmap_new = QPixmap.fromImage(q_image)
                map_pixmap_label.setPixmap(map_pixmap_new)
    else:
        update_map_highlight = None
    
    tables_splitter = QSplitter(Qt.Horizontal)
    main_splitter.addWidget(tables_splitter)
    
    splitter = QSplitter(Qt.Horizontal)
    tables_splitter.addWidget(splitter)
    
    obj_widget = QWidget()
    obj_layout = QVBoxLayout(obj_widget)
    obj_label_title = QLabel("<h2>Objects</h2>")
    obj_layout.addWidget(obj_label_title)
    
    obj_table = QTableWidget()
    obj_table.setColumnCount(4)
    obj_table.setHorizontalHeaderLabels(['Object ID', 'Label', 'Points', 'Area ID'])
    obj_table.setRowCount(len(analysis.get('objects', [])))
    
    for i, obj in enumerate(analysis.get('objects', [])):
        obj_id = str(obj.get('id', ''))
        label = labels['objects'].get(obj_id, '')
        num_points = obj.get('num_points', 0)
        area_id = obj.get('area_id', -1)
        
        obj_table.setItem(i, 0, QTableWidgetItem(str(obj_id)))
        obj_table.setItem(i, 1, QTableWidgetItem(label))
        obj_table.setItem(i, 2, QTableWidgetItem(str(num_points)))
        obj_table.setItem(i, 3, QTableWidgetItem(str(area_id)))
    
    obj_table.resizeColumnsToContents()
    obj_layout.addWidget(obj_table)
    
    obj_edit_layout = QHBoxLayout()
    obj_edit_layout.addWidget(QLabel("Selected Object ID:"))
    obj_id_display = QLineEdit()
    obj_id_display.setReadOnly(True)
    obj_id_display.setMaximumWidth(100)
    obj_edit_layout.addWidget(obj_id_display)
    
    obj_edit_layout.addWidget(QLabel("Label:"))
    obj_label_edit = QLineEdit()
    obj_label_edit.setMaximumWidth(300)
    obj_edit_layout.addWidget(obj_label_edit)
    
    def update_obj_from_table():
        current_row = obj_table.currentRow()
        if current_row >= 0:
            obj_id = obj_table.item(current_row, 0).text()
            obj_id_display.setText(obj_id)
            obj_label_edit.setText(obj_table.item(current_row, 1).text())
            
            if update_map_highlight is not None:
                update_map_highlight(obj_id)
    
    def update_obj_label():
        current_row = obj_table.currentRow()
        if current_row >= 0:
            obj_id = obj_table.item(current_row, 0).text()
            new_label = obj_label_edit.text().strip()
            if new_label:
                labels['objects'][obj_id] = new_label
                obj_table.item(current_row, 1).setText(new_label)
    
    obj_update_btn = QPushButton("Update Label")
    obj_update_btn.clicked.connect(update_obj_label)
    obj_edit_layout.addWidget(obj_update_btn)
    
    obj_table.itemSelectionChanged.connect(update_obj_from_table)
    obj_layout.addLayout(obj_edit_layout)
    
    splitter.addWidget(obj_widget)
    
    area_widget = QWidget()
    area_layout = QVBoxLayout(area_widget)
    area_label_title = QLabel("<h2>Areas</h2>")
    area_layout.addWidget(area_label_title)
    
    area_table = QTableWidget()
    area_table.setColumnCount(3)
    area_table.setHorizontalHeaderLabels(['Area ID', 'Label', 'Size (W x H)'])
    area_table.setRowCount(len(analysis.get('areas', [])))
    
    for i, area in enumerate(analysis.get('areas', [])):
        area_id = str(area.get('id', ''))
        label = labels['areas'].get(area_id, '')
        polygon = area.get('polygon', {})
        bbox = area.get('bbox', {})
        points = polygon.get('points', []) if polygon else bbox.get('points', [])
        if len(points) >= 3:
            xs = [p[0] for p in points if len(p) >= 2]
            ys = [p[1] for p in points if len(p) >= 2]
            if xs and ys:
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                size_str = f"{width:.2f} x {height:.2f}"
            else:
                size_str = "N/A"
        else:
            size_str = "N/A"
        
        area_table.setItem(i, 0, QTableWidgetItem(str(area_id)))
        area_table.setItem(i, 1, QTableWidgetItem(label))
        area_table.setItem(i, 2, QTableWidgetItem(size_str))
    
    area_table.resizeColumnsToContents()
    area_layout.addWidget(area_table)
    
    area_edit_layout = QHBoxLayout()
    area_edit_layout.addWidget(QLabel("Selected Area ID:"))
    area_id_display = QLineEdit()
    area_id_display.setReadOnly(True)
    area_id_display.setMaximumWidth(100)
    area_edit_layout.addWidget(area_id_display)
    
    area_edit_layout.addWidget(QLabel("Label:"))
    area_label_edit = QLineEdit()
    area_label_edit.setMaximumWidth(300)
    area_edit_layout.addWidget(area_label_edit)
    
    def update_area_from_table():
        current_row = area_table.currentRow()
        if current_row >= 0:
            area_id = area_table.item(current_row, 0).text()
            area_id_display.setText(area_id)
            area_label_edit.setText(area_table.item(current_row, 1).text())
    
    def update_area_label():
        current_row = area_table.currentRow()
        if current_row >= 0:
            area_id = area_table.item(current_row, 0).text()
            new_label = area_label_edit.text().strip()
            if new_label:
                labels['areas'][area_id] = new_label
                area_table.item(current_row, 1).setText(new_label)
    
    area_update_btn = QPushButton("Update Label")
    area_update_btn.clicked.connect(update_area_label)
    area_edit_layout.addWidget(area_update_btn)
    
    area_table.itemSelectionChanged.connect(update_area_from_table)
    area_layout.addLayout(area_edit_layout)
    
    splitter.addWidget(area_widget)
    
    splitter.setSizes([600, 600])
    if map_image is not None:
        main_splitter.setSizes([600, 1000])
    
    def save_labels():
        for obj in analysis.get('objects', []):
            obj_id = str(obj.get('id', ''))
            if obj_id in labels['objects']:
                obj['label'] = labels['objects'][obj_id]
        
        for area in analysis.get('areas', []):
            area_id = str(area.get('id', ''))
            if area_id in labels['areas']:
                area['label'] = labels['areas'][area_id]
        
        # Save directly to analysis.json
        with open(analysis_json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        QMessageBox.information(window, "Success", f"Labels saved directly to:\n{analysis_json_path}")
    
    save_btn = QPushButton("Save Labels")
    save_btn.clicked.connect(save_labels)
    layout.addWidget(save_btn)
    
    window.show()
    app.exec_()
    
    return labels


def label_objects_and_areas(analysis_json_path: Optional[Path] = None, labels_json_path: Optional[Path] = None, use_pyqt: bool = False, show_map: bool = True) -> Dict:
    """Main function to label objects and areas using GUI.
    
    Args:
        analysis_json_path: Path to the analysis.json file (if None, will try to find default)
        labels_json_path: Optional path to save labels JSON (default: labels.json in same directory)
        use_pyqt: If True, use PyQt5, otherwise use tkinter
        show_map: If True, display map visualization (default: True)
        
    Returns:
        Dictionary containing labels for objects and areas
    """
    if analysis_json_path is None:
        analysis_json_path = find_default_analysis_json()
        if analysis_json_path is None:
            raise FileNotFoundError(
                "Could not find analysis.json in default locations.\n"
                "Please specify the path: python label_objects_and_areas.py <path/to/analysis.json>"
            )
        print(f"Using default path: {analysis_json_path}")
    
    if use_pyqt and PYQT_AVAILABLE:
        return label_with_pyqt5(analysis_json_path, labels_json_path, show_map)
    elif TKINTER_AVAILABLE:
        return label_with_tkinter(analysis_json_path, labels_json_path, show_map)
    else:
        raise ImportError("Neither tkinter nor PyQt5 is available. Please install one of them.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Label objects and areas from SLAM analysis")
    parser.add_argument("analysis_json", type=Path, nargs='?', default=None, 
                       help="Path to analysis.json file (optional, will auto-detect if not provided)")
    parser.add_argument("--labels-json", type=Path, help="Path to save labels JSON (default: labels.json)")
    parser.add_argument("--pyqt", action="store_true", help="Use PyQt5 instead of tkinter")
    parser.add_argument("--merge", action="store_true", help="Merge labels back into analysis.json after saving")
    parser.add_argument("--output", type=Path, help="Output path for merged JSON (only used with --merge)")
    parser.add_argument("--no-map", action="store_true", help="Don't show map visualization")
    
    args = parser.parse_args()
    
    # Find default path if not provided
    analysis_json = args.analysis_json
    if analysis_json is None:
        analysis_json = find_default_analysis_json()
        if analysis_json is None:
            script_dir = Path(__file__).parent.absolute()
            print("Error: Could not find analysis.json in default locations.")
            print("Default locations checked:")
            print(f"  - {script_dir / 'analysis.json'}")
            print(f"  - {script_dir / 'saved_frames' / 'analysis.json'}")
            print("\nPlease specify the path:")
            print("  python label_objects_and_areas.py <path/to/analysis.json>")
            exit(1)
        print(f"Using default path: {analysis_json}")
    
    # Run labeling GUI
    labels = label_objects_and_areas(analysis_json, args.labels_json, args.pyqt, show_map=not args.no_map)
    print(f"\nLabeled {len(labels.get('objects', {}))} objects and {len(labels.get('areas', {}))} areas")
    
    # Merge labels back into analysis.json if requested
    if args.merge:
        labels_path = args.labels_json if args.labels_json else analysis_json.parent / "labels.json"
        if labels_path.exists():
            merge_labels_with_analysis(analysis_json, labels_path, args.output)
        else:
            print(f"Warning: Labels file not found at {labels_path}. Run without --merge first.")

