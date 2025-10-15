"""
Example script demo'ing robot control with RGB-D and segmentation visualizations.

Options for random actions, as well as selection of robot action space
"""

import torch as th
import numpy as np
import cv2
import rerun as rr
import os
import rerun as rr
from scipy.spatial.transform import Rotation

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True


def choose_controllers(robot, random_selection=False):
    controller_choices = {}
    default_config = robot._default_controller_config
    for controller_name in robot.controller_order:
        options = list(sorted(default_config[controller_name].keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )
        controller_choices[controller_name] = choice
    return controller_choices


def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    og.log.info(f"Demo {__file__}\n" + "*" * 80 + "\n" + (main.__doc__ or "") + "\n" + "*" * 80)

    # Enable headless/offscreen rendering before environment creation if requested
    if headless:
        gm.HEADLESS = True
        # Force EGL by unsetting DISPLAY and setting KIT_USE_EGL
        os.environ.pop("DISPLAY", None)
        os.environ["KIT_USE_EGL"] = "1"

    # 1) Scene 선택
    scene_model = "Rs_int"
    if not quickstart:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    scene_cfg = {"type": "InteractiveTraversableScene"} if scene_model != "empty" else {"type": "Scene"}
    if scene_model != "empty":
        scene_cfg["scene_model"] = scene_model

    # 2) Robot 선택
    robot_name = "Fetch"
    if not quickstart:
        robot_name = choose_from_options(
            options=list(sorted(REGISTERED_ROBOTS.keys())),
            name="robot",
            random_selection=random_selection,
        )

    # 3) Robot config
    robot0_cfg = {
        "type": robot_name,
        "obs_modalities": ["rgb", "depth", "depth_linear", "seg_semantic", "seg_instance", "seg_instance_id"],
        "action_type": "continuous",
        "action_normalize": True,
        "sensor_config": {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": 512,
                    "image_width": 512
                }
            }
        },
    }

    # 4) Environment 구성
    cfg = {"scene": scene_cfg, "robots": [robot0_cfg]}
    env = og.Environment(configs=cfg)

    # 5) Controller 설정
    robot = env.robots[0]
    if quickstart:
        control_mode = "teleop" if not gm.HEADLESS else "random"
        controller_choices = {
            "base": "DifferentialDriveController",
            "arm_0": "InverseKinematicsController",
            "gripper_0": "MultiFingerGripperController",
            "camera": "JointController",
        }
    else:
        controller_choices = choose_controllers(robot, random_selection=random_selection)
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")
        if gm.HEADLESS:
            control_mode = "random"

    robot.reload_controllers({comp: {"name": name} for comp, name in controller_choices.items()})
    #env.scene.update_initial_state()
    env.scene.update_initial_file()

    # ZMQ streaming setup (initialized via global flags parsed in __main__)
    zmq_sock = None
    if getattr(main, "_stream_zmq", False):
        if zmq is None:
            raise RuntimeError("pyzmq is not installed but --stream_zmq was requested")
        zmq_ctx = zmq.Context.instance()
        zmq_sock = zmq_ctx.socket(zmq.PUB)
        zmq_sock.bind(getattr(main, "_zmq_endpoint", "tcp://127.0.0.1:5557"))

    # 6) 카메라 위치 초기화 (viewer 사용 시에만)
    if not gm.HEADLESS:
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([1.46949, -3.97358, 2.21529]),
            orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
        )

    # 7) Reset & Teleop 컨트롤러 준비
    env.reset()
    robot.reset()
    action_generator = KeyboardRobotController(robot=robot)
    if not gm.HEADLESS:
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.R,
            description="Reset the robot",
            callback_fn=lambda: env.reset(),
        )

    if control_mode == "teleop" and not gm.HEADLESS:
        action_generator.print_keyboard_teleop_info()
        print("Running demo. Press ESC or Q to quit")

    # 8) Rerun 뷰어 초기화 (한 번만)
    rr.init("og_quickstart", spawn=True)
    if control_mode == "teleop" and not gm.HEADLESS:
        print("[Info] Teleop mode: Rerun viewer will display images; keep simulator window focused for keyboard input.")
    
    # 8.5) 저장 디렉토리 (옵션일 때만 생성)
    save_dir = getattr(main, "_save_dir", "saved_frames")
    if getattr(main, "_save_frames", False):
        import os
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving frames to: {save_dir}")

    # Point cloud accumulation buffers
    if getattr(main, "_pc_accumulate", False):
        setattr(main, "_pc_accum_points", None)
        setattr(main, "_pc_accum_colors", None)
        setattr(main, "_pc_accum_sem_points", None)
        setattr(main, "_pc_accum_sem_colors", None)
        setattr(main, "_pc_accum_inst_points", None)
        setattr(main, "_pc_accum_inst_colors", None)

    # 9) 메인 루프
    max_steps = -1 if not short_exec else 100
    if gm.HEADLESS and max_steps == -1:
        max_steps = 100
    step = 0
    while step != max_steps:
        # a) 액션 결정
        action = (
            action_generator.get_random_action()
            if control_mode == "random"
            else action_generator.get_teleop_action()
        )
        # Scale action to increase speed and clip to valid range
        gain = float(getattr(main, "_action_gain", 1.0))
        if gain != 1.0:
            try:
                action = (action * gain).clamp(-1.0, 1.0)
            except Exception:
                action = np.clip(np.array(action), -1.0, 1.0)
        env.step(action=action)

        # b) RGB-D & Seg 관측값 가져오기
        data, info = env.get_obs()
        robot_key = next(iter(data.keys()))
        sensor_dict = data[robot_key]
        
        # Get camera data
        cam_key = next(k for k in sensor_dict.keys() if "Camera" in k)
        cam_dict = sensor_dict[cam_key]
        
        # Get camera parameters from the vision sensor
        camera_sensor = None
        for sensor in robot.sensors.values():
            if hasattr(sensor, 'camera_parameters'):
                camera_sensor = sensor
                break
        
        # Pre-compute world-from-camera pose from cameraViewTransform if available (consistent with this frame)
        R_wc_from_view = None
        t_wc_from_view = None
        if camera_sensor and hasattr(camera_sensor, 'camera_parameters'):
            try:
                params = camera_sensor.camera_parameters
                if isinstance(params, dict) and 'cameraViewTransform' in params:
                    vt = np.array(params['cameraViewTransform'], dtype=float).reshape(4, 4)
                    # Treat printed matrix as column-major: translation in last row
                    R_print = vt[:3, :3]
                    t_row = vt[3, :3]
                    # Interpret R_print as world_from_camera rotation (see empirical print format)
                    R_wc_from_view = R_print
                    t_wc_from_view = (-R_wc_from_view @ t_row.reshape(3,)).astype(float)
            except Exception:
                R_wc_from_view = None
                t_wc_from_view = None

        if camera_sensor:
            print(f"\n=== Camera Parameters (Frame {step}) ===")
            camera_params = camera_sensor.camera_parameters
            print(f"Image resolution: {camera_sensor.image_width} x {camera_sensor.image_height}")
            print(f"Focal length: {camera_sensor.focal_length} mm")
            print(f"Horizontal aperture: {camera_sensor.horizontal_aperture} mm")
            print(f"Clipping range: {camera_sensor.clipping_range} meters")
            print(f"Intrinsic matrix:\n{camera_sensor.intrinsic_matrix}")
            
            # Print additional camera parameters if available
            if 'cameraProjection' in camera_params:
                print(f"Camera projection matrix:\n{camera_params['cameraProjection']}")
            if 'cameraViewTransform' in camera_params:
                print(f"Camera view transform:\n{camera_params['cameraViewTransform']}")
            if 'metersPerSceneUnit' in camera_params:
                print(f"Meters per scene unit: {camera_params['metersPerSceneUnit']}")
            print("=" * 50)
        
        # Get LiDAR data if available (for mobile robots)
        lidar_key = next((k for k in sensor_dict.keys() if "Lidar" in k), None)
        if lidar_key:
            lidar_dict = sensor_dict[lidar_key]
            lidar_scan = lidar_dict.get("scan")
            occupancy_grid = lidar_dict.get("occupancy_grid")
            print(f"LiDAR scan shape: {lidar_scan.shape if lidar_scan is not None else 'None'}")
            print(f"Occupancy grid shape: {occupancy_grid.shape if occupancy_grid is not None else 'None'}")
        
        # c) Proprioceptive 데이터 가져오기
        print(f"Available sensor keys: {list(sensor_dict.keys())}")
        
        # Get proprioceptive data directly from robot object
        proprio = robot.get_proprioception()
        print(f"Proprioceptive data type: {type(proprio)}")
        print(f"Proprioceptive data: {proprio}")
        
        # Convert tuple to numpy array if needed
        if isinstance(proprio, tuple):
            # The tuple contains (tensor, dict) - we want the tensor
            proprio_tensor = proprio[0]  # Get the tensor part
            proprio_array = proprio_tensor.cpu().numpy()  # Convert tensor to numpy
            print(f"Proprioceptive data shape: {proprio_array.shape}")
        else:
            proprio_array = proprio
            print(f"Proprioceptive data shape: {proprio_array.shape}")
        
        # Alternative: Get specific proprioceptive components
        # Check if robot has end-effector (manipulation robots only)
        if hasattr(robot, 'get_eef_position'):
            # For manipulation robots
            joint_positions = robot.get_joint_positions()
            joint_velocities = robot.get_joint_velocities()
            eef_position = robot.get_eef_position()
            eef_orientation = robot.get_eef_orientation()
            print(f"Joint positions: {joint_positions}")
            print(f"Joint velocities: {joint_velocities}")
            print(f"End-effector position: {eef_position}")
            print(f"End-effector orientation: {eef_orientation}")
        else:
            # For mobile robots like Turtlebot
            robot_position, robot_orientation = robot.get_position_orientation()
            print(f"Robot position: {robot_position}")
            print(f"Robot orientation: {robot_orientation}")
            
            # Also get joint info for mobile robots (wheels, etc.)
            joint_positions = robot.get_joint_positions()
            joint_velocities = robot.get_joint_velocities()
            print(f"Wheel joint positions: {joint_positions}")
            print(f"Wheel joint velocities: {joint_velocities}")

        # c) RGB-D 시각화
        rgb = np.array(cam_dict["rgb"])[..., :3].astype(np.uint8)
        
        # Check available depth modalities
        print(f"Available camera keys: {list(cam_dict.keys())}")
        
        # Use depth_linear if available, otherwise use depth
        if "depth_linear" in cam_dict:
            depth_linear = np.array(cam_dict["depth_linear"])
            depth_for_viz = depth_linear
            print("Using depth_linear for visualization")
        else:
            depth = np.array(cam_dict["depth"])
            depth_for_viz = depth
            print("Using depth (non-linear) for visualization")
        
        # Depth 정규화하여 컬러맵
        depth_clip = 5.0  # 최대 시각화 거리 (m)
        depth_clipped = np.clip(depth_for_viz, 0.0, depth_clip)
        depth_scaled = (depth_clipped / depth_clip * 255).astype(np.uint8)
        depth_col = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_PLASMA)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # d) Semantic Segmentation (GPU → CPU)
        sem = cam_dict["seg_semantic"].cpu().numpy().astype(np.int32)
        # 클래스 수에 맞춰 정규화
        sem_norm = (sem.astype(np.float32) / sem.max() * 255).astype(np.uint8)
        sem_col = cv2.applyColorMap(sem_norm, cv2.COLORMAP_JET)

        # e) Instance Segmentation (GPU → CPU)
        inst = cam_dict["seg_instance"].cpu().numpy().astype(np.uint8)
        inst_col = cv2.applyColorMap((inst * 50).clip(0,255), cv2.COLORMAP_HSV)

        # f) Instance ID (GPU → CPU)
        inst_id = cam_dict["seg_instance_id"].cpu().numpy().astype(np.int32)
        id_norm = (inst_id.astype(np.float32) / max(inst_id.max(),1) * 255).astype(np.uint8)
        id_col = cv2.applyColorMap(id_norm, cv2.COLORMAP_HSV)

        # g) 모든 이미지를 하나의 창에 결합 (고해상도)
        # 이미지 크기 조정 (더 큰 해상도)
        target_size = (512, 512)  # 각 이미지를 512x512로 확대
        
        # 각 이미지를 target_size로 리사이즈
        rgb_resized = cv2.resize(rgb_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
        depth_resized = cv2.resize(depth_col, target_size, interpolation=cv2.INTER_LANCZOS4)
        sem_resized = cv2.resize(sem_col, target_size, interpolation=cv2.INTER_LANCZOS4)
        inst_resized = cv2.resize(inst_col, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 첫 번째 행: RGB | Depth
        top_row = np.hstack((rgb_resized, depth_resized))
        # 두 번째 행: Semantic | Instance
        bottom_row = np.hstack((sem_resized, inst_resized))
        # 전체 결합
        combined_visualization = np.vstack((top_row, bottom_row))
        
        # h) 로봇 포즈 정보를 이미지에 오버레이
        if not hasattr(robot, 'get_eef_position'):
            # Mobile robot pose visualization
            robot_pos, robot_ori = robot.get_position_orientation()
            
            # Convert tensors to numpy arrays
            pos_np = robot_pos.cpu().numpy() if hasattr(robot_pos, 'cpu') else np.array(robot_pos)
            ori_np = robot_ori.cpu().numpy() if hasattr(robot_ori, 'cpu') else np.array(robot_ori)
            
            # Create text overlay
            overlay = np.zeros((150, 1024, 3), dtype=np.uint8)
            
            # Position text
            pos_text = f"Position: ({pos_np[0]:.3f}, {pos_np[1]:.3f}, {pos_np[2]:.3f})"
            cv2.putText(overlay, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Orientation text (quaternion)
            ori_text = f"Orientation: ({ori_np[0]:.3f}, {ori_np[1]:.3f}, {ori_np[2]:.3f}, {ori_np[3]:.3f})"
            cv2.putText(overlay, ori_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert quaternion to euler angles for easier reading
            import math
            # Simple quaternion to euler conversion (yaw only for 2D)
            yaw = math.atan2(2 * (ori_np[3] * ori_np[2] + ori_np[0] * ori_np[1]), 
                            1 - 2 * (ori_np[1] * ori_np[1] + ori_np[2] * ori_np[2]))
            yaw_deg = math.degrees(yaw)
            yaw_text = f"Yaw: {yaw_deg:.1f}°"
            cv2.putText(overlay, yaw_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add timestamp
            timestamp = f"Step: {step}"
            cv2.putText(overlay, timestamp, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Combine overlay with main visualization
            combined_with_overlay = np.vstack((combined_visualization, overlay))
        else:
            combined_with_overlay = combined_visualization
        
        # 최종 이미지 크기: 1024x1024 (or 1024x1174 with overlay)
        # Rerun 로깅 (RGB, Depth, Semantic, Instance, Combined)
        depth_col_rgb = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)
        sem_col_rgb = cv2.cvtColor(sem_col, cv2.COLOR_BGR2RGB)
        inst_col_rgb = cv2.cvtColor(inst_col, cv2.COLOR_BGR2RGB)
        combined_rgb = cv2.cvtColor(combined_with_overlay, cv2.COLOR_BGR2RGB)
        rr.log("images/rgb", rr.Image(rgb))
        rr.log("images/depth_viz", rr.Image(depth_col_rgb))
        rr.log("images/semantic", rr.Image(sem_col_rgb))
        rr.log("images/instance", rr.Image(inst_col_rgb))
        rr.log("images/combined", rr.Image(combined_rgb))

        # Point cloud 생성 (depth + pose)
        try:
            # Prefer metric depth
            depth_m = depth_linear if "depth_linear" in cam_dict else depth_for_viz
            if camera_sensor is not None and hasattr(camera_sensor, "intrinsic_matrix"):
                K = np.array(camera_sensor.intrinsic_matrix, dtype=float).reshape(3, 3)
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cx, cy = float(K[0, 2]), float(K[1, 2])

                intrinsics = env.simulator.viewer_camera.get_intrinsics()
                fx, fy = intrinsics["fx"], intrinsics["fy"]
                cx, cy = intrinsics["cx"], intrinsics["cy"]

                # Pose: prefer camera sensor pose, else robot pose (handles euler or quaternion)
                if R_wc_from_view is not None and t_wc_from_view is not None:
                    R_wc = R_wc_from_view
                    t_wc = t_wc_from_view
                else:
                    if camera_sensor and hasattr(camera_sensor, 'get_position_orientation'):
                        p_t, q_t = camera_sensor.get_position_orientation()
                    else:
                        p_t, q_t = robot.get_position_orientation()
                    t_wc = np.array([float(p_t[0]), float(p_t[1]), float(p_t[2])], dtype=float)
                    q_arr = np.array(q_t, dtype=float).reshape(-1)
                    if q_arr.shape[0] == 3:
                        R_wc = Rotation.from_euler('xyz', q_arr, degrees=False).as_matrix()
                    else:
                        # assume xyzw
                        R_wc = Rotation.from_quat([q_arr[0], q_arr[1], q_arr[2], q_arr[3]]).as_matrix()

                # Downsample for performance
                stride = 4
                z = depth_m[::stride, ::stride].astype(np.float32)
                h_s, w_s = z.shape
                u = (np.arange(0, w_s) * stride + 0.5).astype(np.float32)
                v = (np.arange(0, h_s) * stride + 0.5).astype(np.float32)
                uu, vv = np.meshgrid(u, v)

                Z = z.flatten()
                min_d = float(getattr(main, "_pc_min_depth", 0.5))
                max_d = float(getattr(main, "_pc_max_depth", 10.0))
                valid = np.isfinite(Z) & (Z > min_d) & (Z < max_d)
                if np.any(valid):
                    print("----------------------------------------------------------------")
                    print(K)
                    print("----------------------------------------------------------------")
                    X = ((uu.flatten() - cx) / fx) * Z
                    Y = ((vv.flatten() - cy) / fy) * Z
                    # Current unprojection yields camera RDF (x right, y down, z forward)
                    # Convert to RUB (x right, y up, z back): flip Y and Z
                    Xv = X
                    Yv = -Y
                    Zv = -Z
                    Pc_rub = np.stack([Xv, Yv, Zv], axis=1)[valid]
                    Pw = (R_wc @ Pc_rub.T).T + t_wc

                    # Colors from RGB
                    rgb_ds = rgb[::stride, ::stride, :]
                    colors = rgb_ds.reshape(-1, 3)[valid]

                    rr.log("pc/world", rr.Points3D(positions=Pw.astype(np.float32), colors=colors.astype(np.uint8)))

                    # Accumulate with voxel dedup and max cap
                    if getattr(main, "_pc_accumulate", False):
                        pc_all = Pw.astype(np.float32)
                        col_all = colors.astype(np.uint8)
                        if getattr(main, "_pc_accum_points", None) is not None:
                            pc_all = np.vstack([getattr(main, "_pc_accum_points"), pc_all])
                            col_all = np.vstack([getattr(main, "_pc_accum_colors"), col_all])

                        voxel = float(getattr(main, "_pc_voxel_size", 0.03))
                        keys = np.floor(pc_all / max(voxel, 1e-6)).astype(np.int32)
                        # Unique by voxel cell
                        _, uniq_idx = np.unique(keys, axis=0, return_index=True)
                        pc_ds = pc_all[uniq_idx]
                        col_ds = col_all[uniq_idx]

                        setattr(main, "_pc_accum_points", pc_ds)
                        setattr(main, "_pc_accum_colors", col_ds)
                        rr.log("pc/world_accum", rr.Points3D(positions=pc_ds, colors=col_ds))

                        # Semantic-colored accumulated point cloud
                        sem_rgb_ds = sem_col_rgb[::stride, ::stride, :]
                        sem_cols = sem_rgb_ds.reshape(-1, 3)[valid].astype(np.uint8)
                        pc_sem_all = Pw.astype(np.float32)
                        col_sem_all = sem_cols
                        if getattr(main, "_pc_accum_sem_points", None) is not None:
                            pc_sem_all = np.vstack([getattr(main, "_pc_accum_sem_points"), pc_sem_all])
                            col_sem_all = np.vstack([getattr(main, "_pc_accum_sem_colors"), col_sem_all])

                        keys_sem = np.floor(pc_sem_all / max(voxel, 1e-6)).astype(np.int32)
                        _, uniq_sem_idx = np.unique(keys_sem, axis=0, return_index=True)
                        pc_sem_ds = pc_sem_all[uniq_sem_idx]
                        col_sem_ds = col_sem_all[uniq_sem_idx]

                        setattr(main, "_pc_accum_sem_points", pc_sem_ds)
                        setattr(main, "_pc_accum_sem_colors", col_sem_ds)
                        rr.log("pc/world_accum_sem", rr.Points3D(positions=pc_sem_ds, colors=col_sem_ds))

                        # Instance-colored accumulated point cloud (stable across frames)
                        inst_ids_ds = inst_id[::stride, ::stride]
                        inst_ids_flat = inst_ids_ds.reshape(-1)[valid].astype(np.uint32)
                        # Stable hash -> RGB
                        h = (inst_ids_flat * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
                        r = (h & np.uint32(0xFF)).astype(np.uint8)
                        g = ((h >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
                        b = ((h >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
                        inst_cols = np.stack([r, g, b], axis=1)
                        # Ensure not too dark
                        inst_cols = np.maximum(inst_cols, 32).astype(np.uint8)
                        pc_inst_all = Pw.astype(np.float32)
                        col_inst_all = inst_cols
                        if getattr(main, "_pc_accum_inst_points", None) is not None:
                            pc_inst_all = np.vstack([getattr(main, "_pc_accum_inst_points"), pc_inst_all])
                            col_inst_all = np.vstack([getattr(main, "_pc_accum_inst_colors"), col_inst_all])

                        keys_inst = np.floor(pc_inst_all / max(voxel, 1e-6)).astype(np.int32)
                        _, uniq_inst_idx = np.unique(keys_inst, axis=0, return_index=True)
                        pc_inst_ds = pc_inst_all[uniq_inst_idx]
                        col_inst_ds = col_inst_all[uniq_inst_idx]

                        setattr(main, "_pc_accum_inst_points", pc_inst_ds)
                        setattr(main, "_pc_accum_inst_colors", col_inst_ds)
                        rr.log("pc/world_accum_inst", rr.Points3D(positions=pc_inst_ds, colors=col_inst_ds))
        except Exception:
            pass

        # 카메라 Intrinsics / Pose 로깅 (cam/current)
        # Use the same world-from-camera pose as for point cloud when available
        if R_wc_from_view is not None and t_wc_from_view is not None:
            translation = t_wc_from_view.tolist()
            from scipy.spatial.transform import Rotation as R
            # print(dir(env))
            # R_wc_from_view = env.simulator.viewer_camera.get_world_rotation_matrix()
            try:
                quat_xyzw = R.from_matrix(R_wc_from_view).as_quat().tolist()
            except ValueError:
                quat_xyzw = [0, 0, 0, 1]
        else:
            if camera_sensor and hasattr(camera_sensor, 'get_position_orientation'):
                p_t, q_t = camera_sensor.get_position_orientation()
            elif not hasattr(robot, 'get_eef_position'):
                p_t, q_t = robot.get_position_orientation()
            else:
                p_t, q_t = robot.get_eef_position(), robot.get_eef_orientation()
            translation = [float(p_t[0]), float(p_t[1]), float(p_t[2])]
            q = np.array([float(q_t[0]), float(q_t[1]), float(q_t[2]), float(q_t[3])], dtype=float)
            q /= (np.linalg.norm(q) or 1.0)
            quat_xyzw = q.tolist()
        rr.log("cam/current", rr.Transform3D(
            translation=translation,
            rotation=rr.Quaternion(xyzw=quat_xyzw)
        ))

        if camera_sensor is not None and hasattr(camera_sensor, "intrinsic_matrix"):
            try:
                K = np.array(camera_sensor.intrinsic_matrix, dtype=float).reshape(3, 3)
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                w = int(getattr(camera_sensor, 'image_width', 0))
                h = int(getattr(camera_sensor, 'image_height', 0))
                rr.log("cam/current", rr.Pinhole(
                    resolution=[w, h],
                    image_from_camera=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    # Match unprojection convention (RDF). Rerun will render frustum accordingly.
                    camera_xyz=rr.ViewCoordinates.RDF,
                ))
            except Exception:
                pass

        # 3D Bounding Boxes → Rerun (as line strips)
        try:
            # Find bboxes in cam_dict or info
            bboxes = None
            for key in [
                'bboxes_3d', 'bbox_3d', '3d_bounding_boxes', 'bbox3d', 'bounding_boxes_3d'
            ]:
                if key in cam_dict:
                    bboxes = cam_dict[key]
                    break
            if bboxes is None and isinstance(info, dict):
                for key in [
                    'bboxes_3d', 'bbox_3d', '3d_bounding_boxes', 'bbox3d', 'bounding_boxes_3d'
                ]:
                    if key in info:
                        bboxes = info[key]
                        break

            def corners_from_minmax(xmin, ymin, zmin, xmax, ymax, zmax):
                return np.array([
                    [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],  # bottom
                    [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],  # top
                ], dtype=float)

            def box_strips_from_corners_world(Cw):
                # Create 4 rectangle loops and 4 vertical edges
                strips = []
                # bottom loop: 0-1-2-3-0
                strips.append(Cw[[0, 1, 2, 3, 0]].astype(np.float32))
                # top loop: 4-5-6-7-4
                strips.append(Cw[[4, 5, 6, 7, 4]].astype(np.float32))
                # verticals: 0-4, 1-5, 2-6, 3-7
                strips.append(Cw[[0, 4]].astype(np.float32))
                strips.append(Cw[[1, 5]].astype(np.float32))
                strips.append(Cw[[2, 6]].astype(np.float32))
                strips.append(Cw[[3, 7]].astype(np.float32))
                return strips

            def color_from_semantic_id(sid):
                h = (np.uint32(sid) * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
                r = int(h & np.uint32(0xFF))
                g = int((h >> np.uint32(8)) & np.uint32(0xFF))
                b = int((h >> np.uint32(16)) & np.uint32(0xFF))
                return [max(r, 32), max(g, 32), max(b, 32)]

            all_strips = []
            strip_colors = []
            if bboxes is not None:
                for entry in bboxes:
                    try:
                        if isinstance(entry, dict):
                            sid = int(entry.get('semanticID', 0))
                            xmin = float(entry.get('x_min', 0.0))
                            ymin = float(entry.get('y_min', 0.0))
                            zmin = float(entry.get('z_min', 0.0))
                            xmax = float(entry.get('x_max', 0.0))
                            ymax = float(entry.get('y_max', 0.0))
                            zmax = float(entry.get('z_max', 0.0))
                            T = np.array(entry.get('transform', np.eye(4)), dtype=float).reshape(4, 4)
                        else:
                            sid = int(entry[0])
                            xmin, ymin, zmin, xmax, ymax, zmax = [float(v) for v in entry[1:7]]
                            T = np.array(entry[7], dtype=float).reshape(4, 4)

                        Cl = corners_from_minmax(xmin, ymin, zmin, xmax, ymax, zmax)
                        Cl_h = np.concatenate([Cl, np.ones((8, 1), dtype=float)], axis=1)
                        Cw_h = (T @ Cl_h.T).T
                        Cw = Cw_h[:, :3] / np.maximum(Cw_h[:, 3:4], 1e-6)

                        strips = box_strips_from_corners_world(Cw)
                        color = color_from_semantic_id(sid)
                        all_strips.extend(strips)
                        strip_colors.extend([color] * len(strips))
                    except Exception:
                        continue

            # Fallback: draw AABBs for scene objects if no boxes found
            if not all_strips and hasattr(env, 'scene') and hasattr(env.scene, 'objects'):
                try:
                    for obj in env.scene.objects:
                        try:
                            ctr = obj.aabb_center
                            ext = obj.aabb_extent
                            # Convert tensors to numpy
                            if hasattr(ctr, 'cpu'):
                                ctr = ctr.cpu().numpy()
                            if hasattr(ext, 'cpu'):
                                ext = ext.cpu().numpy()
                            half = ext * 0.5
                            xmin, ymin, zmin = (ctr - half).tolist()
                            xmax, ymax, zmax = (ctr + half).tolist()
                            Cl = corners_from_minmax(xmin, ymin, zmin, xmax, ymax, zmax)
                            strips = box_strips_from_corners_world(Cl)
                            color = color_from_semantic_id(hash(obj.name) & 0xFFFFFFFF)
                            all_strips.extend(strips)
                            strip_colors.extend([color] * len(strips))
                        except Exception:
                            continue
                except Exception:
                    pass

            if all_strips:
                rr.log("bboxes3d/world", rr.LineStrips3D(all_strips, radii=0.02, colors=strip_colors))
        except Exception:
            pass

        # i) 모든 이미지와 데이터 저장 (옵션일 때만)
        if getattr(main, "_save_frames", False):
            import os
            frame_filename = f"frame_{step:06d}"
            
            # Save RGB image
            cv2.imwrite(os.path.join(save_dir, f"{frame_filename}_rgb.png"), rgb_bgr)
            
            # Save depth visualization
            cv2.imwrite(os.path.join(save_dir, f"{frame_filename}_depth_viz.png"), depth_col)
            
            # Save raw depth data (numpy array)
            if "depth_linear" in cam_dict:
                np.save(os.path.join(save_dir, f"{frame_filename}_depth_raw.npy"), depth_linear)
            else:
                np.save(os.path.join(save_dir, f"{frame_filename}_depth_raw.npy"), depth)
            
            # Save segmentation images
            cv2.imwrite(os.path.join(save_dir, f"{frame_filename}_semantic.png"), sem_col)
            cv2.imwrite(os.path.join(save_dir, f"{frame_filename}_instance.png"), inst_col)
            
            # Save robot pose data
            if not hasattr(robot, 'get_eef_position'):
                robot_pos, robot_ori = robot.get_position_orientation()
                pose_data = {
                    'position': robot_pos.cpu().numpy() if hasattr(robot_pos, 'cpu') else np.array(robot_pos),
                    'orientation': robot_ori.cpu().numpy() if hasattr(robot_ori, 'cpu') else np.array(robot_ori),
                    'step': step,
                    'timestamp': step  # You can add actual timestamp if needed
                }
                np.save(os.path.join(save_dir, f"{frame_filename}_pose.npy"), pose_data)
            
            # Save combined visualization
            cv2.imwrite(os.path.join(save_dir, f"{frame_filename}_combined.png"), combined_with_overlay)
        
        print(f"Saved frame {step} with all data")

        # g) 종료 키 처리는 Rerun 사용 시 생략 (Ctrl-C 또는 ESC 키로 종료)

        step += 1

    # 10) 종료 정리
    og.clear()
    if zmq_sock is not None:
        try:
            zmq_sock.close(0)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teleoperate a robot with segmentation viz.")
    parser.add_argument("--quickstart", action="store_true", help="Use default settings for quick start.")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer using offscreen rendering.")
    parser.add_argument("--short_exec", action="store_true", help="Run a small fixed number of steps.")
    parser.add_argument("--save_frames", action="store_true", help="Save per-frame images and data to disk.")
    parser.add_argument("--save_dir", type=str, default="saved_frames", help="Directory to save frames.")
    parser.add_argument("--pc_min_depth", type=float, default=0.5, help="Min depth (m) for point cloud filtering.")
    parser.add_argument("--pc_max_depth", type=float, default=10.0, help="Max depth (m) for point cloud filtering.")
    parser.add_argument("--pc_accumulate", action="store_true", default=True, help="Accumulate point cloud across frames (default on).")
    parser.add_argument("--pc_voxel_size", type=float, default=0.03, help="Voxel size (m) for deduplication.")
    parser.add_argument("--action_gain", type=float, default=2.0, help="Scale action magnitude (default 2.0).")
    args = parser.parse_args()
    # Stash save flags onto the function to avoid refactoring signature
    setattr(main, "_save_frames", args.save_frames)
    setattr(main, "_save_dir", args.save_dir)
    setattr(main, "_pc_min_depth", args.pc_min_depth)
    setattr(main, "_pc_max_depth", args.pc_max_depth)
    setattr(main, "_pc_accumulate", args.pc_accumulate)
    setattr(main, "_pc_voxel_size", args.pc_voxel_size)
    # Action speed scaling
    setattr(main, "_action_gain", args.action_gain)
    main(quickstart=args.quickstart, short_exec=args.short_exec, headless=args.headless)
