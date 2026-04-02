###########################################################################
# ANYmal Robot RL policy control via keyboard
#
# Shows how to control robot pretrained in IsaacLab with RL.
# The policy is loaded from a file and the robot is controlled via keyboard.
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Use --physx to run with a PhysX-trained policy.
###########################################################################

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import torch
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, State


@dataclass
class RobotConfig:
    """Configuration for a robot including asset paths and policy paths."""

    asset_dir: str
    policy_path: dict[str, str]
    asset_path: str  # USD asset path inside the downloaded asset directory
    yaml_path: str  # Path within the asset directory to the configuration YAML


# Robot configuration for ANYmal C
ROBOT_CONFIG = RobotConfig(
    asset_dir="anybotics_anymal_c",
    policy_path={"mjw": "rl_policies/mjw_anymal.pt", "physx": "rl_policies/physx_anymal.pt"},
    asset_path="usd/anymal_c.usda",
    yaml_path="rl_policies/anymal.yaml",
)


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(
    actions: torch.Tensor,
    state: State,
    joint_pos_initial: torch.Tensor,
    device: str,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Compute observation for robot policy.

    Args:
        actions: Previous actions tensor
        state: Current simulation state
        joint_pos_initial: Initial joint positions
        device: PyTorch device string
        indices: Index mapping for joint reordering
        gravity_vec: Gravity vector in world frame
        command: Command vector

    Returns:
        Observation tensor for policy input
    """
    # Extract state via wp.to_torch() zero-copy (works on both CPU and CUDA)
    joint_q_t = wp.to_torch(state.joint_q)
    joint_qd_t = wp.to_torch(state.joint_qd)

    root_quat_w = joint_q_t[3:7].unsqueeze(0)
    root_lin_vel_w = joint_qd_t[:3].unsqueeze(0)
    root_ang_vel_w = joint_qd_t[3:6].unsqueeze(0)
    joint_pos_current = joint_q_t[7:].unsqueeze(0)
    joint_vel_current = joint_qd_t[6:].unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)

    return obs


def load_policy_and_setup_tensors(robot: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load policy and setup initial tensors for robot control.

    Args:
        robot: AnymalRL instance
        policy_path: Path to the policy file
        num_dofs: Number of degrees of freedom
        joint_pos_slice: Slice for extracting joint positions from state
    """
    device = robot.torch_device
    print("[INFO] Loading policy from:", policy_path)
    robot.policy = torch.jit.load(policy_path, map_location=device)

    # Use wp.to_torch() + clone so joint_pos_initial is a snapshot, not a live view
    joint_q_t = wp.to_torch(robot.state_0.joint_q)
    robot.joint_pos_initial = joint_q_t[joint_pos_slice].clone().unsqueeze(0)
    robot.act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)
    robot.rearranged_act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc

def pd_control(target_pos, parent_class):
    '''Simple PD controller to compute joint torques.

    Uses zero-copy wp.to_torch() to avoid per-step warp→numpy→torch allocations.
    '''
    q_all = wp.to_torch(parent_class.state_0.joint_q)
    qd_all = wp.to_torch(parent_class.state_0.joint_qd)
    q_current = q_all[7:]    # joint positions (skip 7 root DOFs: 3 pos + 4 quat)
    qd_current = qd_all[6:]  # joint velocities (skip 6 root DOFs: 3 lin + 3 ang)
    return parent_class.kp * (target_pos - q_current) - parent_class.kd * qd_current


class AnymalRL:
    def __init__(
        self,
        viewer,
        robot_config: RobotConfig,
        config,
        asset_directory: str,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
    ):
        # Setup simulation parameters first
        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        # Group related attributes by prefix
        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Wall-clock rate limiting: policy runs at fps/decimation Hz
        self.policy_dt = self.frame_dt * self.decimation  # 0.02s per step
        self._wall_time_last = time.perf_counter()

        # Save a reference to the viewer
        self.viewer = viewer

        # Store configuration
        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config

        # Device setup
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        # Build the model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        builder.add_usd(
            newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        builder.approximate_meshes("convex_hull")

        builder.add_ground_plane()
        # builder's gravity isn't a vec3. use model.set_gravity()
        # builder.gravity = wp.vec3(0.0, 0.0, -9.81)

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config["mjw_joint_pos"]

        for i in range(len(config["mjw_joint_stiffness"])):
            # ── Option A (default): POSITION mode — Newton applies PD internally ──
            # builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
            # builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
            # builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)
            # ── Option B: EFFORT mode — manual PD via pd_control() ──
            builder.joint_target_ke[i + 6] = 0.0
            builder.joint_target_kd[i + 6] = 0.0
            builder.joint_target_mode[i + 6] = int(JointTargetMode.EFFORT)
            builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            nconmax=30,
            njmax=100,
        )

        # Initialize state objects
        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        # Set model in viewer
        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        # Ensure FK evaluation (for non-MuJoCo solvers)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Store initial joint state for fast reset
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        # Pre-compute tensors that don't change during simulation
        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._root_zeros = torch.zeros(6, device=self.torch_device, dtype=torch.float32)
        self._reset_key_prev = False

        # PD controller gains (from YAML config)
        self.kp = torch.tensor(config["mjw_joint_stiffness"], device=self.torch_device, dtype=torch.float32)
        self.kd = torch.tensor(config["mjw_joint_damping"], device=self.torch_device, dtype=torch.float32)

        # Initialize policy-related attributes
        # (will be set by load_policy_and_setup_tensors)
        self.policy = None
        self.joint_pos_initial = None
        self.act = None
        self.rearranged_act = None

        self._use_cuda = self.torch_device == "cuda"
        if self._use_cuda:
            print("using CUDA device for policy inference")
        else:
            print("using CPU device for policy inference")

        # Inference timing
        self._inference_times_ms: list[float] = []
        self._torque_log: list[np.ndarray] = []

        # Call capture at the end
        self.capture()

    def capture(self):
        """Put graph capture into it's own method."""
        self.graph = None
        self.use_cuda_graph = False
        if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            torch_tensor = torch.zeros(self.config["num_dofs"] + 6, device=self.torch_device, dtype=torch.float32)
            # ── Option A (default): POSITION mode ──
            # self.control.joint_target_pos = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            # ── Option B: EFFORT mode ──
            self.control.joint_f = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Simulate performs one frame's worth of updates."""
        need_state_copy = self.use_cuda_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            # Swap states - handle CUDA graph case specially
            if need_state_copy and i == self.sim_substeps - 1:
                # Swap states by copying the state arrays for graph capture
                self.state_0.assign(self.state_1)
            else:
                # We can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        print("[INFO] Resetting simulation")
        # Restore initial joint positions and velocities in-place.
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def step(self):
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)
            # Reset when 'P' is pressed (edge-triggered)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        if self._use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t0 = time.perf_counter()

        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
            self.torch_device,
            self.physx_to_mjc_indices,
            self.gravity_vec,
            self.command,
        )
        with torch.no_grad():
            self.act = self.policy(obs)
            self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)

            # ── Option A: Embedded Newton PD control (POSITION mode, default) ──
            # Newton applies PD (ke/kd from builder config) at each physics substep.
            # a = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            # a_with_zeros = torch.cat([self._root_zeros, a.squeeze(0)])
            # a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            # wp.copy(self.control.joint_target_pos, a_wp)

            # ── Option B: Manual PD controller (EFFORT mode) ──
            # To switch: (1) comment out Option A above, (2) uncomment Option B here
            # and in the decimation loop below, (3) switch builder config to EFFORT,
            # (4) switch capture() to use control.joint_f.
            # IMPORTANT: PD torques must be recomputed inside the decimation loop
            # (at physics rate) — NOT once per policy step — otherwise the controller
            # is unstable and the robot diverges.
            target_pos = (self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act).squeeze(0)

        if self._use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            self._inference_times_ms.append(start_event.elapsed_time(end_event))
        else:
            self._inference_times_ms.append((time.perf_counter() - t0) * 1000.0)

        # Record PD events per decimation step (no mid-loop sync)
        if self._use_cuda:
            pd_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        else:
            pd_cpu_ms = 0.0

        for _ in range(self.decimation):
            # ── Option B (cont.): recompute PD torques at physics rate ──
            if self._use_cuda:
                pd_s = torch.cuda.Event(enable_timing=True)
                pd_e = torch.cuda.Event(enable_timing=True)
                pd_s.record()
            else:
                pd_t0 = time.perf_counter()

            with torch.no_grad():
                torques = pd_control(target_pos, self)
                torques_with_zeros = torch.cat([self._root_zeros, torques])
                wp.copy(self.control.joint_f, wp.from_torch(torques_with_zeros, dtype=wp.float32, requires_grad=False))
                # self._torque_log.append(torques.detach().cpu().numpy())  # expensive: forces GPU sync

            if self._use_cuda:
                pd_e.record()
                pd_events.append((pd_s, pd_e))
            else:
                pd_cpu_ms += (time.perf_counter() - pd_t0) * 1000.0

            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        # Accumulate PD time into inference measurement (single sync)
        if self._use_cuda:
            torch.cuda.synchronize()
            self._inference_times_ms[-1] += sum(s.elapsed_time(e) for s, e in pd_events)
        else:
            self._inference_times_ms[-1] += pd_cpu_ms

        self.sim_time += self.policy_dt

        # Rate-limit to real-time: sleep if we're ahead of wall clock
        now = time.perf_counter()
        elapsed = now - self._wall_time_last
        sleep_time = self.policy_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._wall_time_last = time.perf_counter()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--physx", action="store_true", help="Run physX policy instead of MJWarp.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    robot_config = ROBOT_CONFIG
    print("[INFO] Selected robot: anymal")

    # Download assets from newton-assets repository
    asset_directory = str(newton.utils.download_asset(robot_config.asset_dir))
    print(f"[INFO] Asset directory: {asset_directory}")

    # Load robot configuration from YAML file in the downloaded assets
    yaml_file_path = f"{asset_directory}/{robot_config.yaml_path}"
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    if args.physx:
        if "physx" not in robot_config.policy_path or "physx_joint_names" not in config:
            raise ValueError("PhysX policy/joint mapping not available for the anymal robot.")
        policy_path = f"{asset_directory}/{robot_config.policy_path['physx']}"
        mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config["mjw_joint_names"], config["physx_joint_names"])
    else:
        policy_path = f"{asset_directory}/{robot_config.policy_path['mjw']}"

    anymal = AnymalRL(viewer, robot_config, config, asset_directory, mjc_to_physx, physx_to_mjc)
    load_policy_and_setup_tensors(anymal, policy_path, config["num_dofs"], slice(7, None))

    newton.examples.run(anymal, args)

    # Report and save inference timing and torque data
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_data")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    if anymal._inference_times_ms:
        t = np.array(anymal._inference_times_ms)
        print(f"[INFO] Control pipeline: {len(t)} calls, "
              f"mean = {t.mean():.4f} ms, std = {t.std():.4f} ms")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{script_name}_{timestamp}.npy")
        np.save(log_path, t)
        print(f"[INFO] Inference times saved to: {log_path}")
    if anymal._torque_log:
        torque_array = np.array(anymal._torque_log)
        os.makedirs(log_dir, exist_ok=True)
        torque_path = os.path.join(log_dir, f"{script_name}_torques_{timestamp}.npy")
        np.save(torque_path, torque_array)
        print(f"[INFO] Torque log saved to: {torque_path} (shape: {torque_array.shape})")