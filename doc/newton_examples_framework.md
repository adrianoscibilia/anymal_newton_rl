# Newton Examples Framework — Complete Reference

This document describes the `newton.examples` application framework: the loop that
drives simulation, the class interface your application must implement, the CLI
arguments, the viewer API, and every hook/callback with its trigger conditions.

Source: `newton/examples/__init__.py`  
Viewer base class: `newton/_src/viewer/viewer.py`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The `run()` Loop — Step by Step](#2-the-run-loop--step-by-step)
3. [Application Class Interface](#3-application-class-interface)
   - [Required Methods](#31-required-methods)
   - [Optional Methods and Hooks](#32-optional-methods-and-hooks)
   - [Expected Attributes (for `--test` mode)](#33-expected-attributes-for---test-mode)
4. [`create_parser()` — CLI Arguments](#4-create_parser--cli-arguments)
5. [`init()` — Bootstrap](#5-init--bootstrap)
6. [Viewer API Reference](#6-viewer-api-reference)
   - [Viewer Types](#61-viewer-types)
   - [Core Viewer Methods](#62-core-viewer-methods)
   - [Logging / Drawing Methods](#63-logging--drawing-methods)
   - [Display Flags](#64-display-flags)
   - [Built-in Key Bindings (ViewerGL)](#65-built-in-key-bindings-viewergl)
7. [Test Utilities](#7-test-utilities)
8. [Other Utility Functions](#8-other-utility-functions)
9. [Typical Application Skeleton](#9-typical-application-skeleton)
10. [FAQ / Gotchas](#10-faq--gotchas)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  if __name__ == "__main__":                             │
│                                                         │
│    parser = newton.examples.create_parser()             │
│    parser.add_argument(...)          # your own args    │
│    viewer, args = newton.examples.init(parser)          │
│                                                         │
│    app = MyApp(viewer, ...)          # your class       │
│                                                         │
│    newton.examples.run(app, args)    # enters loop      │
│                                                         │
│    # <-- code here runs AFTER viewer is closed          │
└─────────────────────────────────────────────────────────┘
```

`newton.examples.run()` owns the main loop. It calls methods on **your class
instance** (the `example` parameter) at well-defined points. You implement only the
methods you need. After the viewer window is closed (ESC or window close button),
`run()` returns and execution continues in your `__main__` block.

---

## 2. The `run()` Loop — Step by Step

Below is the complete logic of `run(example, args)` annotated with the conditions
that trigger each branch.

```python
def run(example, args):

    # ── 1. GUI registration ──────────────────────────────
    # If your class has a gui() method AND the viewer supports UI callbacks,
    # register it as a side-panel callback.
    if hasattr(example, "gui") and hasattr(example.viewer, "register_ui_callback"):
        example.viewer.register_ui_callback(
            lambda ui: example.gui(ui), position="side"
        )

    # ── 2. Determine test mode ────────────────────────────
    perform_test = args is not None and args.test       # --test flag
    test_post_step = perform_test and hasattr(example, "test_post_step")
    test_final    = perform_test and hasattr(example, "test_final")

    # ── 3. Main simulation loop ───────────────────────────
    while example.viewer.is_running():          # False → window closed / ESC

        if not example.viewer.is_paused():      # Space toggles pause
            example.step()                      # REQUIRED — advance simulation

        if test_post_step:
            example.test_post_step()            # called EVERY frame in test mode
                                                # (even when paused!)

        example.render()                        # REQUIRED — draw current state

    # ── 4. Post-loop: test checks ─────────────────────────
    if perform_test:
        if test_final:
            example.test_final()                # called ONCE after loop exits
        elif not (test_post_step or test_final):
            raise NotImplementedError(
                "Example does not have a test_final or test_post_step method"
            )

    # ── 5. Viewer cleanup ────────────────────────────────
    example.viewer.close()

    # ── 6. Post-loop: NaN checks (test mode only) ────────
    if perform_test:
        for attr in ("state_0", "state_1", "model", "control", "contacts"):
            if hasattr(example, attr):
                nan_members = find_nan_members(getattr(example, attr))
                if nan_members:
                    raise ValueError(f"NaN members found in {attr}: {nan_members}")

    # ── 7. Return ─────────────────────────────────────────
    # Execution returns to caller. Any code after newton.examples.run()
    # in your __main__ block will now execute (cleanup, logging, etc.).
```

### Key observations

| Aspect | Detail |
|--------|--------|
| **Pause** | `step()` is skipped when paused; `render()` is always called |
| **`test_post_step()`** | Called every frame (paused or not) — only in `--test` mode |
| **`test_final()`** | Called once after loop — only in `--test` mode |
| **Viewer close** | `viewer.close()` is always called after the loop |
| **Post-run code** | Any code after `newton.examples.run()` always executes |

---

## 3. Application Class Interface

### 3.1 Required Methods

These two methods **must** be implemented — `run()` calls them unconditionally.

#### `step(self)`

Advance the simulation by one policy/control step. Called once per frame when the
viewer is **not paused**.

Typical responsibilities:
- Read user input (keyboard commands via `self.viewer.is_key_down()`)
- Compute observations and run policy inference
- Apply joint targets / forces
- Step the physics solver (one or more substeps)
- Update `self.sim_time`

#### `render(self)`

Draw the current simulation state. Called **every frame** (even when paused).

Typical implementation:
```python
def render(self):
    self.viewer.begin_frame(self.sim_time)
    self.viewer.log_state(self.state_0)
    self.viewer.log_contacts(self.contacts, self.state_0)
    self.viewer.end_frame()
```

### 3.2 Optional Methods and Hooks

These are detected via `hasattr()` and called only when present and conditions are met.

#### `gui(self, ui)`

**Condition:** class has `gui` attribute AND viewer supports `register_ui_callback`.

Called during ImGui rendering to draw custom UI elements in the side panel.
The `ui` parameter is an ImGui context — use `imgui_bundle` calls to draw widgets.

```python
def gui(self, ui):
    import imgui_bundle.imgui as imgui
    _, self.some_param = imgui.slider_float("Param", self.some_param, 0.0, 1.0)
```

#### `test_post_step(self)`

**Condition:** `--test` flag AND class has `test_post_step`.

Called every frame (regardless of pause state) during test mode. Use for per-step
validation assertions.

#### `test_final(self)`

**Condition:** `--test` flag AND class has `test_final`.

Called once after the main loop exits, before `viewer.close()`. Use for final
validation (e.g., check that the robot hasn't fallen over).

> **Important:** `test_final()` is **never** called in normal (non-test) mode.
> Do not put cleanup/logging code here unless you always run with `--test`.

### 3.3 Expected Attributes (for `--test` mode)

After the loop, `run()` checks these attributes for NaN values (test mode only):

| Attribute | Type |
|-----------|------|
| `state_0` | `newton.State` |
| `state_1` | `newton.State` |
| `model` | `newton.Model` |
| `control` | `newton.Control` |
| `contacts` | `newton.Contacts` |

These are checked via `hasattr()` — they are not required if you don't use `--test`.

### 3.4 Required Attribute

#### `viewer`

Your class **must** store the viewer instance as `self.viewer`. The `run()` function
accesses `example.viewer` for GUI registration.

---

## 4. `create_parser()` — CLI Arguments

`create_parser()` returns an `argparse.ArgumentParser` with these built-in arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--device` | `str` | `None` | Override Warp device (e.g., `"cuda:0"`, `"cpu"`) |
| `--viewer` | `str` | `"gl"` | Viewer backend: `gl`, `usd`, `rerun`, `null`, `viser` |
| `--rerun-address` | `str` | `None` | External Rerun server address |
| `--output-path` | `str` | `"output.usd"` | Output file for USD viewer |
| `--num-frames` | `int` | `100` | Frame count (for `usd`/`null` viewers) |
| `--headless` | `bool` | `False` | Headless mode (GL viewer only) |
| `--test` | `bool` | `False` | Enable test mode (triggers `test_post_step`/`test_final` + NaN checks) |
| `--broad-phase` | `str` | `"explicit"` | Collision broad phase: `nxn`, `sap`, `explicit` |
| `--use-mujoco-contacts` | `bool` | `False` | Use MuJoCo native contact solver |
| `--max-worlds` | `int` | `None` | Limit rendered worlds (performance) |
| `--quiet` | `bool` | `False` | Suppress Warp compilation messages |

You can extend this parser with your own arguments:

```python
parser = newton.examples.create_parser()
parser.add_argument("--my-param", type=float, default=1.0)
viewer, args = newton.examples.init(parser)
# args.my_param is now available
```

---

## 5. `init()` — Bootstrap

```python
viewer, args = newton.examples.init(parser)
```

`init()` performs three operations:

1. **Parses arguments** — calls `parser.parse_args()` (or `parse_known_args()` if
   no parser is provided)
2. **Configures Warp** — sets device (`wp.set_device()`), quiet mode
3. **Creates viewer** — instantiates the appropriate viewer class based on
   `--viewer`:

| `--viewer` value | Viewer class created |
|------------------|---------------------|
| `gl` | `ViewerGL(headless=args.headless)` |
| `usd` | `ViewerUSD(output_path=..., num_frames=...)` |
| `rerun` | `ViewerRerun(address=args.rerun_address)` |
| `null` | `ViewerNull(num_frames=args.num_frames)` |
| `viser` | `ViewerViser()` |

---

## 6. Viewer API Reference

### 6.1 Viewer Types

| Class | Interactive | Rendering | Use case |
|-------|-----------|-----------|----------|
| `ViewerGL` | Yes (window, mouse, keyboard, ImGui) | OpenGL | Development, demos |
| `ViewerUSD` | No | Writes `.usd` file | Offline rendering |
| `ViewerRerun` | Yes (external app) | Streams to Rerun | Data visualization |
| `ViewerViser` | Yes (web browser) | WebGL via Viser | Remote / web-based |
| `ViewerNull` | No | None | Testing, benchmarking |

All inherit from `ViewerBase` (ABC).

### 6.2 Core Viewer Methods

#### `set_model(model: newton.Model, max_worlds: int | None = None)`

Register the simulation model with the viewer. **Must be called once** before the
main loop. Populates shape/geometry data for rendering.

#### `is_running() -> bool`

Returns `True` while the viewer is open. Becomes `False` when:
- **ViewerGL:** ESC pressed or window closed
- **ViewerNull:** `frame_count >= num_frames`
- **ViewerBase (default):** always `True`

#### `is_paused() -> bool`

Returns current pause state.
- **ViewerGL:** toggled by **Space** key or ImGui checkbox
- **Others:** always `False`

#### `begin_frame(time: float)`

Start a new frame. Pass the current simulation time. Must be followed by
`end_frame()`.

#### `end_frame()`

Finalize and display the frame. In ViewerGL this processes window events, renders
the scene, and swaps buffers. In ViewerNull this increments the frame counter.

#### `apply_forces(state: newton.State)`

Apply interactive forces (picking via mouse drag, wind) to the state. Only
meaningful in ViewerGL. Must be called in `step()` before the solver step.

#### `close()`

Clean up viewer resources. Called automatically by `run()` after the loop.

#### `is_key_down(key: str | int) -> bool`

Check if a key is currently pressed. Accepts:
- Single characters: `'a'` through `'z'`, `'0'` through `'9'`
- Special names: `'space'`, `'escape'`, `'shift'`, `'ctrl'`, `'up'`, `'down'`,
  `'left'`, `'right'`
- Raw pyglet key codes (int)

Returns `False` in non-interactive viewers.

#### `set_camera(pos: wp.vec3, pitch: float, yaw: float)`

Set camera position and orientation. Only effective in ViewerGL.

#### `register_ui_callback(callback, position="side")`

Register a callable for ImGui rendering. Positions: `"side"`, `"stats"`, `"free"`.
ViewerGL only.

### 6.3 Logging / Drawing Methods

These are called from `render()` to visualize simulation state.

| Method | Signature | Description |
|--------|-----------|-------------|
| `log_state` | `(state: State)` | Update and render all body/shape transforms from state |
| `log_contacts` | `(contacts: Contacts, state: State)` | Draw contact normals as green lines (if `show_contacts` is True) |
| `log_points` | `(name, points, radii, colors)` | Render a point cloud |
| `log_lines` | `(name, starts, ends, colors, width)` | Render line segments |
| `log_mesh` | `(name, points, indices, normals, uvs, texture)` | Register/update a mesh |
| `log_instances` | `(name, mesh, xforms, scales, colors, materials)` | Render instanced meshes |
| `log_shapes` | `(name, geo_type, geo_scale, xforms, colors, ...)` | High-level: create geometry + render instances |
| `log_scalar` | `(name, value)` | Log a scalar value |
| `log_array` | `(name, array)` | Log an array |
| `log_gizmo` | `(name, transform)` | Interactive 3D gizmo (ViewerGL only) |

### 6.4 Display Flags

Boolean attributes on the viewer instance that control what is rendered:

| Flag | Default | What it shows |
|------|---------|---------------|
| `show_joints` | varies | Joint axes |
| `show_com` | varies | Center of mass markers |
| `show_particles` | varies | Particle positions |
| `show_contacts` | varies | Contact normal lines |
| `show_springs` | varies | Spring connections |
| `show_triangles` | varies | Cloth/mesh triangles |
| `show_collision` | varies | Collision shapes |
| `show_visual` | varies | Visual shapes |
| `show_static` | varies | Static bodies |
| `show_inertia_boxes` | varies | Inertia visualization |
| `picking_enabled` | varies | Mouse-drag force picking |

### 6.5 Built-in Key Bindings (ViewerGL)

| Key | Action |
|-----|--------|
| **Space** | Toggle pause |
| **ESC** | Close viewer (exit loop) |
| **H** | Toggle UI visibility |
| **F** | Frame camera on model |

---

## 7. Test Utilities

These functions are provided for use inside `test_final()` or `test_post_step()`.

### `test_body_state(model, state, test_name, test_fn, indices=None, ...)`

Tests body positions/velocities via a user-provided function. Raises `ValueError`
on failure.

```python
def test_final(self):
    newton.examples.test_body_state(
        self.model,
        self.state_0,
        "all bodies are above ground",
        lambda q, qd: q[2] > 0.0,  # q is wp.transform, qd is wp.spatial_vector
    )
```

Parameters:
- `test_fn`: `(wp.transform, wp.spatial_vectorf) -> bool` — evaluated per body
- `indices`: list of body indices to test (default: all)
- `show_body_q` / `show_body_qd`: include pose/twist in error messages

### `test_particle_state(state, test_name, test_fn, indices=None)`

Same pattern for particles. `test_fn: (wp.vec3, wp.vec3) -> bool` receives
particle position and velocity.

---

## 8. Other Utility Functions

| Function | Description |
|----------|-------------|
| `get_asset(filename)` | Returns absolute path to a file in the bundled assets directory |
| `get_asset_directory()` | Returns the assets directory path |
| `get_source_directory()` | Returns the examples source directory path |
| `create_collision_pipeline(model, args, broad_phase, **kwargs)` | Create a `CollisionPipeline` using `--broad-phase` from args |

---

## 9. Typical Application Skeleton

```python
import newton
import newton.examples
import newton.utils
import warp as wp


class MyApp:
    """Minimal application that integrates with newton.examples.run()."""

    def __init__(self, viewer, ...):
        self.viewer = viewer                    # REQUIRED: store viewer reference

        # Build model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        # ... configure builder ...
        self.model = builder.finalize()

        # Create solver
        self.solver = newton.solvers.SolverMuJoCo(self.model, ...)

        # Create state & control
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(...)

        # Register model with viewer
        self.viewer.set_model(self.model)

        # Simulation parameters
        self.sim_time = 0.0
        self.sim_dt = 1.0 / 200.0

    def step(self):                             # REQUIRED
        """Advance simulation by one step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)  # interactive picking/wind
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)
        self.sim_time += self.sim_dt

    def render(self):                           # REQUIRED
        """Render current state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ── Optional hooks ──

    def gui(self, ui):
        """Custom ImGui panel (only called if viewer supports UI)."""
        import imgui_bundle.imgui as imgui
        _, self.param = imgui.slider_float("Param", self.param, 0.0, 10.0)

    def test_final(self):
        """End-of-run validation (only called with --test)."""
        newton.examples.test_body_state(
            self.model, self.state_0,
            "bodies above ground",
            lambda q, qd: q[2] > 0.0,
        )

    def test_post_step(self):
        """Per-frame validation (only called with --test)."""
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--my-flag", action="store_true")
    viewer, args = newton.examples.init(parser)

    app = MyApp(viewer, ...)

    newton.examples.run(app, args)

    # ── Code here runs AFTER the viewer is closed ──
    # Use this for logging, saving data, printing summaries, etc.
    print("Done!")
```

---

## 10. FAQ / Gotchas

### Why is `test_final()` not called when I close the window?

`test_final()` is only called when you pass `--test` on the command line. For
cleanup/logging code that should always run, place it **after**
`newton.examples.run()` in your `__main__` block.

### How do I run headless (no window)?

```bash
python my_app.py --viewer null --num-frames 1000
```

The `null` viewer runs for `--num-frames` steps then `is_running()` returns `False`.

### How do I save simulation output to USD?

```bash
python my_app.py --viewer usd --output-path output.usd --num-frames 500
```

### Can I use multiple viewers simultaneously?

No. `set_model()` can only be called once per viewer, and `run()` expects a single
viewer on `example.viewer`.

### What happens if `step()` raises an exception?

The exception propagates out of `run()`. `viewer.close()` is **not** called in that
case. Consider wrapping in try/finally if you need guaranteed cleanup.

### How does pause work?

**Space** toggles pause in ViewerGL. When paused:
- `step()` is **skipped**
- `render()` is **still called** (you can orbit the camera, inspect the scene)
- `test_post_step()` is **still called** (if in test mode)

### Can I stop the loop programmatically?

Not directly. The loop runs while `viewer.is_running()` returns `True`. For
ViewerGL, you could call `self.viewer.close()` from within `step()` but this is
not the intended pattern. For headless runs, use `--viewer null --num-frames N`.

### What is `apply_forces()` for?

In ViewerGL, right-clicking on a body and dragging applies a force (picking). The
`Wind` feature can also apply forces. `apply_forces(state)` writes these external
forces into the state before the solver step. Always call it in `step()` for
interactive use.
