"""USD/PhysX builders for the harness benchmark rigs.

The same ``CellConfig`` that generates the MuJoCo model generates these scenes, so the
wire, the fork and the board have identical dimensions and material parameters in both
engines. What necessarily differs is *how* the wire is modelled: MuJoCo integrates a
Cosserat rod (the cable plugin), PhysX has no rod primitive, so the wire is the usual
articulated chain of capsules whose joint drives are set to the same bending stiffness:

    per-joint rotational stiffness  k = EI / L_segment      [N m / rad]

which is the standard lumped-parameter discretisation of a beam. Whether that gets the
same physics out is exactly what the benchmarks measure.

Angles in USD physics are degrees, so drive stiffness is converted from per-radian to
per-degree (k_deg = k_rad * pi / 180).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

DEG = math.pi / 180.0


def _pxr():
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    return Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def setup_stage(stage, timestep: float, gravity: float = 9.81,
                solver_position_iterations: int = 32, solver_velocity_iterations: int = 4):
    """Metric stage, TGS solver, fixed timestep, high iteration counts for a stiff chain."""
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene_path = "/World/physicsScene"
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(gravity)
    px = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    px.CreateTimeStepsPerSecondAttr().Set(int(round(1.0 / timestep)))
    px.CreateSolverTypeAttr().Set("TGS")
    px.CreateEnableCCDAttr().Set(True)
    px.CreateEnableStabilizationAttr().Set(True)
    px.CreateSolverPositionIterationCountAttr().Set(solver_position_iterations)
    px.CreateSolverVelocityIterationCountAttr().Set(solver_velocity_iterations)
    return scene


def physics_material(stage, path: str, static_friction: float, dynamic_friction: float,
                     restitution: float = 0.0, combine: str = "max"):
    """PhysX material; friction combine mode set to `max` to match MuJoCo's convention."""
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(static_friction)
    api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    api.CreateRestitutionAttr().Set(restitution)
    pm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    try:
        pm.CreateFrictionCombineModeAttr().Set(combine)
        pm.CreateRestitutionCombineModeAttr().Set("min")
    except Exception:
        pass
    return mat


def bind_material(stage, prim_path: str, material) -> None:
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics")


def add_ground_box(stage, path: str, size: Sequence[float], pos: Sequence[float],
                   material=None) -> str:
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(s) for s in size]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if material is not None:
        bind_material(stage, path, material)
    return path


def _capsule(stage, path: str, radius: float, length: float, mass: float,
             pos, material=None, contact_offset: float = 0.0005):
    """Capsule rigid body along +x, its axis starting at the body origin."""
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    body = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr().Set(mass)
    cap = UsdGeom.Capsule.Define(stage, path + "/collision")
    cap.CreateAxisAttr().Set("X")
    cap.CreateRadiusAttr().Set(radius)
    cap.CreateHeightAttr().Set(length)
    UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(length / 2.0, 0.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim())
    px.CreateContactOffsetAttr().Set(contact_offset)
    px.CreateRestOffsetAttr().Set(0.0)
    if material is not None:
        bind_material(stage, path + "/collision", material)
    return path


def add_wire(stage, root: str, n_segments: int, segment_length: float, radius: float,
             density: float, EI: float, twist_stiffness: float, joint_damping: float,
             start: Sequence[float], material=None, bend_limit_deg: float = 60.0) -> Dict:
    """Chain of capsules with D6 joints whose drives carry the bending stiffness.

    The chain runs along +x from ``start``. The first body is connected to the world by
    a fixed joint (a clamped wire); callers that want a free or pinned end can replace
    that joint. Returns the prim paths of the segments.
    """
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    UsdGeom.Xform.Define(stage, root)
    seg_mass = math.pi * radius * radius * segment_length * density
    k_bend = EI / segment_length                     # N m / rad
    k_twist = twist_stiffness * (math.pi * radius ** 4 / 2.0) / segment_length
    paths: List[str] = []
    for i in range(n_segments):
        # bodies are siblings in world space (the joints, not the hierarchy, hold the chain)
        path = f"{root}/segment_{i:03d}"
        _capsule(stage, path, radius, segment_length, seg_mass,
                 (start[0] + i * segment_length, start[1], start[2]), material)
        paths.append(path)
    # joints between consecutive segments: rotations free (spring to straight), translations locked
    for i in range(1, n_segments):
        joint = UsdPhysics.Joint.Define(stage, f"{root}/joint_{i:03d}")
        joint.CreateBody0Rel().SetTargets([paths[i - 1]])
        joint.CreateBody1Rel().SetTargets([paths[i]])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(segment_length, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateExcludeFromArticulationAttr().Set(False)
        for axis in ("transX", "transY", "transZ"):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr().Set(0.0)
            limit.CreateHighAttr().Set(0.0)          # locked
        for axis, k in (("rotX", k_twist), ("rotY", k_bend), ("rotZ", k_bend)):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr().Set(-bend_limit_deg)
            limit.CreateHighAttr().Set(bend_limit_deg)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateTargetPositionAttr().Set(0.0)     # rest shape: straight
            drive.CreateStiffnessAttr().Set(k * DEG)      # USD drives are per degree
            drive.CreateDampingAttr().Set(joint_damping * DEG)
            drive.CreateMaxForceAttr().Set(1.0e6)
    return {"segments": paths, "segment_length": segment_length, "radius": radius,
            "bend_stiffness": k_bend, "twist_stiffness": k_twist, "mass": seg_mass}


def fix_to_world(stage, path: str, body: str, local_pos=(0.0, 0.0, 0.0)) -> str:
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody1Rel().SetTargets([body])
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos))
    return path


def pin_to_world(stage, path: str, body: str, world_pos, local_pos=(0.0, 0.0, 0.0)) -> str:
    """Ball joint to the world: position held, rotation free (a pinned wire end)."""
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    joint = UsdPhysics.SphericalJoint.Define(stage, path)
    joint.CreateBody1Rel().SetTargets([body])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in world_pos]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*local_pos))
    return path


def add_prismatic_hand(stage, root: str, axis: str, pos: Sequence[float], mass: float,
                       stiffness: float, damping: float, lower: float, upper: float) -> Dict:
    """A body on a driven prismatic joint to the world: the rig's 'gripper'.

    The drive stiffness doubles as a force gauge: with a quasi-static motion, the force
    the environment applies is stiffness * (target - position), which is how the cell's
    compliance controller infers force as well.
    """
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    body = UsdGeom.Xform.Define(stage, root)
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(mass)
    joint = UsdPhysics.PrismaticJoint.Define(stage, root + "_joint")
    joint.CreateBody1Rel().SetTargets([root])
    joint.CreateAxisAttr().Set(axis.upper())
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in pos]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr().Set(lower)
    joint.CreateUpperLimitAttr().Set(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateTargetPositionAttr().Set(0.0)
    drive.CreateStiffnessAttr().Set(stiffness)
    drive.CreateDampingAttr().Set(damping)
    drive.CreateMaxForceAttr().Set(1.0e6)
    return {"body": root, "joint": root + "_joint", "drive_stiffness": stiffness}


def add_fork(stage, root: str, cfg, pose: Sequence[float], material=None) -> Dict:
    """The snap-in fork: static post plus two spring-loaded jaws with barbed lips.

    Geometry and spring parameters come from ``cfg.fork``, the same values the MJCF
    generator uses. The jaws are prismatic joints with a drive whose stiffness is the
    jaw spring and whose target is the preloaded closed position.
    """
    Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade = _pxr()
    f = cfg.fork
    x, y, z = pose
    UsdGeom.Xform.Define(stage, root)
    UsdGeom.Xformable(stage.GetPrimAtPath(root)).AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    half_d = f.depth / 2.0
    post_half_y = f.slot_width / 2.0 + f.prong_thickness
    post = UsdGeom.Cube.Define(stage, root + "/post")
    post.CreateSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(post)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, f.post_height / 2.0))
    xf.AddScaleOp().Set(Gf.Vec3f(2 * half_d, 2 * post_half_y, f.post_height))
    UsdPhysics.CollisionAPI.Apply(post.GetPrim())
    if material is not None:
        bind_material(stage, root + "/post", material)

    jaws = {}
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        jaw_root = f"{root}/jaw_{side}"
        y0 = sgn * (f.slot_width / 2.0 + f.prong_thickness / 2.0)
        body = UsdGeom.Xform.Define(stage, jaw_root)
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0.0, y0, f.post_height))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(0.002)
        prong = UsdGeom.Cube.Define(stage, jaw_root + "/prong")
        prong.CreateSizeAttr().Set(1.0)
        pf = UsdGeom.Xformable(prong)
        pf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, f.prong_height / 2.0))
        pf.AddScaleOp().Set(Gf.Vec3f(2 * half_d, f.prong_thickness, f.prong_height))
        UsdPhysics.CollisionAPI.Apply(prong.GetPrim())
        lip = UsdGeom.Capsule.Define(stage, jaw_root + "/lip")
        lip.CreateAxisAttr().Set("X")
        lip.CreateRadiusAttr().Set(f.lip_radius)
        lip.CreateHeightAttr().Set(f.depth)
        lf = UsdGeom.Xformable(lip)
        lf.AddTranslateOp().Set(Gf.Vec3d(-half_d, sgn * (f.lip_gap / 2.0 + f.lip_radius) - y0,
                                         f.prong_height - f.lip_radius))
        UsdPhysics.CollisionAPI.Apply(lip.GetPrim())
        barb = UsdGeom.Cube.Define(stage, jaw_root + "/barb")
        barb.CreateSizeAttr().Set(1.0)
        bf = UsdGeom.Xformable(barb)
        barb_half_y = (post_half_y - f.prong_thickness / 2.0 - f.lip_gap / 2.0) / 2.0
        bf.AddTranslateOp().Set(Gf.Vec3d(0.0, sgn * (f.lip_gap / 2.0 + barb_half_y) - y0,
                                         f.prong_height - 1.5 * f.lip_radius))
        bf.AddScaleOp().Set(Gf.Vec3f(2 * half_d, 2 * barb_half_y, f.lip_radius))
        UsdPhysics.CollisionAPI.Apply(barb.GetPrim())
        if material is not None:
            for sub in ("prong", "lip", "barb"):
                bind_material(stage, f"{jaw_root}/{sub}", material)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"{root}/jaw_{side}_joint")
        joint.CreateBody1Rel().SetTargets([jaw_root])
        joint.CreateAxisAttr().Set("Y")
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(x, y + y0, z + f.post_height))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLowerLimitAttr().Set(0.0 if sgn > 0 else -f.jaw_travel)
        joint.CreateUpperLimitAttr().Set(f.jaw_travel if sgn > 0 else 0.0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr().Set("force")
        drive.CreateTargetPositionAttr().Set(-sgn * f.spring_preload)   # preloaded closed
        drive.CreateStiffnessAttr().Set(f.spring_stiffness)
        drive.CreateDampingAttr().Set(f.damping)
        drive.CreateMaxForceAttr().Set(1.0e4)
        jaws[side] = {"body": jaw_root, "joint": f"{root}/jaw_{side}_joint"}
    return {"root": root, "jaws": jaws}
