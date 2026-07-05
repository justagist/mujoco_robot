"""World/model construction helpers built on MuJoCo's ``MjSpec`` API.

A bare robot description (MJCF/URDF from robot_descriptions) usually has no world: no ground
plane, no light, and possibly no free joint on the root body. These helpers use ``mujoco.MjSpec``
to attach a robot into a minimal world (worldbody + plane + light) and to make the base fixed or
floating (structural free joint), then recompile to a fresh ``MjModel``.

Coming soon.
"""

# Coming soon: wrap_in_world(), set_base_freejoint(), etc.
#
# NOTE: MjSpec's compiler defaults to degrees (mujoco.MjSpec().compiler.degree is True). When
# building or modifying a model here, set spec.compiler.degree = False (or use unit-free
# quaternions / axis vectors) so any angle literals are radians, matching the radian public API.
