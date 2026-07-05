"""World/model construction helpers built on MuJoCo's ``MjSpec`` API.

A bare robot description (MJCF/URDF from robot_descriptions) usually has no world: no ground
plane, no light, and possibly no free joint on the root body. These helpers use ``mujoco.MjSpec``
to attach a robot into a minimal world (worldbody + plane + light) and to make the base fixed or
floating (structural free joint), then recompile to a fresh ``MjModel``.

Coming soon.
"""

# Coming soon: wrap_in_world(), set_base_freejoint(), etc.
