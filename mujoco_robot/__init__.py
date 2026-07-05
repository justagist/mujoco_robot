from importlib.metadata import PackageNotFoundError, version

from .mujoco_robot import MujocoRobot

try:
    __version__ = version("mujoco_robot")
except PackageNotFoundError:
    # package not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

__all__ = ["MujocoRobot", "__version__"]
