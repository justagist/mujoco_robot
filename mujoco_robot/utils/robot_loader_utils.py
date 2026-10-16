"""Utils for fetching robot description files (URDF/MJCF) via robot_descriptions.

MuJoCo builds prefer MJCF: use :func:`get_mjcf_from_awesome_robot_descriptions`.

``robot_descriptions`` is an optional dependency (``pip install "mujoco_robot[descriptions]"``).
It is imported only when one of these helpers (or ``AWESOME_ROBOTS``) is used.
"""

import logging
import importlib

_INSTALL_HINT = (
    "robot_descriptions is not installed; install it with "
    '`pip install "mujoco_robot[descriptions]"`.'
)


def _require_robot_descriptions():
    """Import robot_descriptions, or raise an ImportError that says how to install it."""
    try:
        importlib.import_module("robot_descriptions")
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e


def __getattr__(name: str):
    # AWESOME_ROBOTS: list of all available robots from the Awesome Robots List that can be
    # used. Resolved on first access so that importing this module never needs the package.
    if name == "AWESOME_ROBOTS":
        _require_robot_descriptions()
        descriptions = importlib.import_module("robot_descriptions._descriptions").DESCRIPTIONS
        return list(descriptions.keys())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_file_from_ard_pkg(description_pkg_name, filetype):
    _require_robot_descriptions()
    try:
        return getattr(
            importlib.import_module(f"robot_descriptions.{description_pkg_name}"),
            f"{filetype}_PATH",
        )
    except AttributeError as e:
        msg = f"{filetype} has not been provided for robot {description_pkg_name}."
        logging.error(msg)
        raise AttributeError(msg) from e
    except (ModuleNotFoundError, KeyError) as e:
        msg = (
            f"No description package called {description_pkg_name} in"
            " Awesome Robot Descriptions list. Use one of the description names from "
            "https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions."
        )
        logging.error(msg)
        raise ModuleNotFoundError(msg) from e


def get_urdf_from_awesome_robot_descriptions(
    robot_description_pkg_name: str,
) -> str:
    """Get robot urdf for the specified robot description package.

    The specified package should be from the list of awesome robot descriptions
    (https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions).

    The list of available robot descriptions can also be viewed in the variable `AWESOME_ROBOTS`
    imported from `mujoco_robot.utils.robot_loader_utils`.

    Downloads description package for the specified robot and caches it locally (only needs
    downloading once).

    Args:
        robot_description_pkg_name (str): The package name as specified in the Awesome Robot
            Descriptions list.

    Returns:
        str: Path to the robot's urdf file.
    """
    return _get_file_from_ard_pkg(
        description_pkg_name=robot_description_pkg_name, filetype="URDF"
    )


def get_mjcf_from_awesome_robot_descriptions(
    robot_description_pkg_name: str,
) -> str:
    """Get robot mjcf file for the specified robot description package.

    The specified package should be from the list of awesome robot descriptions
    (https://github.com/robot-descriptions/robot_descriptions.py/tree/main?tab=readme-ov-file#descriptions).

    The list of available robot descriptions can also be viewed in the variable `AWESOME_ROBOTS`
    imported from `mujoco_robot.utils.robot_loader_utils`.

    Downloads description package for the specified robot and caches it locally (only needs
    downloading once).

    Args:
        robot_description_pkg_name (str): The package name as specified in the Awesome Robot
            Descriptions list.

    Returns:
        str: Path to the robot's MJCF file.
    """
    return _get_file_from_ard_pkg(
        description_pkg_name=robot_description_pkg_name, filetype="MJCF"
    )
