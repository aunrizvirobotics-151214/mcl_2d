"""
mcl_localization launch file
=============================
Starts:
  1. map_server            — serves ~/robotics/ros2_ws/maps/my_map_save.yaml
  2. lifecycle_manager     — activates map_server automatically
  3. mcl_node              — particle filter, subscribes /map /scan /odom
  4. rviz2                 — particles, likelihood map, laser, estimated pose

Usage:
  ros2 launch mcl_localization mcl.launch.py

Override map at runtime:
  ros2 launch mcl_localization mcl.launch.py \
      map:=/absolute/path/to/other_map.yaml

Override params:
  ros2 launch mcl_localization mcl.launch.py \
      params_file:=/path/to/my_params.yaml
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, LifecycleNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg = FindPackageShare('mcl_localization')

    # ── Launch arguments ─────────────────────────────────────────────────────
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.expanduser(
            '~/robotics/ros2_ws/maps/my_map_save.yaml'),
        description='Absolute path to the saved map YAML file',
    )

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([pkg, 'config', 'mcl_params.yaml']),
        description='Full path to the MCL parameters YAML file',
    )

    rviz_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([pkg, 'config', 'mcl_rviz.rviz']),
        description='Full path to the RViz2 config file',
    )

    # ── 1. map_server (lifecycle node) ───────────────────────────────────────
    # Publishes /map with Transient Local QoS so any late subscriber
    # (including mcl_node) still receives the map even after it starts.
    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[{
            'yaml_filename': LaunchConfiguration('map'),
            'frame_id': 'map',
        }],
    )

    # ── 2. lifecycle_manager ─────────────────────────────────────────────────
    # Drives map_server through unconfigured → inactive → active automatically.
    # Without this map_server never starts publishing.
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server'],
        }],
    )

    # ── 3. MCL node ──────────────────────────────────────────────────────────
    mcl_node = Node(
        package='mcl_localization',
        executable='mcl_node',
        name='mcl_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        # /scan and /odom already match your Gazebo robot — no remapping needed
    )

    # ── 4. RViz2 ─────────────────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        output='screen',
    )

    return LaunchDescription([
        map_arg,
        params_arg,
        rviz_arg,
        map_server,
        lifecycle_manager,
        mcl_node,
        rviz_node,
    ])
