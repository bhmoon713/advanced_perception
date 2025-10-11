from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='advanced_perception',
            executable='topdown_circle_detection',
            name='topdown_circle_detection',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
        Node(
            package='advanced_perception',
            executable='cup_hole_inspector',
            name='cup_hole_inspector',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
    ])
