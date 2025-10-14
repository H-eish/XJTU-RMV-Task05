import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration,EqualsSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    
    input_source_arg = DeclareLaunchArgument(
        'input_source', default_value='camera',
        description='Input source: "camera" to use the Hikvision camera, "video" to use a video file.'
    )
    
    video_path_arg = DeclareLaunchArgument(
        'video_path', default_value='/home/heish/Documents/Projects/XJTU-RMV-Task05/armor_detector/src/blue.mp4', # <--- 修改: 这里设置默认视频路径
        description='Abosolute path to the video file to be used as input (if use_video is true).'
    )
    
    # 定义节点的参数
    # 您可以在这里修改默认值，或者在启动时通过命令行覆盖
    armor_detector_params = [
        {'camera_sn': ''},                  # 相机序列号，如果为空，则连接第一个找到的相机
        {'exposure_time': 8000.0},          # 曝光时间 (us)
        {'gain': 5.0},                      # 增益
        {'frame_rate': 30.0},               # 帧率
        {'pixel_format': 'bgr8'},           # 像素格式 ('mono8', 'bgr8', 'rgb8', etc.)
        {'camera_frame_id': 'camera_optical_frame'}, # 图像坐标系
        {'topic_name': 'image_raw'},       # 图像话题名称
        {'processed_topic_name': 'image_processed'}, # 处理后图像话题名称
        {'input_source': LaunchConfiguration('input_source')}, # 输入源 ('camera' or 'video')
        {'video_path': LaunchConfiguration('video_path')}  # 视频文件路径（如果使用视频作为输入）
    ]
    
    ld = LaunchDescription()
    
    # 定义要启动的节点
    armor_detector_node = Node(
        package='armor_detector',         # 您的功能包名称
        executable='armor_detector_node',      # 在CMakeLists.txt中定义的可执行文件名称
        name='armor_detector',                 # 节点的运行时名称
        output='screen',                   # 将节点的输出打印到屏幕
        emulate_tty=True,                  # 模拟一个终端，以确保日志颜色和格式正确
        parameters=armor_detector_params,      # 传递参数给节点
    )
    
    pkg_share_dir = get_package_share_directory('armor_detector')
    
    camera_rviz_config_path = os.path.join(pkg_share_dir, 'rviz', 'camera.rviz')
    video_rviz_config_path = os.path.join(pkg_share_dir, 'rviz', 'video.rviz')
    
    camera_rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_camera',
        output='screen',
        arguments=['-d', camera_rviz_config_path],
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('input_source'), 'camera'))
    )
    
    video_rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_video',
        output='screen',
        arguments=['-d', video_rviz_config_path],
        condition=IfCondition(EqualsSubstitution(LaunchConfiguration('input_source'), 'video'))
    )
    
    ld.add_action(input_source_arg)
    ld.add_action(video_path_arg)
    ld.add_action(armor_detector_node)
    ld.add_action(camera_rviz_node)
    ld.add_action(video_rviz_node)
    
    return ld