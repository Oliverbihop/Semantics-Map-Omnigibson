from setuptools import find_packages, setup

package_name = 'behavior'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='luan',
    maintainer_email='luan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'map_reader_node = behavior.map_reader:main',
        'navigate_node = behavior.navigation_launch:main',
        'ros2_og_bridge = behavior.ros2_og_bridge:main',
        'tf_publisher = behavior.tf_link:main',
        'goal_publisher = behavior.select_goal:main',
        ],
    },
)
