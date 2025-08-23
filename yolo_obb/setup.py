from setuptools import setup

package_name = 'yolo_obb'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SMX',
    maintainer_email='401435318@qq.com',
    description='YOLOv11-OBB GPU inference ROS2 node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_obb_node = yolo_obb.yolo_obb_node:main',
        ],
    },
)
