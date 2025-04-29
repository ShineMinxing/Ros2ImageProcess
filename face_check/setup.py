#!/usr/bin/env python3
import os
from glob import glob
from setuptools import setup

package_name = 'face_check'

# 列出打包的数据文件，路径必须相对于包根目录
data_files = [
    # ROS2 索引文件
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    # 包的 package.xml
    ('share/' + package_name, ['package.xml']),
    # other 目录下所有图片
    ('share/' + package_name + '/other', glob('other/*')),
    # local_file 目录下所有图片
    ('share/' + package_name + '/local_file', glob('local_file/*')),
]

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smx',
    maintainer_email='you@example.com',
    description='ROS2 face recognition node for Go2 camera',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'face_check_node = face_check.face_check_node:main',
        ],
    },
)
