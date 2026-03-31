from setuptools import setup
import os
from glob import glob

package_name = 'mcl_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aun',
    maintainer_email='you@example.com',
    description='Monte Carlo Localization node for ROS 2 Humble',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mcl_node = mcl_localization.mcl_node:main',
        ],
    },
)
