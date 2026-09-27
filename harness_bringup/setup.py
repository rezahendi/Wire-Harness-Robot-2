import os
from glob import glob

from setuptools import setup

package_name = 'harness_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/harness_bringup']),
        ('share/harness_bringup', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='Reza',
    maintainer_email='rezahendi590@gmail.com',
    description='Launch files and configuration that bring up the simulated wire-harness robot cell.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
        ],
    },
)
