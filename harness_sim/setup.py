from setuptools import setup

package_name = 'harness_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/harness_sim']),
        ('share/harness_sim', ['package.xml']),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='Reza',
    maintainer_email='rezahendi590@gmail.com',
    description='MuJoCo simulation node of the wire-harness cell with a UR-driver-like ROS 2 interface.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mujoco_sim_node = harness_sim.sim_node:main',
        ],
    },
)
