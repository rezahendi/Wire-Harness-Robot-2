from setuptools import setup

package_name = 'harness_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/harness_core']),
        ('share/harness_core', ['package.xml']),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='Reza',
    maintainer_email='rezahendi590@gmail.com',
    description='ROS-independent core of the wire-harness cell: MuJoCo scene generator, UR5e kinematics, Cartesian compliance controller and the force-guided routing expert.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
        ],
    },
)
