from setuptools import setup

package_name = 'harness_task'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/harness_task']),
        ('share/harness_task', ['package.xml']),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='Reza',
    maintainer_email='rezahendi590@gmail.com',
    description='RouteHarness action server running the force-guided expert or a learned policy through ROS 2.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'routing_task_node = harness_task.task_node:main',
            'route_harness = harness_task.route_client:main',
        ],
    },
)
