from setuptools import setup

package_name = 'harness_learning'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/harness_learning']),
        ('share/harness_learning', ['package.xml']),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='Reza',
    maintainer_email='rezahendi590@gmail.com',
    description='Gymnasium environment, expert demonstration recording and replay for learning wire routing.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'record_demos = harness_learning.record_demos:main',
            'replay_demo = harness_learning.replay_demo:main',
            'inspect_demos = harness_learning.inspect_demos:main',
            'run_expert = harness_learning.run_expert:main',
        ],
    },
)
