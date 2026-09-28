from glob import glob

from setuptools import setup

package_name = 'harness_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/harness_agent']),
        ('share/harness_agent', ['package.xml']),
        ('share/harness_agent/specs', glob('specs/*.yaml')),
        ('share/harness_agent/refs', glob('refs/*')),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='Reza',
    maintainer_email='rezahendi590@gmail.com',
    description='Nemotron build agent, harness specs and tool interface for the wire-harness cell.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'run_build = harness_agent.run_build:main',
            'check_nebius = harness_agent.check_nebius:main',
            'vision_eval = harness_agent.vision_eval:main',
            'benchmark = harness_agent.benchmark:main',
            'annotate_video = harness_agent.annotate:main',
            'replay3d = harness_agent.replay3d:main',
            'groot_data = harness_agent.groot_data:main',
            'groot_eval = harness_agent.groot_eval:main',
            'groot_replay_server = harness_agent.groot_replay_server:main',
        ],
    },
)
