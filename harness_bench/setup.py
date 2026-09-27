from setuptools import setup

package_name = 'harness_bench'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/harness_bench']),
        ('share/harness_bench', ['package.xml']),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='Reza',
    maintainer_email='rezahendi590@gmail.com',
    description='Cross-simulator benchmarks (MuJoCo vs Isaac Sim) for the wire-harness cell.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'run_benchmarks = harness_bench.run:main',
            'compare_benchmarks = harness_bench.compare:main',
        ],
    },
)
