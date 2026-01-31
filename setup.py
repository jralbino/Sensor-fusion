from setuptools import setup, find_packages

setup(
    name='sensor_fusion',
    version='0.1.0',
    packages=find_packages(),
    author='Your Name',
    author_email='your.email@example.com',
    description='A multi-modal sensor fusion project for autonomous driving.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jralbino/Sensor-fusion',  # Replace with your repo URL
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires='>=3.8',
    extras_require={
        'vision': open('Vision/requirements.txt').read().splitlines(),
        'lidar': open('Lidar/requirements.txt').read().splitlines(),
        'fusion': open('Fusion/requirements.txt').read().splitlines(),
        'dev': open('requirements-dev.txt').read().splitlines(),
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
    ],
    entry_points={
        'console_scripts': [
            'vision=Vision.cli:cli',
            'lidar_pipeline=Lidar.main:main',
        ],
    },
)
