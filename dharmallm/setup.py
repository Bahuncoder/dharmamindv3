"""
🕉️ DharmaLLM Package Setup
==========================

Setup configuration for the DharmaLLM spiritual AI package.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='dharmallm',
    version='0.1.0',
    description='AI with Soul powered by Dharma',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='DharmaMind Team',
    author_email='team@dharmamind.ai',
    url='https://github.com/dharmamind/dharmallm',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Philosophy',
        'Topic :: Religion',
    ],
    keywords='ai, dharma, spirituality, wisdom, llm',
    entry_points={
        'console_scripts': [
            'dharmallm-api=dharmallm.api.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)