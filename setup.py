from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='style_tta',
    version='0.1.0',
    description='RetriStyle-TTA: Structure-Aware Retrieval and Diffusion for Test-Time Stain Adaptation.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='xAILab Bamberg',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/style_tta',
    packages=find_packages(exclude=['experiments', 'examples', 'tests']),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "scikit-image>=0.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.10.0",
        "diffusers>=0.21.0",
        "transformers>=4.25.0",
        "accelerate>=0.20.0",
        "faiss-cpu>=1.7.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.990',
        ],
        'experiments': [
            'pandas>=2.0.0',
            'matplotlib>=3.8.0',
            # scikit-image pinned for medmnistc compatibility
            'scikit-image==0.23.2',
            'tqdm>=4.60.0',
            'ptflops>=0.7.5',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.10',
    keywords='color-transfer data-augmentation computer-vision pytorch torchvision transforms',
)
