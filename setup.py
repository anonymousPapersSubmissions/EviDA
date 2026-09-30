from setuptools import setup, find_packages

setup(
    name="fake-news-detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'transformers>=4.30.0',
        'timm>=0.9.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'pillow>=9.5.0',
        'opencv-python>=4.8.0',
        'scikit-image>=0.21.0',
        'albumentations>=1.3.0',
        'scikit-learn>=1.3.0',
        'imbalanced-learn>=0.11.0',
        'tensorboard>=2.13.0',
        'tqdm>=4.65.0',
    ],
    python_requires='>=3.8',
    author="Your Name",
    description="Production-ready multimodal fake news detection system",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)