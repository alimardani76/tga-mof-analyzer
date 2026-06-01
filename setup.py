from setuptools import setup, find_packages

setup(
    name="tga-mof-analyzer",
    version="0.3.0",
    description="Automated TGA analysis for metal-organic frameworks",
    author="Hosein Alimardani",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib>=3.5",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "app": ["streamlit>=1.28"],
    },
    entry_points={
        "console_scripts": [
            "tga-analyze=tga_analyze:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)