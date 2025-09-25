from setuptools import setup, find_packages

setup(
    name="KinMethyl",
    version="0.1.0",
    author="Jichen Zhang",
    description="Robust methylation detection in prokaryotic SMRT sequencing via kinetic signal modeling and deep feature integration",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ZhangBio/KinMethyl",
    packages=find_packages(),  # 自动找到 kinmethyl/ 目录里的模块
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "kinmethyl-train=kinmethyl.train:main",
            "kinmethyl-test=kinmethyl.test:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
