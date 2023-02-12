from setuptools import setup, find_packages

setup(
    name='adan',
    python_requires='>=3.8',
    version='0.0.1',
    install_requires=['torch'],
    py_modules=['adan'],
    description=(
        'Adan: Adaptive Nesterov Momentum Algorithm for '
        'Faster Optimizing Deep Models'
    ),
    author=(
        'Xie, Xingyu and Zhou, Pan and Li, Huan and '
        'Lin, Zhouchen and Yan, Shuicheng'
    ),
    packages=find_packages(where="fused_adan"),
    package_dir={"": "fused_adan"},
    include_package_data=True,
)
