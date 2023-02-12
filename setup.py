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
    packages=['fused_adan'],
    package_dir={'fused_adan': 'fused_adan/', 'fused_adan/include': 'fused_adan/include/'},
    package_data={'fused_adan': ['fused_adan/*.cu', 'fused_adan/*.cpp'], 'fused_adan/include': ['fused_adan/include/*.h', 'fused_adan/include/*.cuh']},
)
