import codecs
import io
import os
import re
import sys
import webbrowser
import platform

try:
    from setuptools import setup
except:
    from distutils.core import setup


NAME = "quantaxis_randomprice"
"""
名字，一般放你包的名字即可
"""
PACKAGES = ["QUANTAXIS_RandomPrice"]
"""
包含的包，可以多个，这是一个列表
"""

DESCRIPTION = "QUANTAXIS RANDOM PRICE"
KEYWORDS = ["quantaxis", "quant", "finance", "Backtest", 'Framework']
AUTHOR_EMAIL = "yutiansut@qq.com"
AUTHOR = 'yutiansut'
URL = "https://github.com/yutiansut/QUANTAXIS_RANDOMPRICE"


LICENSE = "MIT"

setup(
    name=NAME,
    version='1.1',
    description=DESCRIPTION,
    long_description='random price',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    install_requires=['click', 'numpy', 'pandas'],
    entry_points={
        'console_scripts': [
            'qarandom=QUANTAXIS_RandomPrice.__init__:generate'
        ]
    },
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=PACKAGES,
    include_package_data=True,
    zip_safe=True
)