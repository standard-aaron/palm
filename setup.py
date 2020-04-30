from palm import VERSION
from distutils.core import setup
setup(
    name='palm',
    version=VERSION,
    description='PALM (Polygenic Adaptation Likelihood Method)\n'+
		'Estimates selection acting on complex traits',

    author='Aaron Stern',
    author_email='ajstern@berkeley.edu',
    packages=['palm'],
    install_requires=['numpy>=1.14.2', 'scipy>=0.19.0','numba>=0.42.0','progressbar>=0.0.0','colorama>=0.3.9'],
    entry_points={
        'console_scripts': ['palm = palm.frontend:main'],
    },
)
