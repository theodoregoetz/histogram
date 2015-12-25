from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='histogram',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='2.0.0.dev1',

    description='A histogram object for scientific data-reduction and statistical analysis',
    long_description=long_description,

    # The project's main homepage.
    url='http://theodoregoetz.bitbucket.org/pyhep',

    # Author details
    author='Johann T. Goetz',
    author_email='theodore.goetz+histogram@gmail.com',

    # Choose your license
    license='GPLv3',

    classifiers=[

        # Development Status
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.2',
        #'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        #'Programming Language :: Python :: 3 :: Only',
    ],

    # What does your project relate to?
    keywords= [
        'histogram',
        'reduction',
        'data reduction',
        'data analysis',
        'scientific',
        'scientific computing',
        'statistics',
        'visualization',
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    #
    # for numpy/scipy on Fedora 23:
    #		sudo dnf install atlas-devel gcc-{c++,gfortran} subversion redhat-rpm-config
    # for extras on Fedora 23:
	#		sudo dnf install {freetype,libpng,hdf5}-devel root-{python,python3}
	#       pip install matplotlib h5py tables pyroot
	#
	# at the time of testing, Flask's pypi configuration was broken
	# which prevented bokeh from being installed through pip
	#
	# I installed PySide using pip and used Qt4Agg backend for matplotlib
	# to work in the virtual environment
    setup_requires=[
        'numpy',
    ],
    install_requires=[
        'numpy',
        'scipy',
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'test': ['coverage'],
        'all' : ['matplotlib',
				 'bokeh',
                 'h5py',
                 'pyroot'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    #package_data={
    #    'sample': ['package_data.dat'],
    #},

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    #data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    #entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
)
