import os
from codecs import open
from os import path
from subprocess import Popen, PIPE

VERSION = 2,0,0
ISRELEASE = False

def long_description():
    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
        return f.read()

# Return the git revision as a string
def revision():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = Popen(cmd, stdout=PIPE, env=env).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        rev = out.strip().decode('ascii')[:7]
    except OSError:
        rev = "Unknown"
    return rev

def write_version_py(filename):
    fmt = """\
# This file generated by setup.py
version = '{ver[0]}.{ver[1]}.{ver[2]}'
revision = '{rev}'
version_info = {ver[0]},{ver[1]},{ver[2]}
release = {isrel}
if not release:
    version = version+'r'+revision
"""
    opts = dict(
        ver = VERSION,
        isrel = ISRELEASE,
        rev = revision())

    with open(filename,'w') as fout:
        fout.write(fmt.format(**opts))

def setup_opts():
    # List run-time dependencies with setup_requires and install_requires.
    # These will be installed by pip when your project is installed. For an
    # analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    #
    # at the time of testing, Flask's pypi configuration was broken
    # which prevented bokeh from being installed through pip
    #
    # I installed PySide using pip and used Qt4Agg backend for matplotlib
    # to work in the virtual environment.

    # Fedora 23
    #   base:       dnf install atlas-devel gcc-{c++,gfortran} subversion redhat-rpm-config
    #   extras:     dnf install {freetype,libpng,hdf5}-devel root-{python,python3}
    #
    # Ubuntu 14.04 LTS
    #   base:       apt-get install build-essential python-dev python3-dev g++ gfortran \
    #                               libatlas-dev libatlas3-base liblapack-dev libblas-dev
    #   extras:     apt-get install libpng12-dev libfreetype6-dev libroot-bindings-python5.34 \
    #                               libhdf5-dev

    opts = dict(
        name='histogram',
        version='.'.join(str(x) for x in VERSION),
        description='A histogram object for scientific data-reduction and statistical analysis',
        long_description=long_description(),
        url='https://github.com/theodoregoetz/histogram',
        author='Johann T. Goetz',
        author_email='theodore.goetz+histogram@gmail.com',
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
            'Programming Language :: Python :: 2',
            #'Programming Language :: Python :: 2.6',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            #'Programming Language :: Python :: 3.2',
            #'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            #'Programming Language :: Python :: 3 :: Only',
        ],
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
        packages=['histogram'],
        setup_requires=['numpy'],
        install_requires=['numpy','scipy'],
        extras_require={
            'test': ['coverage','nose'],
            'all' : ['matplotlib','cycler','bokeh','h5py'],
        },
    )
    return opts

if __name__ == '__main__':
    from setuptools import setup
    write_version_py('histogram/version.py')
    setup(**setup_opts())
