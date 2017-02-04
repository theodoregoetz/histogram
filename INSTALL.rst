Installation Guidance
=====================

For the most part, if you are in the base directory, you can run:

    pip install .[all]

and all dependencies will be included. These will rely on your system having a bunch of things installed. You may not need all of the following packages - this the "maximal list" if installing everything including numpy and scipy through pip:

Fedora 25
  base:       dnf install {python,python3,altas}-devel gcc-{c++,gfortran} subversion \
                          redhat-rpm-config Cython python3-Cython
  extras:     dnf install {freetype,libpng,hdf5}-devel root-{python,python3}

Ubuntu 14.04 LTS
  base:       apt-get install build-essential {python,python3}-dev g++ gfortran \
                              lib{atlas,atlas3,lapack,blas}-dev
  extras:     apt-get install lib{png12,freetype6,hdf5}-dev libroot-bindings-python5.34
