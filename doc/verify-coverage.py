#!/usr/bin/env python3

import re
import sys

from os import path

import histogram

ignored_methods = '''
    _CutResult __class__ __copy__ __deepcopy__ __delattr__ __dict__ __dir__
    __doc__ __format__ __ge__ __getattribute__ __gt__ __hash__ __init__ __le__
    __lt__ __module__ __ne__ __new__ __reduce__ __reduce_ex__ __repr__
    __setattr__ __sizeof__ __subclasshook__ __weakref__ __getitem__
    __setitem__'''.split()

def verify_documentation(docsrc, cls):
    ptrn_summary = re.compile(r'^    \.\. autosummary::((:?\n        \w+)+)', re.M)
    ptrn_method = re.compile(r'^\.\. \w+:: {}\.(\w+)'.format(cls.__name__))

    methods_in_summary = []
    methods_in_doc = []
    with open(docsrc, 'rt') as fin:
        src = fin.read()
        for m in ptrn_summary.finditer(src, re.M):
            methods_in_summary.extend(m.group(1).split())
        for line in src.split('\n'):
            m = ptrn_method.match(line)
            if m:
                methods_in_doc.append(m.group(1))

    methods_in_cls = dir(cls)

    print('### {} methods in documentation but not in summary ###'.format(cls.__name__))
    for m in methods_in_doc:
        if m not in methods_in_summary:
            if m not in ignored_methods:
                print(m)
    print('###')

    print('### {} methods in summary but not in documentation ###'.format(cls.__name__))
    for m in methods_in_summary:
        if m not in methods_in_doc:
            if m not in ignored_methods:
                print(m)
    print('###')

    print('### {} methods not in documentation ###'.format(cls.__name__))
    for m in methods_in_cls:
        if m not in methods_in_doc:
            if m not in ignored_methods:
                print(m)
    print('###')


    print('### {} methods not in implementation ###'.format(cls.__name__))
    for m in methods_in_doc:
        if m not in methods_in_cls:
            if m not in ignored_methods:
                print(m)
    print('###')


def main():
    docsrc = 'source/histogram.rst'
    cls = histogram.Histogram
    verify_documentation(docsrc, cls)


    docsrc = 'source/histogram_axis.rst'
    cls = histogram.HistogramAxis
    verify_documentation(docsrc, cls)



if __name__ == '__main__':
    main()
