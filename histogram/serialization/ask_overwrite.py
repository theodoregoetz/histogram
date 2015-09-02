import sys
import os
from distutils.util import strtobool

import datetime

from .. import rc

def ask_overwrite(filepath):
    global rc
    if not os.path.exists(filepath):
        dirpath = os.path.dirname(os.path.abspath(filepath))
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except:
                print('could not create directory: {}'.format(dirpath))
                return False
        return True
    if rc.overwrite.timestamp is not None:
        if (datetime.datetime.now() - rc.overwrite.timestamp).seconds > rc.overwrite.timeout:
            rc.overwrite.overwrite = 'ask'
            rc.overwrite.timestamp = None
    if rc.overwrite.overwrite == 'never':
        return False
    elif rc.overwrite.overwrite == 'always':
        return True
    else:
        sys.stdout.write('Overwrite file {}? [Yes, No, nEver, Always]\n'.format(filepath))
        while True:

            if sys.version_info < (3,0):
                intext = raw_input().lower()
            else:
                intext = input().lower()

            try:
                return strtobool(intext)
            except ValueError:
                if intext in ['never','e','none']:
                    rc.overwrite.overwrite = 'never'
                    rc.overwrite.timestamp = datetime.datetime.now()
                    return False
                elif intext in ['always','a','all']:
                    rc.overwrite.overwrite = 'always'
                    rc.overwrite.timestamp = datetime.datetime.now()
                    return True
                sys.stdout.write('Please respond with: Yes, No, nEver, Always.\n')
                sys.stdout.write('nEver and Always will be stickied for {} minutes'.format(int(rc.overwrite.timeout / 60.))+'\n')

