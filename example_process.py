"""
Example process.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
import time

i=0
while i<3:
    i += 1
    time.sleep(1)
    print('hello world %d'%(i))
    sys.stdout.flush()