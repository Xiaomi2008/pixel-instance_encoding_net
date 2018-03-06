import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
for i in range(10):
   print 'a'
   sys.stdout.flush()

