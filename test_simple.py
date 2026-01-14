print("Hello from test_simple.py")
import sys
print(sys.version)
try:
    import numpy
    print("Numpy OK")
except:
    print("Numpy fail")
