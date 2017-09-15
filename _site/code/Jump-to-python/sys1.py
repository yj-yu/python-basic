# sys1.py
import sys

args = sys.argv[1:]    # sys1.py aaa bbb ccc: argv[0] = 'sys1.py', argv[1] = 'aaa', ...
for i in args:
    print(i)
    