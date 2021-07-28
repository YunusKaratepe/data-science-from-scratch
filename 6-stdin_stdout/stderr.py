import sys

print("This goes stderr", file=sys.stderr)
sys.stderr.write("This also goes stderr\n")