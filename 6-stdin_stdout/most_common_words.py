#!~/tools/anaconda3/bin/env python
""" import sys, re

all_args = sys.argv

for line in sys.stdin:
    if any([re.search(regex, line) for regex in all_args]):
        sys.stdout.write(line) """


# --------------------------------------

import sys
from collections import Counter

try:
    num_words = int(sys.argv[1])
except:
    sys.stderr.write("error\nusage: python most_common_words.py num_words\n")
    sys.exit(-1)

counter = Counter(word.lower()
    for line in sys.stdin
    for word in line.strip().split()
    if word
    )

for word, count in counter.most_common(num_words):
    sys.stdout.write(f"{str(count)}\t {word}\n")

