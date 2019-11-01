#!/usr/bin/env python

import sys
import json
import codecs

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: {} <json1> <json2> <char_vocab> ...".format(sys.argv[0]))

    char_to_keep = set()
    for json_filename in sys.argv[1:3]:
        with open(json_filename) as json_file:
            for line in json_file.readlines():
                for sentence in json.loads(line)["sentences"]:
                    sentence = [w.encode('utf8') for w in sentence]
                    for w in sentence:
                        for char in w.decode('utf8'):
                            char_to_keep.add(char)

    if len(sys.argv) > 4:
        for dep_filename in sys.argv[4:]:
            with open(dep_filename) as dep_file:
                for line in dep_file.readlines():
                    if line == '\n' or line == '\r\n':
                        continue
                    words = line.split()[1]
                    for char in words.decode('utf8'):
                        char_to_keep.add(char)

    with codecs.open(sys.argv[3], 'w', encoding="utf8") as char_vocab:
        for char in char_to_keep:
            char_vocab.write(char + '\n')
