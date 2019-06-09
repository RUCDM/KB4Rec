#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

import sys
import gzip


def check(mid, mids):
    if mid in mids:
        return 1
    else:
        return 0


def getSub(mids, filename2, filename3):
    f = gzip.GzipFile(filename2, "r")
    f2 = open(filename3, 'w')
    idx = 0
    idx2 = 0
    for line in f:
        idx += 1
        if idx % 10000000 == 0:
            print
            idx, idx2
        line = line.strip()
        if not "<http://rdf.freebase.com/ns/" in line:
            continue
        line2 = line.split("\t")
        if check(line2[0], mids) or check(line2[2], mids):
            f2.write(line + "\n")
            idx2 += 1
    f2.close()
    f.close()


def getEn(filename):
    f = open(filename, 'r')
    mids = set()
    for line in f:
        line = line.strip().split()
        mids.add("<http://rdf.freebase.com/ns/" + line[0] + ">")
    f.close()
    print
    "en: ", len(mids)
    return mids


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print
        "usage: python getSub3.py enfile freebase subbase"
        sys.exit(-1)
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    filename3 = sys.argv[3]
    mids = getEn(filename1)
    getSub(mids, filename2, filename3)