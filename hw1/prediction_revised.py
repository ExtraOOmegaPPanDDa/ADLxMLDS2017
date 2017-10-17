# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:42:41 2017

@author: HSIN
"""

import csv


def most_common(lst):
    return max(set(lst), key=lst.count)

f = open('prediction.csv','r')

ids = []
seqs = []
seq_s = []
seq_e = []
for row in csv.reader(f):
    ids.append(row[0])
    seqs.append(row[1])
    seq_s.append(row[1][0])
    seq_e.append(row[1][-1])
f.close()


most_common_s = most_common(seq_s)
most_common_e = most_common(seq_e)

if seq_s.count(most_common_s)/len(seq_s) > 0.5:
    print('remove Head')
    print(seq_s.count(most_common_s)/len(seq_s))
    
    for i in range(len(seqs)):
        if seqs[i][0] == most_common_s and i > 0:
            seqs[i] = seqs[i][1:]


if seq_e.count(most_common_e)/len(seq_e) > 0.5:
    print('remove Tail')
    print(seq_e.count(most_common_e)/len(seq_e))
    
    for i in range(len(seqs)):
        if seqs[i][-1] == most_common_e and i > 0:
            seqs[i] = seqs[i][:-1]


result = []
for i in range(len(seqs)):
    data = []
    data.append(ids[i])
    data.append(seqs[i])

    result.append(data)


f = open('prediction_revised.csv', 'w', newline='')
w = csv.writer(f)
w.writerows(result)
f.close()

    