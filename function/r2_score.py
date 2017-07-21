#!/usr/bin/python
#################################################################
#
#    file: r2_score.py
#   usage: ./r2_score.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-21 16:40:15
#
#################################################################
from sklearn.metrics import r2_score

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
print r2_score(y_true, y_pred,multioutput='variance_weighted')
