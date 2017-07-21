#!/usr/bin/python
#################################################################
#
#    file: make_regression.py
#   usage: ./make_regression.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-21 16:29:32
#
#################################################################
from sklearn.datasets import make_regression

x,y = make_regression(10,10,1)
print x
print y
