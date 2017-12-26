#_*_ coding:utf-8 _*_
#################################################################
#
#    file: arg_scope.py
#   usage: ./arg_scope.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-12-26 15:48:00
#
#################################################################
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.framework.python.ops.arg_scope import add_arg_scope

'''
给show,play准备了默认参数
要使arg_scope正常运行起来, 需要两个步骤:
    用@add_arg_scope修饰目标函数
    用with arg_scope(...) 设置默认参数
'''
@add_arg_scope
def show(a,b):
    print(a,b)

@add_arg_scope
def play(c,d):
    print(c,d)

def all_arg_scope():
    with arg_scope([show],a='a'):
        with arg_scope([play],c='c') as arg_sc:
            return arg_sc
def test():
    with arg_scope([show,play]):
        with arg_scope([show],a='a1'):
            show(b='b')
            play(d='d')

with arg_scope(all_arg_scope()):
    test()
