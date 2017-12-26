#!/usr/bin/python
# -*- coding: utf-8 -*
#################################################################
#
#    file: generate_captcha.py
#   usage: ./generate_captcha.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-31 16:08:46
#
#################################################################
from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import random
import string

class generateCaptcha():
    def __init__(self,
                 width = 160,
                 height = 60,
                 char_num = 4,
                 characters = string.digits + string.ascii_uppercase + string.ascii_lowercase):
        self.width = width
        self.height = height
        self.char_num = char_num
        self.characters = characters
        self.classes = len(characters)

    def gen_captcha(self,batch_size = 50):
        X = np.zeros([batch_size,self.height,self.width,1])
        img = np.zeros((self.height,self.width),dtype=np.uint8)
        #Y = [np.zeros((batch_size,self.classes),dtype = np.uint8) for i in range(self.char_num)]
        Y = np.zeros([batch_size,self.char_num,self.classes])
        image = ImageCaptcha(width = self.width,height = self.height)

        while True:
            for i in range(batch_size):
                captcha_str = ''.join(random.sample(self.characters,self.char_num))
                img = image.generate_image(captcha_str).convert('L')
                img = np.array(img.getdata())
                X[i] = np.reshape(img,[self.height,self.width,1])/255.0
                for j,ch in enumerate(captcha_str):
                    #Y[j][i,self.characters.find(ch)] = 1
                    Y[i,j,self.characters.find(ch)] = 1
            Y = np.reshape(Y,(batch_size,self.char_num*self.classes))
            yield X,Y

    def decode_captcha(self,y):
        y = np.reshape(y,(len(y),self.char_num,self.classes))
        return ''.join(self.characters[x] for x in np.argmax(y,axis = 2)[0,:])

    def get_parameter(self):
        return self.width,self.height,self.char_num,self.characters,self.classes

    def gen_test_captcha(self):
        image = ImageCaptcha(width = self.width,height = self.height)
        captcha_str = ''.join(random.sample(self.characters,self.char_num))
        img = image.generate_image(captcha_str)
        img.save(captcha_str + '.jpg')

        X = np.zeros([1,self.height,self.width,1])
        Y = np.zeros([1,self.char_num,self.classes])
        img = img.convert('L')
        img = np.array(img.getdata())
        X[0] = np.reshape(img,[self.height,self.width,1])/255.0
        for j,ch in enumerate(captcha_str):
            Y[0,j,self.characters.find(ch)] = 1
        Y = np.reshape(Y,(1,self.char_num*self.classes))
        return X,Y


g = generateCaptcha()
g.gen_test_captcha()

'''
g = generateCaptcha()
X,Y = next(g.gen_captcha(1))
print np.shape(X)
img = Image.fromarray(X[0])
img.save('1.png')
print g.decode_captcha(Y)
'''
