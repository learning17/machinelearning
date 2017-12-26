#-*- coding:utf-8 -*-
from io import open
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def get_chatbot():
    f = open("xiaohuang.conv","r", encoding="utf-8")
    encode = open("train_encode","w",encoding="utf-8")
    decode = open("train_decode","w",encoding="utf-8")
    vocab = open("vocab","w",encoding="utf-8")
    chat = list()
    words = set()
    for line in f.readlines():
        line = line.strip('\n').strip()
        if not line:
            continue
        if line[0] == "E":
            if chat:
                size = len(chat)
                if size > 1:
                    if size % 2 != 0:
                        chat = chat[:-1]
                    for i in range(len(chat)):
                        if i % 2 == 0:
                            encode.write(chat[i] + "\n")
                        else:
                            decode.write(chat[i] + "\n")
            chat = list()
        elif line[0] == "M":
            L = line.split(' ')
            if len(L) > 1:
                content = L[1]
                content = content.replace(" ","")
                content = content.replace("/"," ")
                chat.append(content)
                content_list = content.split()
                for c in content_list:
                    words.add(c)
    for word in words:
        vocab.write(word + "\n")

get_chatbot()
