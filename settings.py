# -*- coding: utf-8 -*-
import os
import time

#=====================================================================
SYSTEM_ROOT = os.path.abspath(os.path.dirname(__file__))
SYSTEM_FILE = os.path.join(SYSTEM_ROOT, "main.py")
SYSTEM_STRING = []
SYSTEM_DEBUG = [False]
SYSTEM_CLOSE = [False]
SYSTEM_TIME = time.asctime(time.localtime(time.time()))
SYSTEM_TICK = lambda : time.time()

def Debug_message(message):
    if SYSTEM_DEBUG[0]:
        SYSTEM_STRING.append(message)
    print(message)

#=====================================================================
CORPUS_DIR = os.path.join(SYSTEM_ROOT, 'Corpus')
VOCAB_FILE = os.path.join(CORPUS_DIR, 'vocab.txt')
EMOTION_FILE = os.path.join(CORPUS_DIR, 'emotion_vocab.txt')

RECORD_DIR = os.path.join(CORPUS_DIR, 'tfrecord')

RESULT_DIR = os.path.join(SYSTEM_ROOT, 'Result')
RESULT_FILE = os.path.join(RESULT_DIR, 'basic')

INFER_LOG_DIR = os.path.join(RESULT_DIR, "infer_log")
TRAIN_LOG_DIR = os.path.join(RESULT_DIR, "train_log_")
DEV_LOG_DIR = os.path.join(RESULT_DIR, "dev_log_")
TEST_LOG_DIR = os.path.join(RESULT_DIR, "test_log_")
#=====================================================================
#=====================================================================
NN_NAME = "My_NN"

SPLIT_LIST = ["'m","'s","'t","'ll","'re","~","$",",",".","!","?"]
RECORD_FILE_NAME_LIST = ['emotionpush','friends']
