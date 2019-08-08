# -*- coding: utf-8 -*-
import codecs
import os
import sys

from tqdm import tqdm
from collections import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

sys.path.append("..")
from Model.hparams import HParams
from settings import SYSTEM_ROOT, VOCAB_FILE, EMOTION_FILE, SPLIT_LIST, RECORD_DIR, RECORD_FILE_NAME_LIST


#RECORD_DIR = 'tfrecord'
corpus_dir = os.path.join(SYSTEM_ROOT, 'Corpus/EmotionLines/EmotionPush')
TRAIN_JSON = 'emotionpush_train.json'

class DataLoader(object):
    def __init__(self, hparams, training=True):
        
        self.training = training
        self.hparams = hparams
        
        self.src_max_len = self.hparams.src_max_len
        self.tgt_max_len = self.hparams.tgt_max_len
        
        self.vocab_size, self.vocab_list = check_vocab(VOCAB_FILE)
        self.emotion_size, self.emotion_list = check_vocab(EMOTION_FILE)
        
        self.vocab_table = lookup_ops.index_table_from_file(
                VOCAB_FILE, default_value=self.hparams.unk_id)
        self.reverse_vocab_table = lookup_ops.index_to_string_table_from_file(
                VOCAB_FILE, default_value=self.hparams.unk_token)
        
        self.emotion_table = lookup_ops.index_table_from_file(
                EMOTION_FILE, default_value=self.hparams.unk_id)
        self.reverse_emotion_table = lookup_ops.index_to_string_table_from_file(
                EMOTION_FILE, default_value=self.hparams.unk_token)
        
        if self.training:
            print('--------------------------------------------------')
            for index, name in enumerate(RECORD_FILE_NAME_LIST):
                print('= {} - {}'.format(index, name))
            RECORD_INDEX = int(input("# Input record file index: "))
            print('--------------------------------------------------')
                                     
            batch_lists = self.get_file_batch_lists('{}_train.json'.format(RECORD_FILE_NAME_LIST[RECORD_INDEX]))
            emotion_num_dict = self.get_emotion_num(batch_lists)
            self.emotion_weight_dict = self.get_emotion_weight(emotion_num_dict)
        
            self.case_table = prepare_case_table()
            self.dev_dataset = self.load_record(os.path.join(RECORD_DIR,'{}_dev.tfrecords'.format(RECORD_FILE_NAME_LIST[RECORD_INDEX])))
            self.test_dataset = self.load_record(os.path.join(RECORD_DIR,'{}_test.tfrecords'.format(RECORD_FILE_NAME_LIST[RECORD_INDEX])))
            self.train_dataset = self.load_record(os.path.join(RECORD_DIR,'{}_train.tfrecords'.format(RECORD_FILE_NAME_LIST[RECORD_INDEX])))
        else:
            self.case_table = None
    
    def get_file_batch_lists(self, filename):
        file_list = []
        for path, dirs, files in sorted(os.walk(corpus_dir)):
                if files:
                    for file in files:
                        if file == filename:
                            file_path = os.path.join(path, file)
                            file_list = [file_path, file]
        batch_lists = openfile(file_list)
        return batch_lists
        
    def get_emotion_num(self, batch_lists, tag='emo_predict'):
        emotion_num_dict = dict()
        for batch_list in batch_lists:
            for batch_dict in batch_list:
                emo_tag = '_{}_'.format(batch_dict[tag])
                if emo_tag not in emotion_num_dict:
                    emotion_num_dict[emo_tag] = 1
                else:
                    emotion_num_dict[emo_tag] += 1
        return emotion_num_dict
    
    def get_emotion_weight(self, emotion_num_dict):
        emotion_weight_dict = dict()
        total_sum, total_mul = 0.0, 1.0
        for index in range(1, self.emotion_size):
            total_sum += emotion_num_dict[self.emotion_list[index]]
#            total_mul *= emotion_num_dict[self.emotion_list[index]]
        for index in range(1, self.emotion_size):
#            emotion_weight_dict.update({self.emotion_list[index]: (1.0 / emotion_num_dict[self.emotion_list[index]]) * 100.0})
            emotion_weight_dict.update({self.emotion_list[index]: (total_sum - emotion_num_dict[self.emotion_list[index]]) / total_sum})
        emotion_weight_dict.update({self.emotion_list[0]: 0.0})
        return emotion_weight_dict
        
    def parse_example(self, serial_exmp):
        context, feature_lists = tf.parse_single_sequence_example(
                serialized=serial_exmp,sequence_features={
                    "sen_input":tf.FixedLenSequenceFeature([], dtype=tf.int64),
                    "emo_predict":tf.FixedLenSequenceFeature([], dtype=tf.int64),
                    "sen_output":tf.FixedLenSequenceFeature([], dtype=tf.int64),
                    "emo_output":tf.FixedLenSequenceFeature([], dtype=tf.int64),
                    "sen_input_length":tf.FixedLenSequenceFeature([], dtype=tf.int64),
                    "sen_output_length":tf.FixedLenSequenceFeature([], dtype=tf.int64)})
        sen_input = tf.cast(feature_lists['sen_input'], tf.int32)
        emo_predict = tf.cast(feature_lists['emo_predict'], tf.int32)
        sen_output = tf.cast(feature_lists['sen_output'], tf.int32)
        emo_output = tf.cast(feature_lists['emo_output'], tf.int32)
        sen_input_length = tf.cast(feature_lists['sen_input_length'], tf.int32)
        sen_output_length = tf.cast(feature_lists['sen_output_length'], tf.int32)
        return sen_input, emo_predict, sen_output, emo_output, sen_input_length, sen_output_length
    
    def load_record(self, LOAD_FILE):
        dataset = tf.data.TFRecordDataset(LOAD_FILE)
        dataset = dataset.map(self.parse_example)
        return dataset
    
    def get_training_batch(self, input_dataset, num_threads=4):
        
        buffer_size = self.hparams.batch_size #* 400
            
        dataset = input_dataset.map(lambda si, ep, so, eo, il, ol: \
            (tf.reshape(si, [tf.size(il), -1]), ep,
             tf.reshape(so, [tf.size(ol), -1]), eo,
             il, ol, tf.size(il))).prefetch(buffer_size)
        
        dataset = dataset.map(lambda si, ep, so, eo, il, ol, tl: \
            (tf.concat([si, tf.concat([tf.constant([[self.hparams.non_id, self.hparams.eos_id]]), tf.fill([1, tf.shape(si)[1]-2], self.hparams.non_id)], axis=1)], axis=0),
             tf.concat([ep, tf.constant([self.hparams.non_id])], axis=0),
             tf.concat([so, tf.concat([tf.constant([[self.hparams.non_id, self.hparams.eos_id]]), tf.fill([1, tf.shape(so)[1]-2], self.hparams.non_id)], axis=1)], axis=0),
             tf.concat([eo, tf.constant([self.hparams.non_id])], axis=0),
             tf.concat([il, tf.constant([0])], axis=0), tf.concat([ol, tf.constant([0])], axis=0), tl+1)).prefetch(buffer_size)
        
#        dataset = dataset.map(lambda si, ep, so, eo, il, ol, tl: \
#            (tf.concat([si, tf.concat([tf.constant([[self.hparams.non_id, self.hparams.eos_id]]), tf.fill([1, tf.shape(si)[1]-2], self.hparams.non_id)], axis=1)], axis=0),
#             tf.concat([ep, tf.constant([self.hparams.non_id])], axis=0),
#             tf.concat([so, tf.concat([tf.constant([[self.hparams.ets_id, self.hparams.eos_id]]), tf.fill([1, tf.shape(so)[1]-2], self.hparams.non_id)], axis=1)], axis=0),
#             tf.concat([eo, tf.constant([self.hparams.non_id])], axis=0),
#             tf.concat([il, tf.constant([2])], axis=0), tf.concat([ol, tf.constant([2])], axis=0), tl+1)).prefetch(buffer_size)
        
        shuffle_dataset = dataset.shuffle(buffer_size=buffer_size)
        
        batched_dataset = shuffle_dataset.padded_batch(self.hparams.batch_size,
                    padded_shapes=(tf.TensorShape([None, None]), tf.TensorShape([None]),
                                   tf.TensorShape([None, None]), tf.TensorShape([None]),
                                   tf.TensorShape([None]), tf.TensorShape([None]),
                                   tf.TensorShape([])),
                    padding_values=(self.hparams.non_id, self.hparams.non_id,
                                    self.hparams.non_id, self.hparams.non_id, 
                                    0, 0, 0))
        
        iterator = batched_dataset.make_initializable_iterator()
        sen_input, emo_predict, sen_output, emo_output, \
        sen_input_length, sen_output_length, turn_length = iterator.get_next()
        
        return BatchedInput(iterator=iterator, batched_dataset=batched_dataset, handle=iterator.string_handle(), 
                            initializer=iterator.initializer, 
                            sen_input=sen_input,emo_predict=emo_predict,
                            sen_output=sen_output,emo_output=emo_output,
                            sen_input_length=sen_input_length,sen_output_length=sen_output_length,
                            turn_length=turn_length)
    
    def multiple_batch(self, handler, dataset):
        """Make Iterator switch to change batch input"""
        iterator = tf.data.Iterator.from_string_handle(handler, dataset.output_types, dataset.output_shapes)
        sen_input, emo_predict, sen_output, emo_output, \
        sen_input_length, sen_output_length, turn_length = iterator.get_next()
        return BatchedInput(iterator=None, batched_dataset=None, handle=None,
                            initializer=None,
                            sen_input=sen_input,emo_predict=emo_predict,
                            sen_output=sen_output,emo_output=emo_output,
                            sen_input_length=sen_input_length,sen_output_length=sen_output_length,
                            turn_length=turn_length)
        
    def get_inference_batch(self, src_dataset):
        
        with tf.name_scope("make_inference_batch"):
            dataset = src_dataset.map(lambda src, length: (tf.string_split([src]).values, length))
            if self.hparams.src_max_len_infer:
                dataset = dataset.map(lambda src, length: (src[:self.hparams.src_max_len_infer], length))
            dataset = dataset.map(lambda src, length: (tf.cast(self.vocab_table.lookup(src),tf.int32), length))
            if self.hparams.source_reverse:
                dataset = dataset.map(lambda src, length: (tf.reverse(src, axis=[0]), length))
            dataset = dataset.map(lambda src, length: (tf.concat([src, tf.fill([1], self.hparams.eos_id)], 0), length+1))
            
            batched_dataset = dataset.padded_batch(self.hparams.batch_size_infer,
                           padded_shapes=(tf.TensorShape([None]),tf.TensorShape([])),
                           padding_values=(self.hparams.eos_id,0))
            
            iterator = batched_dataset.make_initializable_iterator()
            sen_input, sen_input_length = iterator.get_next()
            
        return BatchedInput(iterator=iterator, batched_dataset=batched_dataset, handle=iterator.string_handle(),
                            initializer=iterator.initializer,
                            sen_input=sen_input, emo_predict=None,
                            sen_output=sen_input, emo_output=None,
                            sen_input_length=sen_input_length,sen_output_length=sen_input_length,
                            turn_length=tf.constant(1))
        
def process_string(string):
    for split_token in SPLIT_LIST:
        if split_token in string:
            string = string.replace(split_token, " "+split_token)
    tokens = string.strip().split(' ')
    string = " ".join(tokens).lower()
    return string

def openfile(file):
    df = pd.read_json(file[0], orient='records')
    indexs, turns = np.shape(df)
    
    batch_list = []
    for index in range(indexs):
        sentence_list = []
        speaker_list = []
        for turn in range(turns):
            try:
                if df[turn][index] is not None:
                    if df[turn][index]['speaker'] not in speaker_list:
                        speaker_list.append(df[turn][index]['speaker'])
                    sentence_list.append({'speaker':df[turn][index]['speaker'], 'utterance':process_string(df[turn][index]['utterance']),
                                          'emotion':df[turn][index]['emotion'], 'annotation':df[turn][index]['annotation']})
            except:
                pass
        for speaker in speaker_list:
            output_list=[]
            sen_input, emo_predict, sen_output, emo_output = '_non_','_non_','_non_','_non_'
            Q_turn = False
            for index, sentence in enumerate(sentence_list):
                if Q_turn and sentence['speaker'] != speaker:
                    output_list.append(dict({'sen_input':sen_input, 'emo_predict':emo_predict, 'sen_output':sen_output, 'emo_output':emo_output}))
                    sen_input, emo_predict, sen_output, emo_output = '_non_','_non_','_non_','_non_'
                    Q_turn = False
                if not Q_turn and sentence['speaker'] != speaker:
                    sen_input, emo_predict = sentence['utterance'], sentence['emotion']
                    Q_turn = True
                elif sentence['speaker'] == speaker:
                    output_list.append(dict({'sen_input':sen_input, 'emo_predict':emo_predict, 'sen_output':sentence['utterance'], 'emo_output':sentence['emotion']}))
                    sen_input, emo_predict, sen_output, emo_output = '_non_','_non_','_non_','_non_'
                    Q_turn = False
                else:
                    output_list.append(dict({'sen_input':sen_input, 'emo_predict':emo_predict, 'sen_output':sen_output, 'emo_output':emo_output}))
                    sen_input, emo_predict, sen_output, emo_output = '_non_','_non_','_non_','_non_'
                    Q_turn = False
            if Q_turn:
                output_list.append(dict({'sen_input':sen_input, 'emo_predict':emo_predict, 'sen_output':sen_output, 'emo_output':emo_output}))
                sen_input, emo_predict, sen_output, emo_output = '_non_','_non_','_non_','_non_'
                Q_turn = False
                
            batch_list.append(output_list)
    
    return batch_list

def save_dataset(file, batch_id):
    writer = tf.python_io.TFRecordWriter(os.path.join(RECORD_DIR,"{}.tfrecords".format(file)))
    for senin_id, emopd_id, senin_length, \
        senop_id, emoop_id, senop_length in batch_id:
        sen_input = [tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(tag, np.int64)])) for tag in senin_id]
        emo_predict = [tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(tag, np.int64)])) for tag in emopd_id]
        sen_output = [tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(tag, np.int64)])) for tag in senop_id]
        emo_output = [tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(tag, np.int64)])) for tag in emoop_id]
        sen_input_length = [tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(tag, np.int64)])) for tag in senin_length]
        sen_output_length = [tf.train.Feature(int64_list=tf.train.Int64List(value=[np.array(tag, np.int64)])) for tag in senop_length]
        feature_list = tf.train.FeatureLists(feature_list={
                    'sen_input':tf.train.FeatureList(feature=sen_input),
                    'emo_predict':tf.train.FeatureList(feature=emo_predict),
                    'sen_output':tf.train.FeatureList(feature=sen_output),
                    'emo_output':tf.train.FeatureList(feature=emo_output),
                    'sen_input_length':tf.train.FeatureList(feature=sen_input_length),
                    'sen_output_length':tf.train.FeatureList(feature=sen_output_length)})
        example = tf.train.SequenceExample(feature_lists=feature_list)
        writer.write(example.SerializeToString())
    writer.close()
        
def save_vocab(file_name, vocab_list):
    try:
        os.remove(file_name)
    except:
        pass
    with open(file_name, 'a', encoding = 'utf8') as f:
        for vocab in vocab_list:
            f.write("{}\n".format(vocab))
    print("Save {} file.".format(file_name))

functional_tokens_dict = {'_person_name_':['person_','name_'],
                          '_organization_':['organization_'],
                          '_thing_':['show_'],
                          '_time_':['time_','year_','month_','_day_'] ,
                          '_location_':['location_','center_','avenue_'],
                          '_object_':['ranking_','data_'],
                          '_url_':['http','link_','website_','server_']}

def generate_vocab(file_batch_list):
    vocab_list = []
    # Special tokens, with IDs: 0, 1, 2, 3
    special_tokens = ['_non_', '_eos_', '_bos_', '_unk_']
    # Functional tokens
    functional_tokens = ['_my_name_', '_person_name_', '_organization_', '_thing_', '_time_', '_location_', '_object_', '_url_']
    # The word following this punctuation should be capitalized in the prediction output.
    punctuation = ['.', '!', '?']
    # The word following this punctuation should not precede with a space in the prediction output.
    punctuation_space = ['(', ')', '[', ']', '{', '}', '``', '$']

    for t in (special_tokens + functional_tokens + punctuation + punctuation_space):
        vocab_list.append(t)
    
    vocab_dict = dict()
    for batch_list in tqdm(file_batch_list):
        for batch in batch_list:
            sentence_list = []
            for sentence in batch:
                sentence_list.append('_{}_'.format(sentence['emo_predict']))
                sentence_list.append('_{}_'.format(sentence['emo_output']))
            for sentence in batch:
                sentence_list.append(sentence['sen_input'])
                sentence_list.append(sentence['sen_output'])
            for line in sentence_list:
                tokens = line.strip().split(' ')
                for token in tokens:
                    if len(token) and token != ' ':
                        token = token.lower()
                        for functional_token in functional_tokens_dict:
                            replace_tokens = functional_tokens_dict[functional_token]
                            for rp_token in replace_tokens:
                                if token.startswith('{}'.format(rp_token)):
                                    token = functional_token
                        
#                        vocab_list.append(token)
                        if token not in vocab_dict:
                            vocab_dict[token] = 1
                        else: 
                            vocab_dict[token] += 1
                            if vocab_dict[token] >= 2:
                                if t.startswith('.') or t.startswith('-') or t.endswith('..') or t.endswith('-'):
                                    continue
                                vocab_list.append(token)
    save_vocab(VOCAB_FILE, vocab_list)
    return vocab_list
    
def generate_emotion_vocab(file_batch_list):
    vocab_list = []
    # Special tokens, with IDs: 0, 1, 2, 3
    special_tokens = ['__non__']

    for t in special_tokens:
        vocab_list.append(t)
        
    for batch_list in tqdm(file_batch_list):
        for batch in batch_list:
            sentence_list = []
            for sentence in batch:
                sentence_list.append('_{}_'.format(sentence['emo_predict']))
                sentence_list.append('_{}_'.format(sentence['emo_output']))
            for line in sentence_list:
                tokens = line.strip().split(' ')
                for token in tokens:
                    if len(token) and token != ' ':
                        token = token.lower()
                        if token not in vocab_list:
                            vocab_list.append(token)
    save_vocab(EMOTION_FILE, vocab_list)
    return vocab_list

def rebuild_vocab(corpus_file):
    vocab_list = []
    with open(corpus_file, 'r', encoding = 'utf8') as f:
        for line in tqdm(f):
            l = line.strip()
            if not l:
                continue
            tokens = l.strip().split(' ')
            for token in tokens:
                if len(token) and token != ' ':
                    t = token.lower()
                    if t not in vocab_list:
                        vocab_list.append(t)
    save_vocab(corpus_file, vocab_list)
    return vocab_list

def search_id(hparams, batch_list, vocab_list, target, add_eos = False, EM = False):
    batch_id_list = []
    batch_length_list = []
    for batch in tqdm(batch_list):
        
        sentence_id_list = []
        sentence_length_list = []
        max_length = 0
        for sentence in batch:
            tokens = sentence[target].strip().split(' ')
            sen_length = len(tokens) + 1 if add_eos else len(tokens)
            sen_length = 0 if tokens[0] == '_non_' else sen_length
            max_length = sen_length if max_length < sen_length else max_length
            tokens_id = []
            for token in tokens:
                if EM:
                    token = '_{}_'.format(token)
                if token != '':
                    for functional_token in functional_tokens_dict:
                        replace_tokens = functional_tokens_dict[functional_token]
                        for rp_token in replace_tokens:
                            if token.startswith('{}'.format(rp_token)):
                                token = functional_token
                    try:
                        tokens_id.append(vocab_list.index(token))
                    except:
                        # append _nuk_ id
                        tokens_id.append(hparams.unk_id)
            if add_eos:
                tokens_id.append(hparams.eos_id)
            sentence_id_list.append(tokens_id)
            sentence_length_list.append(sen_length)
            
        first = True
        for sentence_id in sentence_id_list:
            if len(sentence_id) < max_length:
                pad_sentence_id = np.append(sentence_id, np.zeros((max_length - len(sentence_id)), np.int64))
            else:
                pad_sentence_id = np.array(sentence_id, np.int64)
            if first:
                first = False
                new_sentence_id_list = np.array(pad_sentence_id, np.int64)
            else:
                new_sentence_id_list = np.append(new_sentence_id_list, pad_sentence_id)
                
        batch_id_list.append(new_sentence_id_list)
        batch_length_list.append(sentence_length_list)
        
    return batch_id_list, batch_length_list

def check_vocab(file_path):
    if tf.gfile.Exists(file_path):
        vocab_list = []
        with codecs.getreader("utf-8")(tf.gfile.GFile(file_path, "rb")) as f:
            for word in f:
                vocab_list.append(word.strip())
    else:
        raise ValueError("The vocab_file does not exist. Please run the vocab_generator.py to create it.")
    return len(vocab_list), vocab_list

def prepare_case_table():
    keys = tf.constant([chr(i) for i in range(32, 127)])

    l1 = [chr(i) for i in range(32, 65)]
    l2 = [chr(i) for i in range(97, 123)]
    l3 = [chr(i) for i in range(91, 127)]
    values = tf.constant(l1 + l2 + l3)
    return tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys, values), ' ')
    
class BatchedInput(namedtuple("BatchedInput",
                              ["iterator","batched_dataset","handle","initializer",
                               "sen_input","emo_predict","sen_output","emo_output",
                               "sen_input_length","sen_output_length","turn_length"])):
    pass

def generate_dataset(hparams, corpus_dir):
    # Generate vocab
    file_list = []
    for path, dirs, files in sorted(os.walk(corpus_dir)):
            if files:
                for file in files:
                    file_path = os.path.join(path, file)
                    if file.lower().endswith('.json'):
                        file_list.append([file_path, file])
    file_batch_list = []
    for file in file_list:
        batch_list = openfile(file)
        file_batch_list.append(batch_list)
    
    vocab_list = generate_vocab(file_batch_list)
    emotion_list = generate_emotion_vocab(file_batch_list)
    vocab_list = rebuild_vocab(VOCAB_FILE)
    
    # Generate tfrecord
    for file, batch_list in zip(file_list, file_batch_list):
        senin_id, senin_length = search_id(hparams, batch_list, vocab_list, 'sen_input', add_eos = True)
        emopd_id, emopd_length = search_id(hparams, batch_list, emotion_list, 'emo_predict', EM = True)
        senop_id, senop_length = search_id(hparams, batch_list, vocab_list, 'sen_output', add_eos = True)
        emoop_id, emoop_length = search_id(hparams, batch_list, emotion_list, 'emo_output', EM = True)
        batch_id = zip(senin_id, emopd_id, senin_length,
                       senop_id, emoop_id, senop_length)
        save_dataset(file[1][:-5], batch_id)

if __name__ == "__main__":
    hparams = HParams(SYSTEM_ROOT).hparams
#    loader = DataLoader(hparams, training=True)
#    batch_input = loader.get_training_batch(loader.dev_dataset)
#    
#    file_list = []
#    for path, dirs, files in sorted(os.walk(corpus_dir)):
#            if files:
#                for file in files:
#                    file_path = os.path.join(path, file)
#                    if file.lower().endswith('.json'):
#                        file_list.append([file_path, file])
#    file_batch_list = []
#    for file in file_list:
#        batch_list = openfile(file)
#        file_batch_list.append(batch_list)
#    
#    file_emotion_predict_num = []
#    for index, batch_lists in enumerate(file_batch_list):
#        emotion_predict_num = dict()
#        for batch_list in batch_lists:
#            for batch_dict in batch_list:
#                if batch_dict['emo_predict'] not in emotion_predict_num:
#                    emotion_predict_num[batch_dict['emo_predict']] = 1
#                else:
#                    emotion_predict_num[batch_dict['emo_predict']] += 1
#        file_emotion_predict_num.append({file_list[index][1]: emotion_predict_num})
#    
#    file_emotion_output_num = []
#    for index, batch_lists in enumerate(file_batch_list):
#        emotion_output_num = dict()
#        for batch_list in batch_lists:
#            for batch_dict in batch_list:
#                if batch_dict['emo_output'] not in emotion_output_num:
#                    emotion_output_num[batch_dict['emo_output']] = 1
#                else:
#                    emotion_output_num[batch_dict['emo_output']] += 1
#        file_emotion_output_num.append({file_list[index][1]: emotion_output_num})
    
    generate_dataset(hparams, corpus_dir)
    
    # Load tfrecord
    loader = DataLoader(hparams, training=True)
    batch_input = loader.get_training_batch(loader.dev_dataset)
    
    initializer = tf.random_uniform_initializer(-0.1, 0.1, seed= 0.1)
    tf.get_variable_scope().set_initializer(initializer)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        
        sess.run(batch_input.initializer)
        output = sess.run([batch_input.sen_input, batch_input.emo_predict, batch_input.sen_output, batch_input.emo_output,
                           batch_input.sen_input_length, batch_input.sen_output_length, batch_input.turn_length])