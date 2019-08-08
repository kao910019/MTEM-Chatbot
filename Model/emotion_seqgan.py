# -*- coding: utf-8 -*-
import os
import sys
import time

import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import control_flow_ops

from Model.hparams import HParams
from Model.generator import Generator
from Model.data_process import DataLoader

#from hparams import HParams
#from generator import Generator
#from data_process import DataLoader

#Load settings file
sys.path.append("..")
from settings import SYSTEM_ROOT
from settings import RESULT_DIR, RESULT_FILE, TRAIN_LOG_DIR, DEV_LOG_DIR, TEST_LOG_DIR, INFER_LOG_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
train_sequence = ['pre-train','dis-train','gan-train']

np.set_printoptions(threshold=np.inf)

class EMSeqGAN(object):
    
    def __init__(self, session, training, hparams = None):
        self.session = session
        self.training = training
        print("# Prepare dataset placeholder and hyper parameters ...")
        #Load Hyper-parameters file
        if hparams is None:
            self.hparams = HParams(SYSTEM_ROOT).hparams
        else:
            self.hparams = hparams
            
            
        # Initializer
        initializer = self.get_initializer(self.hparams.init_op, self.hparams.random_seed, self.hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)
        
        self.data_loader = DataLoader(hparams = self.hparams, training = self.training)
        
        self.last_latent_state = None
        self.last_emotion_state = None
        
        if self.training:
            self.build_train_model()
            #tensorboard
            self.train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR+self.train_mode, self.session.graph)
            self.dev_summary_writer = tf.summary.FileWriter(DEV_LOG_DIR+self.train_mode) if self.hparams.dev_dataset else None
            self.test_summary_writer = tf.summary.FileWriter(TEST_LOG_DIR+self.train_mode) if self.hparams.test_dataset else None
        else:
            self.build_predict_model()
            #Tensorboard
            tf.summary.FileWriter(INFER_LOG_DIR, self.session.graph)
            
    def get_initializer(self, init_op, seed=None, init_weight=None):
        """Create an initializer. init_weight is only for uniform."""
        if init_op == "uniform":
            assert init_weight
            return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
        elif init_op == "glorot_normal":
            return tf.contrib.keras.initializers.glorot_normal(seed=seed)
        elif init_op == "glorot_uniform":
            return tf.contrib.keras.initializers.glorot_uniform(seed=seed)
        else:
            raise ValueError("Unknown init_op %s" % init_op)
# =============================================================================
# Create model
# =============================================================================
                
    def build_model(self, batch_input, mode = 'inference'):
        print("# Build model =",mode)
        with tf.variable_scope('Emotion_SeqGAN'):
            
            with tf.variable_scope("vocab_embeddings", dtype = tf.float32):
                self.embedding = tf.get_variable("embedding", [self.data_loader.vocab_size, self.hparams.embedding_size], tf.float32)
            with tf.variable_scope("emotion_embeddings", dtype = tf.float32):
                self.emotion_embedding = tf.get_variable("embedding", [self.data_loader.emotion_size, self.hparams.emotion_embedding_size], tf.float32)
            
            model = Generator(mode = mode,
                              hparams = self.hparams,
                              data_loader = self.data_loader,
                              embedding = (self.embedding, self.emotion_embedding),
                              batch_input = batch_input)
            global_step = model.total_global_step
        return model, global_step
    
    def get_tensors_in_checkpoint_file(self, file_name, all_tensors=True, tensor_name=None):
        varlist, var_value = [], []
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
          var_to_shape_map = reader.get_variable_to_shape_map()
          for key in sorted(var_to_shape_map):
            varlist.append(key)
            var_value.append(reader.get_tensor(key))
        else:
            varlist.append(tensor_name)
            var_value.append(reader.get_tensor(tensor_name))
        return (varlist, var_value)

    def build_tensors_in_checkpoint_file(self, loaded_tensors):
        full_var_list = []
        for i, tensor_name in enumerate(loaded_tensors[0]):
            try:
                tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
                full_var_list.append(tensor_aux)
            except:
                print('* Not found: '+tensor_name)
        return full_var_list
    
    def variable_loader(self):
        self.ckpt = tf.train.get_checkpoint_state(RESULT_DIR)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            print("# Find checkpoint file:", self.ckpt.model_checkpoint_path)
            restored_vars  = self.get_tensors_in_checkpoint_file(file_name = self.ckpt.model_checkpoint_path)
            tensors_to_load = self.build_tensors_in_checkpoint_file(restored_vars)
            self.saver = tf.train.Saver(tensors_to_load, max_to_keep=4, keep_checkpoint_every_n_hours=1.0)
            if self.training == True:
                if input("# Keep training? [y/n]: ") not in ["no","n"]:
                    print("# Restoring model weights ...")
                    self.saver.restore(self.session, self.ckpt.model_checkpoint_path)
                    return True
            else:
                self.saver.restore(self.session, self.ckpt.model_checkpoint_path)
                return True
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=4, keep_checkpoint_every_n_hours=1.0)
        return False
        
    def build_train_model(self):
        
        with self.session.graph.as_default():
            
            train_mode = input("# Select train mode [gen/dis/adv]: ")
            if train_mode in ['gen-train','gen','g']:
                self.train_mode = 'gen-train'
            elif train_mode in ['dis-train','dis','d']:
                self.train_mode = 'dis-train'
            elif train_mode in ['adv-train','adv','a']:
                self.train_mode = 'adv-train'
                
            with tf.variable_scope('Network_Operator'):
                self.loss_log_var = tf.Variable(0.0, name = "loss_log_var", trainable=False)
                self.specific_emotion_predict_accuracy_log_var, self.specific_emotion_output_accuracy_log_var, self.specific_emotion_output_predict_accuracy_log_var = [], [], []
                for index in range(self.data_loader.emotion_size):
                    self.specific_emotion_predict_accuracy_log_var.append(tf.Variable(0.0, name = "{}_emotion_predict_accuracy_log_var".format(self.data_loader.emotion_list[index]), trainable=False))
                    self.specific_emotion_output_accuracy_log_var.append(tf.Variable(0.0, name = "{}_emotion_output_accuracy_log_var".format(self.data_loader.emotion_list[index]), trainable=False))
                    self.specific_emotion_output_predict_accuracy_log_var.append(tf.Variable(0.0, name = "{}_emotion_output_predict_accuracy_log_var".format(self.data_loader.emotion_list[index]), trainable=False))
                self.response_flag_accuracy_log_var = tf.Variable(0.0, name = "response_flag_accuracy_log_var", trainable=False)
                self.emotion_predict_accuracy_log_var = tf.Variable(0.0, name = "emotion_predict_accuracy_log_var", trainable=False)
                self.emotion_output_accuracy_log_var = tf.Variable(0.0, name = "emotion_output_accuracy_log_var", trainable=False)
                self.decoder_perplexity_log_var = tf.Variable(0.0, name = "decoder_perplexity_log_var", trainable=False)
                self.discriminator_accuracy_log_var = tf.Variable(0.0, name = "discriminator_accuracy_log_var", trainable=False)
                self.emotion_output_predict_accuracy_log_var = tf.Variable(0.0, name = "emotion_output_predict_accuracy_log_var", trainable=False)
                self.teacher_forcing_perplexity_log_var = tf.Variable(0.0, name = "teacher_forcing_perplexity_log_var", trainable=False)
                self.teacher_forcing_accuracy_log_var = tf.Variable(0.0, name = "teacher_forcing_accuracy_log_var", trainable=False)
                self.adversarial_discriminate_log_var = tf.Variable(0.0, name = "adversarial_discriminate_log_var", trainable=False)
                
                if self.train_mode == 'gen-train':
                    tf.summary.merge([tf.summary.scalar(self.train_mode+"_loss", self.loss_log_var),
                                      tf.summary.scalar(self.train_mode+"_response_flag_accuracy", self.response_flag_accuracy_log_var),
                                      tf.summary.scalar(self.train_mode+"_emotion_predict_accuracy", self.emotion_predict_accuracy_log_var),
                                      tf.summary.scalar(self.train_mode+"_emotion_output_accuracy", self.emotion_output_accuracy_log_var),
                                      tf.summary.scalar(self.train_mode+"_decoder_perplexity", self.decoder_perplexity_log_var)] \
                                      + [tf.summary.scalar(self.train_mode+"_{}_emotion_predict_accuracy".format(self.data_loader.emotion_list[index]), self.specific_emotion_predict_accuracy_log_var[index]) for index in range(self.data_loader.emotion_size)] \
                                      + [tf.summary.scalar(self.train_mode+"_{}_emotion_output_accuracy".format(self.data_loader.emotion_list[index]), self.specific_emotion_output_accuracy_log_var[index]) for index in range(self.data_loader.emotion_size)])
                elif self.train_mode == 'dis-train':
                    tf.summary.merge([tf.summary.scalar(self.train_mode+"_loss", self.loss_log_var),
                                      tf.summary.scalar(self.train_mode+"_discriminator_accuracy", self.discriminator_accuracy_log_var)])
#                elif self.train_mode == 'adv-train':
#                    tf.summary.merge([tf.summary.scalar(self.train_mode+"_loss", self.loss_log_var),
#                                      tf.summary.scalar(self.train_mode+"_teacher_forcing_perplexity", self.teacher_forcing_perplexity_log_var),
#                                      tf.summary.scalar(self.train_mode+"_teacher_forcing_accuracy", self.teacher_forcing_accuracy_log_var),
#                                      tf.summary.scalar(self.train_mode+"_adversarial_discriminate", self.adversarial_discriminate_log_var)])
                elif self.train_mode == 'adv-train':
                    tf.summary.merge([tf.summary.scalar(self.train_mode+"_loss", self.loss_log_var),
                                      tf.summary.scalar(self.train_mode+"_response_flag_accuracy", self.response_flag_accuracy_log_var),
                                      tf.summary.scalar(self.train_mode+"_emotion_predict_accuracy", self.emotion_predict_accuracy_log_var),
                                      tf.summary.scalar(self.train_mode+"_emotion_output_accuracy", self.emotion_output_accuracy_log_var),
                                      tf.summary.scalar(self.train_mode+"_decoder_perplexity", self.decoder_perplexity_log_var),
                                      tf.summary.scalar(self.train_mode+"_discriminator_accuracy", self.discriminator_accuracy_log_var),
                                      tf.summary.scalar(self.train_mode+"_adversarial_discriminate", self.adversarial_discriminate_log_var)] \
                                      + [tf.summary.scalar(self.train_mode+"_{}_emotion_predict_accuracy".format(self.data_loader.emotion_list[index]), self.specific_emotion_predict_accuracy_log_var[index]) for index in range(self.data_loader.emotion_size)] \
                                      + [tf.summary.scalar(self.train_mode+"_{}_emotion_output_accuracy".format(self.data_loader.emotion_list[index]), self.specific_emotion_output_accuracy_log_var[index]) for index in range(self.data_loader.emotion_size)])
                        
                self.write_loss_op = tf.summary.merge_all()
                self.dataset_handler = tf.placeholder(tf.string, shape=[], name='dataset_handler')
                self.train_batch_iter = self.data_loader.get_training_batch(self.data_loader.train_dataset)
                self.dev_batch_iter = self.data_loader.get_training_batch(self.data_loader.dev_dataset) if self.hparams.dev_dataset else None
                self.test_batch_iter = self.data_loader.get_training_batch(self.data_loader.test_dataset) if self.hparams.test_dataset else None
                input_batch = self.data_loader.multiple_batch(self.dataset_handler, self.train_batch_iter.batched_dataset)
                
            self.model, self.global_step = self.build_model(batch_input = input_batch, mode = self.train_mode)
            
            if self.train_mode == 'gen-train':
                self.epoch_num = self.model.generator_train_epoch
                self.learning_rate = self.model.generator_learning_rate
                self.last_loss = self.model.generator_last_loss
            elif self.train_mode == 'dis-train':
                self.epoch_num = self.model.discriminator_train_epoch
                self.learning_rate = self.model.discriminator_learning_rate
                self.last_loss = self.model.discriminator_last_loss
            elif self.train_mode == 'adv-train':
                self.epoch_num = self.model.adversarial_train_epoch
                self.learning_rate = self.model.adversarial_learning_rate
                self.last_loss = self.model.adversarial_last_loss
        
#            self.restore = self.variable_loader()
            
    def build_predict_model(self):
        self.src_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name = 'NN_input')
        self.src_length_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name = 'NN_input_length')
        src_dataset = tf.data.Dataset.from_tensor_slices((self.src_placeholder, self.src_length_placeholder))
        self.infer_batch = self.data_loader.get_inference_batch(src_dataset)
        print("# Creating inference model ...")
        self.model, _ = self.build_model(self.infer_batch, mode = 'inference')
        print("# Restoring model weights ...")
        self.restore = self.variable_loader()
        assert self.restore
        self.session.run(tf.tables_initializer())
# =============================================================================
# Create model
# =============================================================================
    def train(self):
        
        hp_epoch_times = self.hparams.num_epochs
        
        print("# Get dataset handler.")
        training_handle = self.session.run(self.train_batch_iter.handle)
        deving_handle = self.session.run(self.dev_batch_iter.handle)  if self.hparams.dev_dataset else None
        testing_handle = self.session.run(self.test_batch_iter.handle) if self.hparams.test_dataset else None
        dataset_dict = [{'tag':'Train', 'last_loss':100000.0, 'loss':1000.0,
#                         'writer':self.train_summary_writer, 'handler':{self.dataset_handler: training_handle}, 'initer':self.train_batch_iter.initializer}]
                         'writer':self.train_summary_writer, 'handler':{self.dataset_handler: testing_handle}, 'initer':self.test_batch_iter.initializer}]
        if self.hparams.dev_dataset:
            dataset_dict.append({'tag':'Dev', 'last_loss':100000.0, 'loss':1000.0,
                         'writer':self.dev_summary_writer, 'handler':{self.dataset_handler: deving_handle}, 'initer':self.dev_batch_iter.initializer})
        if self.hparams.test_dataset:
            dataset_dict.append({'tag':'Test', 'last_loss':100000.0, 'loss':1000.0,
                         'writer':self.test_summary_writer, 'handler':{self.dataset_handler: testing_handle}, 'initer':self.test_batch_iter.initializer})
        init_list = [self.train_batch_iter.initializer]
        if self.hparams.dev_dataset:
            init_list.append(self.dev_batch_iter.initializer)
        if self.hparams.test_dataset:
            init_list.append(self.test_batch_iter.initializer)
        self.session.run(init_list)
        self.session.run(tf.tables_initializer())
        train_step, first_step = len(dataset_dict), True
        self.train_D = self.hparams.discriminator_training_first
        
        self.session.run(tf.global_variables_initializer(), feed_dict = dataset_dict[0]['handler'])
        self.restore = self.variable_loader()
        
        # Initialize the statistic variables
        train_epoch_times = 0
        global_step = self.global_step.eval(session=self.session)
        epoch_num = self.epoch_num.eval(session=self.session)
        learning_rate = self.learning_rate.eval(session=self.session)
        dataset_dict[0]['last_loss'] = self.last_loss.eval(session=self.session)
        print("# Global step = {}".format(global_step))
        print("="*50)
        print("# Training loop started @ {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        print("# Epoch training {} times.".format(hp_epoch_times))
        
        while train_epoch_times < hp_epoch_times:
            
            for index in range(train_step):
                epoch_start_time = time.time()
                
                if self.train_mode == 'gen-train':
                    ckpt_dict = {'response_flag_accuracy_correct':0.0, 'emotion_predict_accuracy_correct':0.0, 'emotion_output_accuracy_correct':0.0,
                                 'generator_target_count':0.0, 'decoder_perplexity_sum':0.0}
                    for i in range(self.data_loader.emotion_size):
                        ckpt_dict.update({'{}_specific_emotion_predict_count'.format(self.data_loader.emotion_list[i]):0.0})
                        ckpt_dict.update({'{}_specific_emotion_predict_correct'.format(self.data_loader.emotion_list[i]):0.0})
                        ckpt_dict.update({'{}_specific_emotion_output_count'.format(self.data_loader.emotion_list[i]):0.0})
                        ckpt_dict.update({'{}_specific_emotion_output_correct'.format(self.data_loader.emotion_list[i]):0.0})
                elif self.train_mode == 'dis-train':
                    ckpt_dict = {'discriminator_accuracy_correct':0.0}
                elif self.train_mode == 'adv-train':
#                    ckpt_dict = {'adversarial_target_count':0.0, 'adversarial_discriminate_sum': 0.0}
#                    ckpt_dict = {'teacher_forcing_target_count':0.0, 'teacher_forcing_accuracy_correct': 0.0, 'teacher_forcing_perplexity_sum':0.0, 
#                                 'adversarial_target_count':0.0, 'adversarial_discriminate_sum': 0.0}
                    
                    ckpt_dict = {'generator_batch_turn_size':0.0,'response_flag_accuracy_correct':0.0, 'emotion_predict_accuracy_correct':0.0, 'emotion_output_accuracy_correct':0.0,
                                 'generator_target_count':0.0, 'decoder_perplexity_sum':0.0, 'discriminator_batch_turn_size':0.0, 'discriminator_accuracy_correct':0.0, 'adversarial_target_count':0.0, 'adversarial_discriminate_sum': 0.0}
                    for i in range(self.data_loader.emotion_size):
                        ckpt_dict.update({'{}_specific_emotion_predict_count'.format(self.data_loader.emotion_list[i]):0.0})
                        ckpt_dict.update({'{}_specific_emotion_predict_correct'.format(self.data_loader.emotion_list[i]):0.0})
                        ckpt_dict.update({'{}_specific_emotion_output_count'.format(self.data_loader.emotion_list[i]):0.0})
                        ckpt_dict.update({'{}_specific_emotion_output_correct'.format(self.data_loader.emotion_list[i]):0.0})
                        
                ckpt_loss, ckpt_count, ckpt_bt_count = 0.0, 0.0, 0.0
                
                print("# Start {} step.".format(dataset_dict[index]['tag']))
                self.session.run(dataset_dict[index]['initer'])
                dataset_handler = dataset_dict[index]['handler']
                
                while True:
                    try:
                        if dataset_dict[index]['tag'] == 'Train':
                            step_loss, batch_size, batch_turn_size, result_dict, _, global_step = self.update(self.train_mode, dataset_handler)
                        else:
                            step_loss, batch_size, batch_turn_size, result_dict = self.test(self.train_mode, dataset_handler)
                            
                        if dataset_dict[index]['tag'] == 'Train' and ((global_step % 100 == 0) or first_step):
                            print("Step:", global_step,"loss =", step_loss)
                            first_step = False
                        
                        ckpt_loss, ckpt_count, ckpt_bt_count, ckpt_dict = self.calculate_ckpt(
                                ckpt_loss, ckpt_count, ckpt_bt_count, ckpt_dict,
                                step_loss, batch_size, batch_turn_size, result_dict)
                        
                    except tf.errors.OutOfRangeError:
                        epoch_dur = time.time() - epoch_start_time
                        dataset_dict[index]['loss'] = ckpt_loss / ckpt_count
                        
                        if dataset_dict[index]['tag'] == 'Train':
                            learning_rate = self.get_learning_rate(dataset_dict[index]['loss'], dataset_dict[index]['last_loss'])
                            self.session.run(tf.assign(self.learning_rate, learning_rate))
                            self.session.run(tf.assign(self.last_loss, dataset_dict[index]['loss']))
                        self.write_summary(self.train_mode, self.session, dataset_dict[index]['writer'], epoch_num, dataset_dict[index]['loss'], ckpt_count, ckpt_bt_count, ckpt_dict)
                        if dataset_dict[index]['tag'] == 'Test':
                            train_epoch_times += 1
                            epoch_num = self.session.run(tf.assign_add(self.epoch_num, 1))
                            if dataset_dict[index]['loss'] < dataset_dict[index]['last_loss']:
                                self.saver.save(self.session, RESULT_FILE, global_step = global_step)
                        
                        dataset_dict[index]['last_loss'] = dataset_dict[index]['loss']
                        print("# {} epoch: {}, loss = {}".format(dataset_dict[index]['tag'], epoch_num, dataset_dict[index]['loss']))
                        print("# {} step {:5d} @ {} | {:.2f} seconds elapsed."
                          .format(dataset_dict[index]['tag'], global_step, time.strftime("%Y-%m-%d %H:%M:%S"), round(epoch_dur, 2)))
                        break
            print("# Finished epoch {:2d}/{:2d} @ {} | {:.2f} seconds elapsed."
              .format(train_epoch_times, hp_epoch_times, time.strftime("%Y-%m-%d %H:%M:%S"), round(epoch_dur, 2)))
            
        self.train_summary_writer.close()
        if self.hparams.dev_dataset:
            self.dev_summary_writer.close()
        if self.hparams.test_dataset:
            self.test_summary_writer.close()
    
    def update(self, mode, dataset_handler):
        result = self.model.update(mode, self.session, dataset_handler, self.train_D)
        global_step = self.session.run(self.model.total_global_step)
        step_loss, batch_size, batch_turn_size, result_dict, step_summary, local_step = result
        self.train_summary_writer.add_summary(step_summary, global_step)
        return step_loss, batch_size, batch_turn_size, result_dict, local_step, global_step
    
    def test(self, mode, dataset_handler):
        loss, batch_size, batch_turn_size, result_dict = self.model.test(mode, self.session, dataset_handler)
        return loss, batch_size, batch_turn_size, result_dict
    
    def calculate_ckpt(self, ckpt_loss, ckpt_count, ckpt_bt_count, ckpt_dict, step_loss, batch_size, batch_turn_size, result_dict):
        new_loss = ckpt_loss + (step_loss * batch_size)
        new_count = ckpt_count + batch_size
        new_bt_count = ckpt_bt_count + batch_turn_size
        new_dict = {}
        for dict_name in result_dict:
            new_dict.update({dict_name: ckpt_dict[dict_name] + result_dict[dict_name]})
        return new_loss, new_count, new_bt_count, new_dict
        
    def write_summary(self, mode, session, summary_writer, epoch, epoch_loss, epoch_count, epoch_bt_count, epoch_dict):
        feed_dict = {self.loss_log_var: epoch_loss}
        if mode == 'gen-train':
            feed_dict.update({self.response_flag_accuracy_log_var: epoch_dict['response_flag_accuracy_correct'] / epoch_bt_count, 
                              self.emotion_predict_accuracy_log_var: epoch_dict['emotion_predict_accuracy_correct'] / epoch_bt_count,
                              self.emotion_output_accuracy_log_var: epoch_dict['emotion_output_accuracy_correct'] / epoch_bt_count, 
                              self.decoder_perplexity_log_var: np.exp(epoch_dict['decoder_perplexity_sum'] / epoch_dict['generator_target_count'])})
            for index in range(self.data_loader.emotion_size):
                feed_dict.update({self.specific_emotion_predict_accuracy_log_var[index]: epoch_dict['{}_specific_emotion_predict_correct'.format(self.data_loader.emotion_list[index])] / epoch_dict['{}_specific_emotion_predict_count'.format(self.data_loader.emotion_list[index])]})
                feed_dict.update({self.specific_emotion_output_accuracy_log_var[index]: epoch_dict['{}_specific_emotion_output_correct'.format(self.data_loader.emotion_list[index])] / epoch_dict['{}_specific_emotion_output_count'.format(self.data_loader.emotion_list[index])]})
        elif mode == 'dis-train':
            feed_dict.update({self.discriminator_accuracy_log_var: epoch_dict['discriminator_accuracy_correct'] / epoch_bt_count})
        elif mode == 'adv-train':
#            feed_dict.update({self.adversarial_discriminate_log_var: np.exp(epoch_dict['adversarial_discriminate_sum'] / epoch_dict['adversarial_target_count'])})
#            feed_dict.update({self.teacher_forcing_perplexity_log_var: np.exp(epoch_dict['teacher_forcing_perplexity_sum'] / epoch_dict['teacher_forcing_target_count']),
#                              self.teacher_forcing_accuracy_log_var: epoch_dict['teacher_forcing_accuracy_correct'] / epoch_dict['teacher_forcing_target_count'],
#                              self.adversarial_discriminate_log_var: epoch_dict['adversarial_discriminate_sum'] / epoch_dict['adversarial_target_count']})
            feed_dict.update({self.response_flag_accuracy_log_var: epoch_dict['response_flag_accuracy_correct'] / epoch_dict['generator_batch_turn_size'], 
                              self.emotion_predict_accuracy_log_var: epoch_dict['emotion_predict_accuracy_correct'] / epoch_dict['generator_batch_turn_size'],
                              self.emotion_output_accuracy_log_var: epoch_dict['emotion_output_accuracy_correct'] / epoch_dict['generator_batch_turn_size'], 
                              self.decoder_perplexity_log_var: np.exp(epoch_dict['decoder_perplexity_sum'] / epoch_dict['generator_target_count'])})
            for index in range(self.data_loader.emotion_size):
                feed_dict.update({self.specific_emotion_predict_accuracy_log_var[index]: epoch_dict['{}_specific_emotion_predict_correct'.format(self.data_loader.emotion_list[index])] / epoch_dict['{}_specific_emotion_predict_count'.format(self.data_loader.emotion_list[index])]})
                feed_dict.update({self.specific_emotion_output_accuracy_log_var[index]: epoch_dict['{}_specific_emotion_output_correct'.format(self.data_loader.emotion_list[index])] / epoch_dict['{}_specific_emotion_output_count'.format(self.data_loader.emotion_list[index])]})
            feed_dict.update({self.discriminator_accuracy_log_var: epoch_dict['discriminator_accuracy_correct'] / epoch_dict['discriminator_batch_turn_size'],
                              self.adversarial_discriminate_log_var: epoch_dict['adversarial_discriminate_sum'] / epoch_dict['adversarial_target_count']})
            if (epoch_dict['discriminator_accuracy_correct'] / epoch_dict['discriminator_batch_turn_size']) < 0.8:
                self.train_D = True
            else:
                self.train_D = False
            
        summary = session.run(self.write_loss_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()
        
    def get_learning_rate(self, loss, last_loss):
        default_learning_rate = 8e-4
        percent = np.abs((loss / (loss + last_loss))*100.0)
        if percent > 50.0:
            return default_learning_rate
        return np.abs(default_learning_rate - (((8e-4 - 8e-6)/50.0)*percent))
    
    def generate(self, question, passing_flag):
        tokens = nltk.word_tokenize(question.lower())
        sentence = [' '.join(tokens[:]).strip()]
        length = len(sentence[0].split()) if question != '_non_' else 0
        
        if self.last_latent_state == self.last_emotion_state:
            passing_flag = False
            self.last_latent_state = np.zeros([self.hparams.num_layers, 1, self.hparams.num_units],np.float32) if self.hparams.num_layers > 1 else np.zeros([1, self.hparams.num_units],np.float32)
            self.last_emotion_state = np.zeros([self.hparams.num_layers, 1, self.hparams.num_emotion_units],np.float32) if self.hparams.num_layers > 1 else np.zeros([1, self.hparams.num_emotion_units],np.float32)
        else:
            passing_flag = True
        feed_dict = {self.src_placeholder: sentence,
                     self.src_length_placeholder: np.array([length]),
                     self.model.passing_flag: passing_flag,
                     self.model.latent_state_placeholder: self.last_latent_state,  
                     self.model.emotion_state_placeholder: self.last_emotion_state}
        
        response_flag, sentence_output, emotion_predict, emotion_output, latent_state_output, emotion_state_output = \
        self.model.generate(self.session, feed_dict=feed_dict)
        
        self.last_latent_state = latent_state_output
        self.last_emotion_state = emotion_state_output
        
        if self.hparams.beam_width > 0:
            sentence_output = sentence_output[0]
        eos_token = self.hparams.eos_token.encode("utf-8")
        sentence_output = sentence_output.tolist()[0]
        if eos_token in sentence_output:
            sentence_output = sentence_output[:sentence_output.index(eos_token)]
        sentence_output = b' '.join(sentence_output).decode('utf-8')
        
        response_flag = str(response_flag)
        emotion_predict = str(emotion_predict, encoding = "utf-8")
        emotion_output = str(emotion_output, encoding = "utf-8")
        
        return response_flag, sentence_output, emotion_predict, emotion_output

if __name__ == "__main__":

    with tf.Session() as sess:
        print("# Start")
        model = EMSeqGAN(sess, training = False)
        print("# Generate")
        while True:
            sentence = input("Q: ")
            answer = model.generate(sentence, False)
            print('-'*20)
            print("A:", answer)
            print('-'*20)
    
    