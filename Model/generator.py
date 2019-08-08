# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class Generator(object):
    """Generator model, base on Seq2Seq."""
    def __init__(self, mode, hparams, data_loader, embedding, batch_input):
        
        self.mode = mode
        if mode == 'inference':
            self.training = False
        else:
            self.training = True
            
        self.hparams = hparams
        self.embedding, self.emotion_embedding = embedding
        
        self.vocab_list = data_loader.vocab_list
        self.vocab_size = data_loader.vocab_size
        self.vocab_table = data_loader.vocab_table
        self.reverse_vocab_table = data_loader.reverse_vocab_table
        self.emotion_list = data_loader.emotion_list
        self.emotion_size = data_loader.emotion_size
        self.emotion_table = data_loader.emotion_table
        self.reverse_emotion_table = data_loader.reverse_emotion_table
        
        self.batch_input = batch_input
        self.batch_size = tf.size(self.batch_input.turn_length)
        self.sample_times = self.hparams.sample_times if self.hparams.reward_type == 'MC_Search' else 1
        self.time_major = self.hparams.time_major
        
        
        with tf.variable_scope('placeholder'):
            self.generator_learning_rate = tf.Variable(8e-4, name = 'generator_learning_rate', trainable=False)
            self.discriminator_learning_rate = tf.Variable(8e-4, name = 'discriminator_learning_rate', trainable=False)
            self.adversarial_learning_rate = tf.Variable(8e-4, name = 'adversarial_learning_rate', trainable=False)
            
            self.generator_train_epoch = tf.Variable(1, name = 'generator_train_epoch', trainable=False)
            self.discriminator_train_epoch = tf.Variable(1, name = 'discriminator_train_epoch', trainable=False)
            self.adversarial_train_epoch = tf.Variable(1, name = 'adversarial_train_epoch', trainable=False)
            
            self.generator_last_loss = tf.Variable(100000.0, name = 'generator_last_loss', trainable=False)
            self.discriminator_last_loss = tf.Variable(100000.0, name = 'discriminator_last_loss', trainable=False)
            self.adversarial_last_loss = tf.Variable(100000.0, name = 'adversarial_last_loss', trainable=False)
            
            self.generator_step = tf.Variable(0, name = 'generator_step', trainable=False)
            self.discriminator_step = tf.Variable(0, name = 'discriminator_step', trainable=False)
            self.adversarial_step = tf.Variable(0, name="adversarial_step", trainable=False)
            
            self.discriminator_accuracy_var = tf.Variable(0.0, name = 'discriminator_accuracy', trainable=False)
            
            self.total_global_step = tf.cast((self.generator_step + self.discriminator_step + self.adversarial_step), tf.int32, name = 'total_global_step')
            
            self.passing_flag = tf.placeholder(dtype=tf.bool, shape=[], name='passing_flag')
            if self.hparams.num_layers > 1:
                self.latent_state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,None,None], name='latent_state')
                self.emotion_state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,None,None], name='emotion_state')
                self.latent_state_input = tuple([tf.squeeze(tf.slice(self.latent_state_placeholder, [index,0,0], [1, -1, -1]), [0]) for index in range(self.hparams.num_layers)])
                self.emotion_state_input = tuple([tf.squeeze(tf.slice(self.emotion_state_placeholder, [index,0,0], [1, -1, -1]), [0]) for index in range(self.hparams.num_layers)])
            else:
                self.latent_state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,None], name='latent_state')
                self.emotion_state_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,None], name='emotion_state')
                self.latent_state_input = self.latent_state_placeholder
                self.emotion_state_input = self.emotion_state_placeholder
                
        # Training or inference graph
        print("# Building graph for the model ...")
              
        if self.mode == 'inference':
            with tf.variable_scope('Multi_turn_generator'):
                self.response_flag_logits, self.sentence_output_logits, self.sentence_output_id, self.output_sample_logits, self.output_sample_id, self.latent_state, self.emotion_state, \
                self.emotion_predict_encoder_logits, self.emotion_predict_decoder_logits, self.emotion_output_logits = self.build_model(
                        self.hparams, self.passing_flag, self.batch_input.sen_input, self.batch_input.sen_output, 
                        self.batch_input.sen_input_length, self.batch_input.sen_output_length, self.latent_state_input, self.emotion_state_input)
            # Generate response to user
            self.response_flag_output = tf.greater(tf.argmax(tf.squeeze(self.response_flag_logits)), 0)
            self.sentnece_output = self.reverse_vocab_table.lookup(tf.to_int64(self.sentence_output_id), name = 'NN_output')
            self.emotion_predict = self.reverse_emotion_table.lookup(tf.to_int64(tf.argmax(tf.squeeze(tf.nn.softmax(self.emotion_predict_encoder_logits)))), name = 'MT_Emotion_predict')
            self.emotion_output = self.reverse_emotion_table.lookup(tf.to_int64(tf.argmax(tf.squeeze(tf.nn.softmax(self.emotion_output_logits)))), name = 'MT_Emotion_outputs')
        else:
            self.emotion_weight_dict = data_loader.emotion_weight_dict
            self.response_flag_logits, self.sentence_output_logits, self.sentence_output_id, self.output_sample_logits, self.output_sample_id, self.latent_state, self.emotion_state, \
            self.emotion_predict_encoder_logits, self.emotion_predict_decoder_logits, self.emotion_output_logits, \
            self.response_length, self.response_ids, self.response_logits, self.reward_labels, self.reward_logits = self.build_multi_turn_model(self.hparams)
            
            self.train(self.hparams)
            
            self.generator_update_list = [self.generator_loss, self.batch_size, self.generator_batch_turn_size, 
                         self.response_flag_accuracy, self.emotion_predict_accuracy, self.emotion_output_accuracy, 
                         self.generator_target_count, self.decoder_perplexity_sum] \
                         + self.specific_emotion_predict_count + self.specific_emotion_predict_correct \
                         + self.specific_emotion_output_count + self.specific_emotion_output_correct \
                         + [self.generator_update, self.generator_summary, self.generator_step]
                         
            self.discriminator_update_list = [self.discriminator_loss, self.batch_size, self.discriminator_batch_turn_size, 
                         self.discriminator_accuracy] \
                         + [self.discriminator_update, self.discriminator_summary, self.discriminator_step]
                         
            if self.mode == 'adv-train':
                self.adversarial_update_list = [self.adversarial_loss, self.batch_size, self.adversarial_batch_turn_size, self.generator_batch_turn_size, 
                             self.response_flag_accuracy, self.emotion_predict_accuracy, self.emotion_output_accuracy, 
                             self.generator_target_count, self.decoder_perplexity_sum, \
                             self.discriminator_batch_turn_size, self.discriminator_accuracy, self.adversarial_target_count, self.adversarial_discriminate] \
                             + self.specific_emotion_predict_count + self.specific_emotion_predict_correct \
                             + self.specific_emotion_output_count + self.specific_emotion_output_correct \
                             + [self.adversarial_update, self.teacher_forcing_update, self.adversarial_summary, self.adversarial_step]
                            
    def generate(self, sess, feed_dict):
        sess.run(self.batch_input.initializer, feed_dict=feed_dict)
        
        response_flag, sentence_output, emotion_predict, emotion_output, latent_state, emotion_state, sen_input, sen_input_length = sess.run([self.response_flag_output,
                self.sentnece_output, self.emotion_predict, self.emotion_output, self.latent_state, self.emotion_state,  self.batch_input.sen_input, self.batch_input.sen_input_length], feed_dict=feed_dict)
        
        print(sen_input)
        print(sen_input_length)
        print('----------------')
        print(sentence_output)
        
        latent_state_output = np.array([latent_state[i] for i in range(self.hparams.num_layers)]) if self.hparams.num_layers > 1 else latent_state
        emotion_state_output = np.array([emotion_state[i] for i in range(self.hparams.num_layers)]) if self.hparams.num_layers > 1 else emotion_state
        
        return response_flag, sentence_output, emotion_predict, emotion_output, latent_state_output, emotion_state_output
    
# =============================================================================
# Training model
# =============================================================================    
    def train(self, hparams):
        """To train generator model."""
        with tf.name_scope("Generator_loss"):
            # labels [batch_size, max_turn]
            sentence_labels = self.batch_input.sen_output
#            sentence_length = tf.cast(tf.argmin(tf.concat([sentence_labels, tf.zeros([self.batch_size, tf.shape(sentence_labels)[1], 1], tf.int32)], axis=2), axis=2), tf.int32)
            response_flag_labels = tf.cast(tf.minimum(self.batch_input.sen_output_length, 1), tf.int32)
            
            emotion_predict_labels = self.batch_input.emo_predict
            emotion_output_labels = self.batch_input.emo_output
            max_turn, max_time = tf.shape(sentence_labels)[1], tf.shape(sentence_labels)[2]
            # logits [batch_size, max_turn, num_unit]
            sentence_logits = self.sentence_output_logits
            response_flag_logits = self.response_flag_logits
            emotion_predict_logits = self.emotion_predict_encoder_logits
            emotion_output_logits = self.emotion_output_logits
            
            # reshape [batch_size * max_turn]
            batch_turn_size = self.batch_size*max_turn
            sentence_labels = tf.reshape(sentence_labels, [batch_turn_size, max_time])
            sentence_logits = tf.reshape(sentence_logits, [batch_turn_size, max_time, self.vocab_size])
            sentnece_output_length = tf.reshape(self.batch_input.sen_output_length, [batch_turn_size])
            
            response_flag_labels = tf.reshape(response_flag_labels, [batch_turn_size])
            response_flag_logits = tf.reshape(response_flag_logits, [batch_turn_size, 2])
            emotion_predict_labels = tf.reshape(emotion_predict_labels, [batch_turn_size])
            emotion_predict_logits = tf.reshape(emotion_predict_logits, [batch_turn_size, self.emotion_size])
            emotion_output_labels = tf.reshape(emotion_output_labels, [batch_turn_size])
            emotion_output_logits = tf.reshape(emotion_output_logits, [batch_turn_size, self.emotion_size])
            #-------------------------------------------------------------------------------------------------
            target_weight = tf.sequence_mask(sentnece_output_length, max_time, dtype=tf.float32)
            target_count = tf.reduce_sum(target_weight)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sentence_labels, logits=sentence_logits)
            self.decoder_loss = tf.reduce_sum(crossent * target_weight) / tf.to_float(batch_turn_size)
            self.decoder_perplexity_sum = tf.reduce_sum(crossent * target_weight)
            self.decoder_perplexity = tf.exp(tf.reduce_sum(crossent * target_weight) / target_count)
            
            self.response_flag_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=response_flag_labels, logits=response_flag_logits))
            self.response_flag_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(response_flag_logits, axis=1, output_type=tf.int32), response_flag_labels), tf.float32))
            
            #-------------------------------------------------------------------------------------------------
            specific_emotion_predict_weight, self.specific_emotion_predict_count, self.specific_emotion_predict_correct, self.specific_emotion_predict_accuracy, self.specific_emotion_predict_loss = [], [], [], [], []
            self.emotion_predict_loss = tf.constant(0.0)
            emotion_predict_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=emotion_predict_labels, logits=emotion_predict_logits)
            #-------------------------------------------------------------------------------------------------
            specific_emotion_output_weight, self.specific_emotion_output_count, self.specific_emotion_output_correct, self.specific_emotion_output_accuracy, self.specific_emotion_output_loss = [], [], [], [], []
            self.emotion_output_loss = tf.constant(0.0)
            emotion_output_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=emotion_output_labels, logits=emotion_output_logits)
            zero_weight = tf.zeros([batch_turn_size], tf.float32)
            for index in range(self.emotion_size):
                #-------------------------------------------------------------------------------------------------
                specific_emotion_predict_weight.append(tf.where(tf.equal(emotion_predict_labels, index), zero_weight+1, zero_weight))
                self.specific_emotion_predict_count.append(tf.reduce_sum(specific_emotion_predict_weight[index]))
                self.specific_emotion_predict_correct.append(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(emotion_predict_logits, axis=1, output_type=tf.int32), emotion_predict_labels), tf.float32) * specific_emotion_predict_weight[index]))
                self.specific_emotion_predict_accuracy.append(self.specific_emotion_predict_correct[index] / tf.math.maximum(self.specific_emotion_predict_count[index], 1))
                self.specific_emotion_predict_loss.append(tf.reduce_mean(emotion_predict_crossent * specific_emotion_predict_weight[index] * self.emotion_weight_dict[self.emotion_list[index]]))
                self.emotion_predict_loss += self.specific_emotion_predict_loss[index]
                #-------------------------------------------------------------------------------------------------
                specific_emotion_output_weight.append(tf.where(tf.equal(emotion_output_labels, index), zero_weight+1, zero_weight))
                self.specific_emotion_output_count.append(tf.reduce_sum(specific_emotion_output_weight[index]))
                self.specific_emotion_output_correct.append(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(emotion_output_logits, axis=1, output_type=tf.int32), emotion_output_labels), tf.float32) * specific_emotion_output_weight[index]))
                self.specific_emotion_output_accuracy.append(self.specific_emotion_output_correct[index] / tf.math.maximum(self.specific_emotion_output_count[index], 1))
                self.specific_emotion_output_loss.append(tf.reduce_mean(emotion_output_crossent * specific_emotion_output_weight[index] * self.emotion_weight_dict[self.emotion_list[index]]))
                self.emotion_output_loss += self.specific_emotion_output_loss[index]
                
            self.emotion_predict_filter = tf.where(tf.equal(specific_emotion_predict_weight[hparams.non_id], 1), zero_weight, zero_weight+1)
            self.emotion_predict_correct = tf.where(tf.equal(tf.argmax(emotion_predict_logits, axis=1, output_type=tf.int32), emotion_predict_labels), zero_weight+1, zero_weight) * self.emotion_predict_filter
            self.emotion_predict_accuracy = tf.to_float(tf.reduce_sum(self.emotion_predict_correct)) / tf.to_float(tf.reduce_sum(self.emotion_predict_filter))
            self.emotion_output_filter = tf.where(tf.equal(specific_emotion_output_weight[hparams.non_id], 1), zero_weight, zero_weight+1)
            self.emotion_output_correct = tf.where(tf.equal(tf.argmax(emotion_output_logits, axis=1, output_type=tf.int32), emotion_output_labels), zero_weight+1, zero_weight) * self.emotion_output_filter
            self.emotion_output_accuracy = tf.to_float(tf.reduce_sum(self.emotion_output_correct)) / tf.to_float(tf.reduce_sum(self.emotion_output_filter))
            
            self.generator_loss = (self.response_flag_loss + self.emotion_predict_loss*10.0 + self.decoder_loss + self.emotion_output_loss*10.0) / 4.0
            self.generator_target_count = target_count
            self.generator_batch_turn_size = batch_turn_size
            
        with tf.name_scope("Discriminator_loss"):
            # labels [batch_size, max_turn]
            labels = self.discriminator_reward_labels
            rebuild_max_time = tf.shape(self.discriminator_response_ids)[1]
#            rebuild_batch_size = tf.shape(self.response_ids)[1]
            # labels [batch_size, max_turn, 2]
            logits = self.discriminator_reward_logits
            #-------------------------------------------------------------------------------------------------
            target_weight = tf.sequence_mask(self.discriminator_response_length, rebuild_max_time, dtype=logits.dtype)
            target_count = tf.reduce_sum(target_weight)
            self.discriminator_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) * target_weight) / target_count
            self.discriminator_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=2, output_type=tf.int32), labels), tf.float32)  * target_weight) / target_count
            self.discriminator_batch_turn_size = target_count
            
        with tf.name_scope("Teacher_forcing_loss"):
            teacher_forcing_id = self.batch_input.sen_output
            teacher_forcing_logits = self.sentence_output_logits
            teacher_forcing_length = self.batch_input.sen_output_length
            max_turn = tf.shape(teacher_forcing_id)[1]
            max_time = tf.shape(teacher_forcing_id)[2]
            batch_turn_size = self.batch_size*max_turn
            teacher_forcing_id = tf.reshape(teacher_forcing_id, [batch_turn_size, max_time])
            teacher_forcing_logits = tf.reshape(teacher_forcing_logits, [batch_turn_size, max_time, self.vocab_size])
            teacher_forcing_length = tf.reshape(teacher_forcing_length, [batch_turn_size])
            
            teacher_forcing_weight = tf.sequence_mask(teacher_forcing_length, max_time, dtype=tf.float32)
            teacher_forcing_count = tf.reduce_sum(teacher_forcing_weight)
            teacher_forcing_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=teacher_forcing_id, logits=teacher_forcing_logits)
            self.teacher_forcing_loss = tf.reduce_sum(teacher_forcing_crossent * teacher_forcing_weight) / tf.to_float(batch_turn_size)
            self.teacher_forcing_perplexity_sum = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=teacher_forcing_id, logits=teacher_forcing_logits) * teacher_forcing_weight)
            self.teacher_forcing_perplexity = tf.exp(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=teacher_forcing_id, logits=teacher_forcing_logits) * teacher_forcing_weight) / tf.to_float(teacher_forcing_count))
            self.teacher_forcing_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(teacher_forcing_logits, axis=2, output_type=tf.int32), teacher_forcing_id), tf.float32)  * teacher_forcing_weight) / tf.to_float(teacher_forcing_count)
            self.teacher_forcing_target_count = teacher_forcing_count
        
        with tf.name_scope("Adversarial_loss"):
            sample_reward = tf.squeeze(tf.slice(tf.nn.softmax(self.sample_reward_logits), [0,0,1], [-1, -1, 1]), [2])
            target_reward = tf.squeeze(tf.slice(tf.nn.softmax(self.target_reward_logits), [0,0,1], [-1, -1, 1]), [2])
            sample_reward = tf.subtract(sample_reward, 0.5)
            target_reward = tf.subtract(target_reward, 0.5)
            sample_rebuild_batch_size = tf.shape(sample_reward)[0]
            target_rebuild_batch_size = tf.shape(target_reward)[0]
            sample_rebuild_max_time = tf.shape(sample_reward)[1]
            target_rebuild_max_time = tf.shape(target_reward)[1]
            
            sample_response_length = self.sample_response_length
            target_response_length = self.target_response_length
            sample_response_ids = self.sample_response_ids
            target_response_ids = self.target_response_ids
            sample_response_logits = self.sample_response_logits
            
            batch_size_diff = target_rebuild_batch_size - sample_rebuild_batch_size
            
            target_response_length = tf.cond(
                    tf.greater(batch_size_diff, 0),lambda: tf.slice(target_response_length, [0], [sample_rebuild_batch_size]), lambda: target_response_length)
            target_reward = tf.cond(
                    tf.greater(batch_size_diff, 0),lambda: tf.slice(target_reward, [0,0], [sample_rebuild_batch_size,-1]), lambda: target_reward)
            target_response_ids = tf.cond(
                    tf.greater(batch_size_diff, 0),lambda: tf.slice(target_response_ids, [0,0], [sample_rebuild_batch_size,-1]), lambda: target_response_ids)
            target_response_logits = tf.cond(
                    tf.less(batch_size_diff, 0), lambda: tf.slice(sample_response_logits, [0,0,0], [target_rebuild_batch_size,-1,-1]), lambda: sample_response_logits)
            
            sample_one_hot = tf.one_hot(sample_response_ids, self.vocab_size)
            target_one_hot = tf.one_hot(target_response_ids, self.vocab_size)
            sample_weight = tf.sequence_mask(sample_response_length, sample_rebuild_max_time, dtype=tf.float32)
            target_weight = tf.sequence_mask(target_response_length, target_rebuild_max_time, dtype=tf.float32)
            sample_count = tf.reduce_sum(sample_weight)
            target_count = tf.reduce_sum(target_weight)
            sample_prob = tf.reduce_sum(tf.clip_by_value(tf.nn.softmax(sample_response_logits), 1e-20, 1) * sample_one_hot, axis = 2)
            target_prob = tf.reduce_sum(tf.clip_by_value(tf.nn.softmax(target_response_logits), 1e-20, 1) * target_one_hot, axis = 2)
            
            self.sample_loss = tf.negative(tf.reduce_sum(tf.log(sample_prob) * sample_reward * sample_weight)) / tf.to_float(sample_rebuild_batch_size)
            self.target_loss = tf.negative(tf.reduce_sum(tf.log(target_prob) * target_reward * target_weight)) / tf.to_float(target_rebuild_batch_size)
            
            self.adversarial_loss = self.sample_loss
            self.adversarial_discriminate = tf.reduce_sum(sample_reward * sample_weight) / tf.to_float(sample_count)
            self.adversarial_batch_turn_size = sample_rebuild_batch_size
            self.adversarial_target_count = sample_count
#            self.adversarial_loss = (self.sample_loss + self.target_loss) / 2.0
            self.adversarial_discriminate = tf.reduce_sum(sample_reward * sample_weight) / tf.to_float(sample_count)
            self.adversarial_batch_turn_size = (sample_rebuild_batch_size + target_rebuild_batch_size)
            self.adversarial_target_count = (sample_count + target_count)
            
        
        with tf.variable_scope("Generator_Gradient"):
            gradients = tf.gradients(self.generator_loss, self.generator_params)
            grad, self.generator_gradient_norm_summary = self.gradient_clip(gradients, hparams.max_gradient_norm, add_string = "generator_")
            opt = tf.train.AdamOptimizer(self.generator_learning_rate)
            self.generator_update = opt.apply_gradients(zip(grad, self.generator_params), global_step = self.generator_step)
            
        with tf.variable_scope("Discriminator_Gradient"):
            gradients = tf.gradients(self.discriminator_loss, self.discriminator_params)
            grad, self.discriminator_gradient_norm_summary = self.gradient_clip(gradients, hparams.max_gradient_norm, add_string = "discriminator_")
            opt = tf.train.AdamOptimizer(self.discriminator_learning_rate)
            self.discriminator_update = opt.apply_gradients(zip(grad, self.discriminator_params), global_step = self.discriminator_step)
            
        with tf.variable_scope("Adversarial_Gradient"):
            gradients = tf.gradients(self.adversarial_loss, self.decoder_params)
            grad, self.adversarial_gradient_norm_summary = self.gradient_clip(gradients, hparams.max_gradient_norm, add_string = "adversarial_")
            opt = tf.train.AdamOptimizer(self.adversarial_learning_rate)
            self.adversarial_update = opt.apply_gradients(zip(grad, self.decoder_params), global_step = self.adversarial_step)
            
            gradients = tf.gradients(self.generator_loss, self.generator_params)
            grad, self.teacher_forcing_gradient_norm_summary = self.gradient_clip(gradients, hparams.max_gradient_norm, add_string = "teacher_forcing_")
            tf_opt = tf.train.AdamOptimizer(self.adversarial_learning_rate)
            
            with tf.control_dependencies([self.adversarial_update]):
                self.teacher_forcing_update = tf_opt.apply_gradients(zip(grad, self.generator_params), global_step = self.adversarial_step)
            
            
        total_params = self.generator_params + self.discriminator_params
        print("# Trainable variables:")
        for param in total_params:
            print("  {}, {}, {}".format(param.name, str(param.get_shape()), param.op.device))

        # Tensorboard
        self.generator_summary = tf.summary.merge([
                tf.summary.scalar("generator_learning_rate", self.generator_learning_rate),
                tf.summary.scalar("response_flag_loss", self.response_flag_loss),
                tf.summary.scalar("response_flag_accuracy", self.response_flag_accuracy),
                tf.summary.scalar("emotion_predict_loss", self.emotion_predict_loss),
                tf.summary.scalar("emotion_predict_accuracy", self.emotion_predict_accuracy),
                tf.summary.scalar("emotion_output_loss", self.emotion_output_loss),
                tf.summary.scalar("emotion_output_accuracy", self.emotion_output_accuracy),
                tf.summary.scalar("decoder_loss", self.decoder_loss),
                tf.summary.scalar("decoder_perplexity", self.decoder_perplexity),
                tf.summary.scalar("generator_loss", self.generator_loss)] \
                + [tf.summary.scalar("emotion_predict_accuracy_{}".format(self.emotion_list[index]), self.specific_emotion_predict_accuracy[index]) for index in range(self.emotion_size)]
                + [tf.summary.scalar("emotion_output_accuracy_{}".format(self.emotion_list[index]), self.specific_emotion_output_accuracy[index]) for index in range(self.emotion_size)]
                + self.generator_gradient_norm_summary)
        self.discriminator_summary = tf.summary.merge([
                tf.summary.scalar("discriminator_learning_rate", self.discriminator_learning_rate),
                tf.summary.scalar("discriminator_loss", self.discriminator_loss),
                tf.summary.scalar("discriminator_accuracy", self.discriminator_accuracy)] \
                + self.discriminator_gradient_norm_summary)
        self.adversarial_summary = tf.summary.merge([
                tf.summary.scalar("adversarial_learning_rate", self.adversarial_learning_rate),
                tf.summary.scalar("response_flag_loss", self.response_flag_loss),
                tf.summary.scalar("response_flag_accuracy", self.response_flag_accuracy),
                tf.summary.scalar("emotion_predict_loss", self.emotion_predict_loss),
                tf.summary.scalar("emotion_predict_accuracy", self.emotion_predict_accuracy),
                tf.summary.scalar("emotion_output_loss", self.emotion_output_loss),
                tf.summary.scalar("emotion_output_accuracy", self.emotion_output_accuracy),
                tf.summary.scalar("decoder_loss", self.decoder_loss),
                tf.summary.scalar("decoder_perplexity", self.decoder_perplexity),
                tf.summary.scalar("generator_loss", self.generator_loss),
                tf.summary.scalar("discriminator_accuracy", self.discriminator_accuracy),
                tf.summary.scalar("sample_loss", self.sample_loss),
                tf.summary.scalar("target_loss", self.target_loss),
                tf.summary.scalar("adversarial_loss", self.adversarial_loss),
                tf.summary.scalar("adversarial_discriminate", self.adversarial_discriminate)] \
                + [tf.summary.scalar("emotion_predict_accuracy_{}".format(self.emotion_list[index]), self.specific_emotion_predict_accuracy[index]) for index in range(self.emotion_size)]
                + [tf.summary.scalar("emotion_output_accuracy_{}".format(self.emotion_list[index]), self.specific_emotion_output_accuracy[index]) for index in range(self.emotion_size)]
                + self.teacher_forcing_gradient_norm_summary + self.adversarial_gradient_norm_summary)
    
    def gradient_clip(self, gradients, max_gradient_norm, add_string = ""):
        """Clipping gradients of model."""
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        gradient_norm_summary = [tf.summary.scalar(add_string+"grad_norm", gradient_norm)]
        gradient_norm_summary.append(tf.summary.scalar(add_string+"clipped_gradient", tf.global_norm(clipped_gradients)))
        return clipped_gradients, gradient_norm_summary
    
    def update(self, mode, sess, feed_dict, train_D = False):
        assert self.training
            
        if mode == 'gen-train':
            update_list = self.generator_update_list
        elif mode == 'dis-train':
            update_list = self.discriminator_update_list
        elif mode == 'adv-train':
            if train_D:
                update_list = [self.adversarial_loss, self.batch_size, self.adversarial_batch_turn_size, self.generator_batch_turn_size, 
                                 self.response_flag_accuracy, self.emotion_predict_accuracy, self.emotion_output_accuracy, 
                                 self.generator_target_count, self.decoder_perplexity_sum, \
                                 self.discriminator_batch_turn_size, self.discriminator_accuracy, self.adversarial_target_count, self.adversarial_discriminate] \
                                 + self.specific_emotion_predict_count + self.specific_emotion_predict_correct \
                                 + self.specific_emotion_output_count + self.specific_emotion_output_correct \
                                 + [self.adversarial_update, self.teacher_forcing_update, self.discriminator_update, self.adversarial_summary, self.adversarial_step]
            else:
                update_list = [self.adversarial_loss, self.batch_size, self.adversarial_batch_turn_size, self.generator_batch_turn_size, 
                                 self.response_flag_accuracy, self.emotion_predict_accuracy, self.emotion_output_accuracy, 
                                 self.generator_target_count, self.decoder_perplexity_sum, \
                                 self.discriminator_batch_turn_size, self.discriminator_accuracy, self.adversarial_target_count, self.adversarial_discriminate] \
                                 + self.specific_emotion_predict_count + self.specific_emotion_predict_correct \
                                 + self.specific_emotion_output_count + self.specific_emotion_output_correct \
                                 + [self.adversarial_update, self.teacher_forcing_update, self.adversarial_summary, self.adversarial_step]
                                 
        response_flag_logits = tf.greater(tf.argmax(self.response_flag_logits, axis=2), 0)
        emotion_predict_encoder_logits = tf.to_int64(tf.argmax(tf.nn.softmax(self.emotion_predict_encoder_logits),axis=2))
        emotion_output_logits = tf.to_int64(tf.argmax(tf.nn.softmax(self.emotion_output_logits),axis=2))
        sentence_output_id = self.reverse_vocab_table.lookup(tf.to_int64(self.sentence_output_id))
        output_sample_id = self.reverse_vocab_table.lookup(tf.to_int64(self.output_sample_id))
        sen_input = self.reverse_vocab_table.lookup(tf.to_int64(self.batch_input.sen_input))
        sen_output = self.reverse_vocab_table.lookup(tf.to_int64(self.batch_input.sen_output))
        
        question = self.reverse_vocab_table.lookup(tf.to_int64(self.question_ids))
        response = self.reverse_vocab_table.lookup(tf.to_int64(self.discriminator_response_ids))
        
        emotion_predict_guess = self.reverse_emotion_table.lookup(emotion_predict_encoder_logits)
        emotion_output_guess = self.reverse_emotion_table.lookup(emotion_output_logits)
        
        emotion_predict_label = self.reverse_emotion_table.lookup(tf.to_int64(self.batch_input.emo_predict))
        emotion_output_label = self.reverse_emotion_table.lookup(tf.to_int64(self.batch_input.emo_output))
        
        correct_num = 0
        error_num = 0
        while True:
            
            response_flag_logits_1, sentence_output_id_1, output_sample_id_1, \
            emotion_predict_guess_1, emotion_output_guess_1, \
            sen_input_1, emo_predict_label_1, sen_output_1, emo_output_label_1, \
            question_text_1, response_text_1, reward_labels_1 \
                = sess.run([response_flag_logits, sentence_output_id, output_sample_id,
                emotion_predict_guess, emotion_output_guess,
                sen_input, emotion_predict_label, sen_output, emotion_output_label,
                question, response, self.discriminator_reward_labels], feed_dict=feed_dict)
            
            print('response_flag_logits', response_flag_logits_1)
            print('-'*30)
            print('output_sample_id',output_sample_id_1)
            print('sen_output',sen_output_1)
            print('sen_input', sen_input_1)
            print('-'*30)
            print('emotion_predict_encoder_logits',emotion_predict_guess_1)
            print('emo_predict',emo_predict_label_1)
            print('-'*30)
            print('emotion_output_logits',emotion_output_guess_1)
            print('emo_output',emo_output_label_1)
            print('='*30)
            
            non_token = self.hparams.non_token.encode("utf-8")
            eos_token = self.hparams.eos_token.encode("utf-8")
            
            for batch_question, batch_response_target, batch_response_sample, b_emo_pre_lab, b_emo_pre_gus, b_emo_out_lab, b_emo_out_gus  in zip(
                    sen_input_1, sen_output_1, output_sample_id_1, emo_predict_label_1, emotion_predict_guess_1, emo_output_label_1, emotion_output_guess_1):
                print('-'*30)
                for qust, rtgt, rsmp, empl, empg, emol, emog in zip(batch_question, batch_response_target, batch_response_sample, b_emo_pre_lab, b_emo_pre_gus, b_emo_out_lab, b_emo_out_gus):
                    try:
                        qust = qust.tolist()
                        if non_token in qust:
                            qust = qust[:qust.index(non_token)]
                            qust = b' '.join(qust).decode('utf-8')
                    except:
                        pass
                    try:
                        rtgt = rtgt.tolist()
                        if non_token in rtgt:
                            rtgt = rtgt[:rtgt.index(non_token)]
                            rtgt = b' '.join(rtgt).decode('utf-8')
                    except:
                        pass
                    try:
                        rsmp = rsmp.tolist()
                        if non_token in rsmp:
                            rsmp = rsmp[:rsmp.index(non_token)]
                            rsmp = b' '.join(rsmp).decode('utf-8')
                    except:
                        pass
                    
                    if type(qust) == type(list()):
                        qust = b' '.join(qust).decode('utf-8')
                    if type(rtgt) == type(list()):
                        rtgt = b' '.join(rtgt).decode('utf-8')
                    if type(rsmp) == type(list()):
                        rsmp = b' '.join(rsmp).decode('utf-8')
                    
                    empl = empl.decode('utf-8')
                    empg = empg.decode('utf-8')
                    emol = emol.decode('utf-8')
                    emog = emog.decode('utf-8')
                    print('Q E(T:{})/(G:{}) : {}'.format(empl, empg, qust))
                    print('A(T) E({}) : {}'.format(emol, rtgt))
                    print('A(G) E({}) : {}'.format(emog, rsmp))
            
            print('='*30)
            print('-'*30)
            print('='*30)
            
            guessing_list = []
            for question_print, response_print, reward_print in zip(question_text_1, response_text_1, reward_labels_1):
                question_print = question_print.tolist()
                response_print = response_print.tolist()
                if non_token in question_print:
                    question_print = question_print[:question_print.index(non_token)]
                if non_token in response_print:
                    response_print = response_print[:response_print.index(non_token)]
                question_print = b' '.join(question_print).decode('utf-8')
                response_print = b' '.join(response_print).decode('utf-8')
                guessing_list.append({'question' : question_print,
                                      'response' : response_print,
                                      'reward' : str(reward_print[0])})
#            random.shuffle(guessing_list)
            for Dict in guessing_list:
                split_text = Dict['question'].split('_eos_')
                for text in split_text:
                    if text != '':
                        print('Q : {}'.format(text))
                        
                split_text = Dict['response'].split('_eos_')
                for text in split_text:
                    if text != '':
                        print('A : {}'.format(text))
                
                print('ground truth :', Dict['reward'])
#                answer = input('enter 1/0 :')
#                if answer == Dict['reward']:
#                    correct_num += 1
#                    print('- O -')
#                else:
#                    error_num += 1
#                    print('- X -')
#                print('--------------------')
#            print('correct / error  = {} / {}'.format(correct_num, error_num))
            
            if input('enter q to quit') == 'q':
                sys.exit()
        
        
        result = sess.run(update_list, feed_dict=feed_dict)
        loss, batch_size, batch_turn_size, summary, step = result[0], result[1], result[2], result[-2], result[-1]
        result_dict = self.dict_handle(mode, result)
        
        return loss, batch_size, batch_turn_size, result_dict, summary, step
    
    def test(self, mode, sess, dataset_handler):
        """update model params"""
        assert self.training
            
        if mode == 'gen-train':
            test_list = self.generator_update_list[:-3]
        elif mode == 'dis-train':
            test_list = self.discriminator_update_list[:-3]
        elif mode == 'adv-train':
            test_list = self.adversarial_update_list[:-4]
            
        result = sess.run(test_list, feed_dict = dataset_handler)
        loss, batch_size, batch_turn_size = result[0], result[1], result[2]
        result_dict = self.dict_handle(mode, result)
        
        return loss, batch_size, batch_turn_size, result_dict
                         
    def dict_handle(self, mode, result):
        if mode == 'gen-train':
            result_dict = {'response_flag_accuracy_correct':result[3] * result[2],
                           'emotion_predict_accuracy_correct':result[4] * result[2],
                           'emotion_output_accuracy_correct':result[5] * result[2],
                           'generator_target_count':result[6],
                           'decoder_perplexity_sum':result[7]}
            for index in range(self.emotion_size):
                result_dict.update({'{}_specific_emotion_predict_count'.format(self.emotion_list[index]):result[8+index]})
                result_dict.update({'{}_specific_emotion_predict_correct'.format(self.emotion_list[index]):result[8+self.emotion_size+index]})
                result_dict.update({'{}_specific_emotion_output_count'.format(self.emotion_list[index]):result[8+self.emotion_size*2+index]})
                result_dict.update({'{}_specific_emotion_output_correct'.format(self.emotion_list[index]):result[8+self.emotion_size*3+index]})
        elif mode == 'dis-train':
            result_dict = {'discriminator_accuracy_correct':result[3] * result[2]}
        elif mode == 'adv-train':
            result_dict = {'generator_batch_turn_size':result[3],
                           'response_flag_accuracy_correct':result[4] * result[3],
                           'emotion_predict_accuracy_correct':result[5] * result[3],
                           'emotion_output_accuracy_correct':result[6] * result[3],
                           'generator_target_count':result[7],
                           'decoder_perplexity_sum':result[8],
                           'discriminator_batch_turn_size':result[9],
                           'discriminator_accuracy_correct':result[10] * result[9],
                           'adversarial_target_count':result[11],
                           'adversarial_discriminate_sum':result[12] * result[11]}
            for index in range(self.emotion_size):
                result_dict.update({'{}_specific_emotion_predict_count'.format(self.emotion_list[index]):result[13+index]})
                result_dict.update({'{}_specific_emotion_predict_correct'.format(self.emotion_list[index]):result[13+self.emotion_size+index]})
                result_dict.update({'{}_specific_emotion_output_count'.format(self.emotion_list[index]):result[13+self.emotion_size*2+index]})
                result_dict.update({'{}_specific_emotion_output_correct'.format(self.emotion_list[index]):result[13+self.emotion_size*3+index]})
            
        return result_dict
# =============================================================================
# Create model
# =============================================================================
    def build_multi_turn_model(self, hparams):

        with tf.variable_scope('Multi_turn_generator'):
            with tf.name_scope('init_output_placeholder'):
                turn = tf.constant(0)
                init_latent_state = tf.zeros([hparams.num_layers, self.batch_size, hparams.num_units],tf.float32) if hparams.num_layers > 1 else tf.zeros([self.batch_size, hparams.num_units],tf.float32)
                init_emotion_state = tf.zeros([hparams.num_layers, self.batch_size, hparams.num_emotion_units],tf.float32) if hparams.num_layers > 1 else tf.zeros([self.batch_size, hparams.num_emotion_units],tf.float32)
                init_latent_state_shape = tf.TensorShape([None,None,None]) if hparams.num_layers > 1 else tf.TensorShape([None,None])
                init_emotion_state_shape = tf.TensorShape([None,None,None]) if hparams.num_layers > 1 else tf.TensorShape([None,None])
                init_response_flag_logits = tf.slice(tf.zeros([self.batch_size, 1, 2],tf.float32) , [0,0,0], [self.batch_size, 0, 2])
                init_sentence_output_logits = tf.slice(tf.zeros([self.batch_size, 1, tf.shape(self.batch_input.sen_output)[2], self.vocab_size],tf.float32) , [0,0,0,0], [self.batch_size, 0, tf.shape(self.batch_input.sen_output)[2], self.vocab_size])
                init_sentence_output_id = tf.slice(tf.zeros([self.batch_size, 1, tf.shape(self.batch_input.sen_output)[2]],tf.int32) , [0,0,0], [self.batch_size, 0, tf.shape(self.batch_input.sen_output)[2]])
                init_output_sample_logits = tf.slice(tf.zeros([self.batch_size, 1, tf.shape(self.batch_input.sen_output)[2], self.vocab_size],tf.float32) , [0,0,0,0], [self.batch_size, 0, tf.shape(self.batch_input.sen_output)[2], self.vocab_size])
                init_output_sample_id = tf.slice(tf.zeros([self.batch_size, 1, tf.shape(self.batch_input.sen_output)[2]],tf.int32) , [0,0,0], [self.batch_size, 0, tf.shape(self.batch_input.sen_output)[2]])
                init_emotion_predict_encoder_logits = tf.slice(tf.zeros([self.batch_size, 1, self.emotion_size],tf.float32) , [0,0,0], [self.batch_size, 0, self.emotion_size])
                init_emotion_predict_decoder_logits = tf.slice(tf.zeros([self.batch_size, 1, self.emotion_size],tf.float32) , [0,0,0], [self.batch_size, 0, self.emotion_size])
                init_emotion_output_logits = tf.slice(tf.zeros([self.batch_size, 1, self.emotion_size],tf.float32) , [0,0,0], [self.batch_size, 0, self.emotion_size])
            def multi_turn_model(turn, response_flag_logits, sentence_output_logits, sentence_output_id, output_sample_logits, output_sample_id, latent_state, emotion_state, emotion_predict_encoder_logits, emotion_predict_decoder_logits, emotion_output_logits):
                # Batch input
                sentence_input_id = tf.squeeze(tf.slice(self.batch_input.sen_input, [0, turn, 0], [-1, 1, -1]), [1])
                sentence_input_length = tf.squeeze(tf.slice(self.batch_input.sen_input_length, [0, turn], [-1, 1]), [1])
#                emotion_target_predict = tf.squeeze(tf.slice(self.batch_input.emo_predict, [0, turn], [-1, 1]), [1])
                sentence_target_id = tf.squeeze(tf.slice(self.batch_input.sen_output, [0, turn, 0], [-1, 1, -1]), [1])
                sentence_target_length = tf.squeeze(tf.slice(self.batch_input.sen_output_length, [0, turn], [-1, 1]), [1])
                emotion_target_output = tf.squeeze(tf.slice(self.batch_input.emo_output, [0, turn], [-1, 1]), [1])
                # Last input
                if self.training:
                    passing_flag = tf.greater(turn, 0)
                latent_state = tuple([tf.squeeze(tf.slice(latent_state, [index,0,0], [1, -1, -1]), [0]) for index in range(hparams.num_layers)])
                emotion_state = tuple([tf.squeeze(tf.slice(emotion_state, [index,0,0], [1, -1, -1]), [0]) for index in range(hparams.num_layers)])
                # Build model
                res_flag_logits, sent_output_logits, sen_output_id, out_sample_logits, out_sample_id, lat_state, \
                emo_state, emo_predict_encoder_logits, emo_predict_decoder_logits, emo_output_logits = self.build_model(
                        hparams, passing_flag, sentence_input_id, sentence_target_id,
                        sentence_input_length, sentence_target_length, emotion_target_output, latent_state, emotion_state)
                # Output combine
                latent_state = tf.concat([tf.expand_dims(state, axis=0) for state in lat_state], axis=0)
                emotion_state = tf.concat([tf.expand_dims(state, axis=0) for state in emo_state], axis=0)
                response_flag_logits = tf.concat([response_flag_logits, res_flag_logits], axis=1)
                
                sent_output_logits = tf.expand_dims(sent_output_logits, axis=1)
                sen_output_id = tf.expand_dims(sen_output_id, axis=1)
                out_sample_logits = tf.expand_dims(out_sample_logits, axis=1)
                out_sample_id = tf.expand_dims(out_sample_id, axis=1)
                
                tims_diff = tf.shape(sentence_output_id)[2] - tf.shape(sen_output_id)[2]
                non_id = tf.zeros([self.batch_size, 1, tims_diff],tf.int32)
                non_logit = tf.one_hot(non_id, self.vocab_size)
                sent_output_logits = tf.concat([sent_output_logits, non_logit], axis=2)
                sen_output_id = tf.concat([sen_output_id, non_id], axis=2)
                
                tims_diff = tf.shape(sentence_output_id)[2] - tf.shape(out_sample_id)[2]
                non_id = tf.zeros([self.batch_size, 1, tims_diff],tf.int32)
                non_logit = tf.one_hot(non_id, self.vocab_size)
                out_sample_logits = tf.concat([out_sample_logits, non_logit], axis=2)
                out_sample_id = tf.concat([out_sample_id, non_id], axis=2)
                
                sentence_output_logits = tf.concat([sentence_output_logits, sent_output_logits], axis=1)
                sentence_output_id = tf.concat([sentence_output_id, sen_output_id], axis=1)
                output_sample_logits = tf.concat([output_sample_logits, out_sample_logits], axis=1)
                output_sample_id = tf.concat([output_sample_id, out_sample_id], axis=1)
                
                emotion_predict_encoder_logits = tf.concat([emotion_predict_encoder_logits, emo_predict_encoder_logits], axis=1)
                emotion_predict_decoder_logits = tf.concat([emotion_predict_decoder_logits, emo_predict_decoder_logits], axis=1)
                emotion_output_logits = tf.concat([emotion_output_logits, emo_output_logits], axis=1)
                
                return tf.add(turn, 1), response_flag_logits, sentence_output_logits, sentence_output_id, output_sample_logits, output_sample_id, latent_state, emotion_state, emotion_predict_encoder_logits, emotion_predict_decoder_logits, emotion_output_logits
            _ , response_flag_logits, sentence_output_logits, sentence_output_id, output_sample_logits, output_sample_id, \
            latent_state, emotion_state, emotion_predict_encoder_logits, emotion_predict_decoder_logits, emotion_output_logits = tf.while_loop(
                    lambda turn, *_: tf.less(turn, tf.reduce_max(self.batch_input.turn_length)),
                    multi_turn_model,
                    loop_vars=[turn, init_response_flag_logits, init_sentence_output_logits, init_sentence_output_id, init_output_sample_logits, init_output_sample_id,
                               init_latent_state, init_emotion_state, init_emotion_predict_encoder_logits, init_emotion_predict_decoder_logits, init_emotion_output_logits],
                    shape_invariants=[turn.get_shape(), tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None,None]), tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None,None]), tf.TensorShape([None,None,None]),
                                      init_latent_state_shape, init_emotion_state_shape, tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None])])
            
        with tf.variable_scope('Multi_turn_discriminator'):
            response_length, response_ids, response_logits, reward_labels, reward_logits = self.build_discriminator(hparams, self.batch_input.sen_input, self.batch_input.sen_output, output_sample_id, output_sample_logits)
        
        return response_flag_logits, sentence_output_logits, sentence_output_id, output_sample_logits, output_sample_id, latent_state, emotion_state, emotion_predict_encoder_logits, emotion_predict_decoder_logits, emotion_output_logits, response_length, response_ids, response_logits, reward_labels, reward_logits

    def build_model(self, hparams, passing_flag, sentence_input_id, sentence_target_id, sentence_input_length, sentence_target_length, emotion_target_output, latent_state, emotion_state):
        """Build generator graph."""
        #Build Generator
        with tf.variable_scope('Generator') as g_scope:
            
            #Build Encoder
            with tf.variable_scope('Encoder') as scope:
                encoder_outputs, encoder_state = self.build_encoder(
                        hparams, sentence_input_id, embedding=self.embedding,
                        length=tf.math.maximum(sentence_input_length, 1), time_major=False, scope=scope)
                encoder_state_input = tf.concat([state for state in encoder_state],axis=1) if hparams.num_layers > 1 else encoder_state
                self.encoder_params = scope.trainable_variables()
                
            #Build Latent
            with tf.variable_scope('Latent', reuse=tf.AUTO_REUSE) as scope:
                response_flag_logits, latent_state, latent_cell = self.build_latent_encoder(hparams, passing_flag, latent_state, encoder_state_input, scope)
#                if self.mode == 'inference':
#                    response_flag = tf.argmax(response_flag_logits, axis=2)
#                else:
#                    # teacher_forcing
#                    response_flag = tf.expand_dims(tf.cast(tf.minimum(sentence_target_length, 1), tf.int32),axis=1)
                
                response_flag = tf.argmax(response_flag_logits, axis=2)    
                    
                latent_state_input = tf.concat([state for state in encoder_state],axis=1) if hparams.num_layers > 1 else latent_state
                emotion_input = tf.layers.dense(latent_state_input, hparams.num_emotion_units*2, activation=tf.nn.leaky_relu, name = "emotion_dense1")
                emotion_input = tf.layers.dense(emotion_input, hparams.num_emotion_units, activation=tf.nn.leaky_relu, name = "emotion_dense2")
                emotion_predict_encoder_logits = tf.layers.dense(tf.expand_dims(emotion_input, axis=1), self.emotion_size, name="latent_emotion_classifier")

            #Build Emotion
            with tf.variable_scope('Emotion') as scope:
                state_input = tf.concat([latent_state_input, encoder_state_input], axis=-1)
                emotion_output_logits, emotion_state = self.build_emotion_transfer(hparams, passing_flag, emotion_state, state_input, scope)
#                if self.mode == 'inference':
#                    emotion_output = self.reverse_emotion_table.lookup(tf.to_int64(tf.argmax(emotion_output_logits, axis=2)) * response_flag)
#                else:
#                    # teacher_forcing
#                    emotion_output = self.reverse_emotion_table.lookup(tf.to_int64(emotion_target_output))
                
                emotion_output = self.reverse_emotion_table.lookup(tf.to_int64(tf.argmax(emotion_output_logits, axis=2)) * response_flag)
                    
                self.emotion_params = scope.trainable_variables()
                
            #Build Decoder
            with tf.variable_scope('Decoder') as scope:
                sentence_output_logits, sentence_output_id, decoder_state, output_sample_logits, output_sample_id = self.build_decoder(
                        hparams, sentence_target_id, tf.maximum(sentence_input_length, 1), tf.maximum(sentence_target_length, 1),
                        encoder_outputs, encoder_state, latent_state, emotion_output)
                
                sentence_output_id = sentence_output_id * tf.concat([tf.cast(response_flag, tf.int32), tf.ones([self.batch_size, tf.shape(sentence_output_id)[1] -1], tf.int32)], axis=1)
                output_sample_id = output_sample_id * tf.concat([tf.cast(response_flag, tf.int32), tf.ones([self.batch_size, tf.shape(output_sample_id)[1] -1], tf.int32)], axis=1)
                
                decoder_state_input = tf.concat([state for state in decoder_state],axis=1) if hparams.num_layers > 1 else decoder_state.cell_state
                decoder_state_input = tf.reshape(decoder_state_input, [-1, hparams.num_units*2])
                self.decoder_params = scope.trainable_variables()
                
            with tf.variable_scope('Latent', reuse=tf.AUTO_REUSE) as scope:
                if not self.training and hparams.beam_width > 0:
                    if hparams.num_layers > 1:
                        latent_state = tf.concat([tf.expand_dims(state, axis=0) for state in decoder_state],axis=0) if hparams.num_layers > 1 else decoder_state.cell_state
                        latent_state = tuple([tf.squeeze(tf.slice(latent_state, [index,0,0], [1, 1,-1]), [0]) for index in range(hparams.num_layers)])
                    
                _, latent_state, _ = self.build_latent_encoder(hparams, None, latent_state, decoder_state_input, scope, cell = latent_cell)
                latent_state_input = tf.concat([state for state in encoder_state],axis=1) if hparams.num_layers > 1 else latent_state
                emotion_input = tf.layers.dense(latent_state_input, hparams.num_emotion_units*2, activation=tf.nn.leaky_relu, name = "emotion_dense1")
                emotion_input = tf.layers.dense(emotion_input, hparams.num_emotion_units, activation=tf.nn.leaky_relu, name = "emotion_dense2")
                emotion_predict_decoder_logits = tf.layers.dense(tf.expand_dims(emotion_input, axis=1), self.emotion_size, name="latent_emotion_classifier")
                self.latent_params = scope.trainable_variables()
                
            self.generator_params = g_scope.trainable_variables()
            
        return response_flag_logits, sentence_output_logits, sentence_output_id, output_sample_logits, output_sample_id, latent_state, emotion_state, emotion_predict_encoder_logits, emotion_predict_decoder_logits, emotion_output_logits
    
    def build_discriminator(self, hparams, sentence_input_id, sentence_target_id, output_sample_id, output_sample_logits):
        # [batch_size, max_turn, max_time, num_units]
        encode_length = tf.cast(tf.argmin(tf.concat([sentence_input_id, tf.zeros([self.batch_size, tf.shape(sentence_input_id)[1], 1], tf.int32)], axis=2), axis=2), tf.int32)
        decode_length = tf.cast(tf.argmin(tf.concat([output_sample_id, tf.zeros([self.batch_size, tf.shape(output_sample_id)[1], 1], tf.int32)], axis=2), axis=2), tf.int32)
        encode_ids = sentence_input_id
        decode_ids = output_sample_id
        encode_logits = tf.one_hot(sentence_input_id, self.vocab_size)
        decode_logits = output_sample_logits
        
        target_length = tf.cast(tf.argmin(tf.concat([sentence_target_id, tf.zeros([self.batch_size, tf.shape(sentence_target_id)[1],1], tf.int32)], axis=2), axis=2), tf.int32)
        encode_length = tf.concat([encode_length, encode_length], axis=0)
        decode_length = tf.concat([decode_length, target_length], axis=0)
        target_ids = sentence_target_id
        encode_ids = tf.concat([encode_ids, encode_ids], axis=0)
        decode_ids = tf.concat([decode_ids, target_ids], axis=0)
        target_logits = tf.one_hot(sentence_target_id, self.vocab_size)
        encode_logits = tf.concat([encode_logits, encode_logits], axis=0)
        decode_logits = tf.concat([decode_logits, target_logits], axis=0)
        
        tims_diff = tf.shape(encode_ids)[2] - tf.shape(decode_ids)[2]
        encode_ids = tf.cond(tf.less(tims_diff, 0), lambda: tf.concat([encode_ids, tf.zeros([tf.shape(encode_ids)[0], tf.shape(encode_ids)[1], tf.abs(tims_diff)], tf.int32)], axis=2),  lambda: encode_ids)
        decode_ids = tf.cond(tf.greater(tims_diff, 0), lambda: tf.concat([decode_ids, tf.zeros([tf.shape(decode_ids)[0], tf.shape(decode_ids)[1], tims_diff], tf.int32)], axis=2), lambda: decode_ids)
        encode_logits = tf.cond(tf.less(tims_diff, 0), lambda: tf.concat([encode_logits, tf.zeros([tf.shape(encode_ids)[0], tf.shape(encode_ids)[1], tf.abs(tims_diff), self.vocab_size], tf.float32)], axis=2), lambda: encode_logits)
        decode_logits = tf.cond(tf.greater(tims_diff, 0), lambda: tf.concat([decode_logits, tf.zeros([tf.shape(encode_ids)[0], tf.shape(encode_ids)[1], tims_diff, self.vocab_size], tf.float32)], axis=2), lambda: decode_logits)
        
        batch_size, max_turn, max_time, logits_size = tf.shape(encode_logits)[0], tf.shape(encode_logits)[1], tf.shape(encode_logits)[2], tf.shape(encode_logits)[3]
        if self.mode == 'dis-train':
            reward_labels = tf.concat([tf.zeros([batch_size/2, max_turn], tf.int32), tf.ones([batch_size/2, max_turn], tf.int32)], axis=0)
        else:
            reward_labels = tf.zeros([batch_size/2, max_turn], tf.int32)
        reward_labels = tf.concat([tf.zeros([batch_size/2, max_turn], tf.int32), tf.ones([batch_size/2, max_turn], tf.int32)], axis=0)
            
        reward_output_labels = tf.slice(tf.zeros([1], tf.int32), [0], [0])
        encode_output_length = tf.slice(tf.zeros([1], tf.int32), [0], [0])
        decode_output_length = tf.slice(tf.zeros([1], tf.int32), [0], [0])
        encode_output_ids = tf.slice(tf.zeros([1, max_time], tf.int32), [0,0], [0,max_time])
        decode_output_ids = tf.slice(tf.zeros([1, max_time], tf.int32), [0,0], [0,max_time])
        encode_output_logits = tf.slice(tf.zeros([1, max_time, logits_size], tf.float32), [0,0,0], [0,max_time,logits_size])
        decode_output_logits = tf.slice(tf.zeros([1, max_time, logits_size], tf.float32), [0,0,0], [0,max_time,logits_size])
        batch = tf.constant(0)
        def rebuild_sentence_input(batch, reward_output_labels, encode_output_length, decode_output_length, encode_output_ids, decode_output_ids, encode_output_logits, decode_output_logits):
            reward_slice_labels = tf.squeeze(tf.slice(reward_labels, [batch,0], [1,max_turn]),[0])
            encode_slice_length = tf.squeeze(tf.slice(encode_length, [batch,0], [1,max_turn]),[0])
            decode_slice_length = tf.squeeze(tf.slice(decode_length, [batch,0], [1,max_turn]),[0])
            encode_slice_ids = tf.squeeze(tf.slice(encode_ids, [batch,0,0], [1,max_turn,max_time]),[0])
            decode_slice_ids = tf.squeeze(tf.slice(decode_ids, [batch,0,0], [1,max_turn,max_time]),[0])
            encode_slice_logits = tf.squeeze(tf.slice(encode_logits, [batch,0,0,0], [1,max_turn,max_time,logits_size]),[0])
            decode_slice_logits = tf.squeeze(tf.slice(decode_logits, [batch,0,0,0], [1,max_turn,max_time,logits_size]),[0])
            batch_start = tf.cast(tf.squeeze(tf.reduce_min(tf.where(tf.not_equal(encode_slice_length, 0)), axis=0)), tf.int32)
            batch_end = tf.cast(tf.squeeze(tf.reduce_max(tf.where(tf.not_equal(decode_slice_length, 0)), axis=0)), tf.int32)
            batch_num = tf.maximum((batch_end - batch_start) + 1, 0)
            batch_num = tf.cond(tf.less(batch_start, 0), lambda: 0, lambda: batch_num)
            batch_start = tf.cond(tf.less(batch_start, 0), lambda: 0, lambda: batch_start)

            reward_output_labels = tf.concat([reward_output_labels, tf.slice(reward_slice_labels, [batch_start], [batch_num])], axis=0)
            encode_output_length = tf.concat([encode_output_length, tf.slice(encode_slice_length, [batch_start], [batch_num])], axis=0)
            decode_output_length = tf.concat([decode_output_length, tf.slice(decode_slice_length, [batch_start], [batch_num])], axis=0)
            encode_output_ids = tf.concat([encode_output_ids, tf.slice(encode_slice_ids, [batch_start,0], [batch_num,max_time])], axis=0)
            decode_output_ids = tf.concat([decode_output_ids, tf.slice(decode_slice_ids, [batch_start,0], [batch_num,max_time])], axis=0)
            encode_output_logits = tf.concat([encode_output_logits, tf.slice(encode_slice_logits, [batch_start,0,0], [batch_num,max_time,logits_size])], axis=0)
            decode_output_logits = tf.concat([decode_output_logits, tf.slice(decode_slice_logits, [batch_start,0,0], [batch_num,max_time,logits_size])], axis=0)
            return tf.add(batch, 1), reward_output_labels, encode_output_length, decode_output_length, encode_output_ids, decode_output_ids, encode_output_logits, decode_output_logits
        _, reward_output_labels, encode_output_length, decode_output_length, encode_output_ids, decode_output_ids, encode_output_logits, decode_output_logits \
         = tf.while_loop(lambda batch, *_: tf.less(batch, batch_size), rebuild_sentence_input,
                         loop_vars=[batch, reward_output_labels, encode_output_length, decode_output_length, encode_output_ids, decode_output_ids, encode_output_logits, decode_output_logits],
                         shape_invariants=[batch.get_shape(), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None,None]), tf.TensorShape([None,None]), tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None])],
                         name='rebuild_sentence_input')
        
        suffle_reward_labels = tf.reshape(tf.transpose(tf.concat([tf.expand_dims(reward_output_labels, axis=0),tf.expand_dims(reward_output_labels, axis=0)], axis=0)), [-1])
        suffle_sentence_length = tf.reshape(tf.transpose(tf.concat([tf.expand_dims(encode_output_length, axis=0),tf.expand_dims(decode_output_length, axis=0)], axis=0)), [-1])
        suffle_sentence_tag = tf.reshape(tf.transpose(tf.concat([tf.expand_dims(tf.ones_like(encode_output_length),axis=0),tf.expand_dims(tf.zeros_like(decode_output_length),axis=0)], axis=0)), [-1])
        suffle_sentence_ids = tf.reshape(tf.transpose(tf.concat([tf.expand_dims(encode_output_ids, axis=0),tf.expand_dims(decode_output_ids, axis=0)], axis=0), [1, 0, 2]), [-1,max_time])
        suffle_sentence_logits = tf.reshape(tf.transpose(tf.concat([tf.expand_dims(encode_output_logits, axis=0),tf.expand_dims(decode_output_logits, axis=0)], axis=0), [1, 0, 2, 3]), [-1,max_time,logits_size])
        
        indices = tf.where(tf.not_equal(suffle_sentence_length, 0))
        suffle_size = tf.size(indices)
        suffle_reward_gather_labels = tf.reshape(tf.gather(suffle_reward_labels, indices),[suffle_size])
        suffle_sentence_gather_length = tf.reshape(tf.gather(suffle_sentence_length, indices),[suffle_size])
        suffle_sentence_gather_tag = tf.reshape(tf.gather(suffle_sentence_tag, indices),[suffle_size])
        suffle_sentence_gather_ids = tf.reshape(tf.gather(suffle_sentence_ids, indices),[suffle_size,max_time])
        suffle_sentence_gather_logits = tf.reshape(tf.gather(suffle_sentence_logits, indices),[suffle_size,max_time,logits_size])
        
        turn_num = tf.cast(tf.squeeze(tf.reduce_max(tf.where(tf.equal(suffle_sentence_gather_tag, 0)), axis=0)), tf.int32) + 1
        suffle_reward_gather_labels = tf.slice(suffle_reward_gather_labels, [0], [turn_num])
        suffle_sentence_gather_tag = tf.slice(suffle_sentence_gather_tag, [0], [turn_num])
        suffle_sentence_gather_length = tf.slice(suffle_sentence_gather_length, [0], [turn_num])
        suffle_sentence_gather_ids = tf.slice(suffle_sentence_gather_ids, [0,0], [turn_num,max_time])
        suffle_sentence_gather_logits = tf.slice(suffle_sentence_gather_logits, [0,0,0], [turn_num,max_time,logits_size])

        rebuild_max_turn = tf.size(suffle_sentence_gather_length)
        segment_ids = tf.slice(tf.zeros([1], tf.int32),[0],[0])
        suffle_sentence_ids = tf.slice(tf.zeros([1], tf.int32),[0],[0])
        suffle_sentence_logits = tf.slice(tf.zeros([1,logits_size], tf.float32),[0,0],[0,logits_size])
        turn, iter_index, last_coder = tf.constant(0), tf.constant(0), tf.constant(0)
        def get_segment_data(turn, iter_index, last_coder, segment_ids, suffle_sentence_ids, suffle_sentence_logits):
            coder, length = suffle_sentence_gather_tag[turn], suffle_sentence_gather_length[turn]
            iter_index = iter_index + tf.where(tf.not_equal(last_coder, coder), 1, 0)
            segment_ids = tf.concat([segment_ids, [iter_index-1]], axis=0)
            suffle_sentence_ids = tf.concat([suffle_sentence_ids, tf.squeeze(tf.slice(suffle_sentence_gather_ids, [turn, 0], [1, length]),[0])], axis=0)
            suffle_sentence_logits = tf.concat([suffle_sentence_logits, tf.squeeze(tf.slice(suffle_sentence_gather_logits, [turn, 0, 0], [1, length, -1]),[0])], axis=0)
            last_coder = coder
            return tf.add(turn, 1), iter_index, last_coder, segment_ids, suffle_sentence_ids, suffle_sentence_logits
        _, _, _, segment_ids, suffle_sentence_ids, suffle_sentence_logits\
         = tf.while_loop(lambda turn, *_: tf.less(turn, rebuild_max_turn), get_segment_data,
                         loop_vars=[turn, iter_index, last_coder, segment_ids, suffle_sentence_ids, suffle_sentence_logits],
                         shape_invariants=[turn.get_shape(), iter_index.get_shape(), last_coder.get_shape(), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None,None])],
                         name='get_segment_data')
        
        segment_num = tf.reduce_max(segment_ids)+1
        error_flag = tf.not_equal(tf.mod(segment_num, 2), 0)
        suffle_reward_labels = tf.unsorted_segment_max(suffle_reward_gather_labels, segment_ids, segment_num)
        suffle_sentence_length = tf.unsorted_segment_sum(suffle_sentence_gather_length, segment_ids, segment_num)
        
        suffle_reward_labels = tf.cond(error_flag, lambda: tf.concat([suffle_reward_labels,suffle_reward_labels],axis=0), lambda: suffle_reward_labels)
        suffle_sentence_length = tf.cond(error_flag, lambda: tf.concat([suffle_sentence_length,suffle_sentence_length],axis=0), lambda: suffle_sentence_length)
        suffle_sentence_ids = tf.cond(error_flag, lambda: tf.concat([suffle_sentence_ids,suffle_sentence_ids],axis=0), lambda: suffle_sentence_ids)
        suffle_sentence_logits = tf.cond(error_flag, lambda: tf.concat([suffle_sentence_logits,suffle_sentence_logits],axis=0), lambda: suffle_sentence_logits)
        
        batch_reward_labels = tf.transpose(tf.reshape(suffle_reward_labels, [-1, 2]))[0]
        batch_sentence_length = tf.reshape(suffle_sentence_length, [-1, 2])
        
        rebuild_max_time = tf.reduce_max(batch_sentence_length)
        rebuild_batch_size = tf.shape(batch_sentence_length)[0]
        question_length = tf.transpose(batch_sentence_length)[0]
        response_length = tf.transpose(batch_sentence_length)[1]
        question_ids = tf.slice(tf.zeros([1,rebuild_max_time], tf.int32),[0,0],[0,rebuild_max_time])
        response_ids = tf.slice(tf.zeros([1,rebuild_max_time], tf.int32),[0,0],[0,rebuild_max_time])
        question_logits = tf.slice(tf.zeros([1,rebuild_max_time,logits_size], tf.float32),[0,0,0],[0,rebuild_max_time,logits_size])
        response_logits = tf.slice(tf.zeros([1,rebuild_max_time,logits_size], tf.float32),[0,0,0],[0,rebuild_max_time,logits_size])
        batch, iter_length = tf.constant(0), tf.constant(0)
        def rebuild_discriminator_batch(batch, iter_length, question_ids, response_ids, question_logits, response_logits):
            question_slice_ids = tf.expand_dims(tf.concat([tf.slice(suffle_sentence_ids, [iter_length], [question_length[batch]]), tf.fill([rebuild_max_time - question_length[batch]], 0)], axis=0), axis=0)
            question_slice_logits = tf.expand_dims(tf.concat([tf.slice(suffle_sentence_logits, [iter_length,0], [question_length[batch],-1]), tf.fill([rebuild_max_time - question_length[batch],logits_size], 0.0)], axis=0), axis=0)
            iter_length = iter_length + question_length[batch]
            response_slice_ids = tf.expand_dims(tf.concat([tf.slice(suffle_sentence_ids, [iter_length], [response_length[batch]]), tf.fill([rebuild_max_time - response_length[batch]], 0)], axis=0), axis=0)
            response_slice_logits = tf.expand_dims(tf.concat([tf.slice(suffle_sentence_logits, [iter_length,0], [response_length[batch],-1]), tf.fill([rebuild_max_time - response_length[batch],logits_size], 0.0)], axis=0), axis=0)
            iter_length = iter_length + response_length[batch]
            
            question_ids = tf.concat([question_ids, question_slice_ids],axis=0)
            response_ids = tf.concat([response_ids, response_slice_ids],axis=0)
            question_logits = tf.concat([question_logits, question_slice_logits],axis=0)
            response_logits = tf.concat([response_logits, response_slice_logits],axis=0)
            return tf.add(batch, 1), iter_length, question_ids, response_ids, question_logits, response_logits
        _, _, question_ids, response_ids, question_logits, response_logits\
         = tf.while_loop(lambda batch, *_: tf.less(batch, rebuild_batch_size), rebuild_discriminator_batch,
                         loop_vars=[batch, iter_length, question_ids, response_ids, question_logits, response_logits],
                         shape_invariants=[batch.get_shape(), iter_length.get_shape(), tf.TensorShape([None,None]), tf.TensorShape([None,None]), tf.TensorShape([None,None,None]), tf.TensorShape([None,None,None])],
                         name='rebuild_discriminator_batch')
         
        reward_labels = tf.reshape(tf.contrib.seq2seq.tile_batch(batch_reward_labels, rebuild_max_time), [-1, rebuild_max_time])
        
#        """
#            question_length : shape = [rebuild_batch_size]
#            response_length : shape = [rebuild_batch_size]
#            question_ids : shape = [rebuild_batch_size, rebuild_max_time]
#            response_ids : shape = [rebuild_batch_size, rebuild_max_time]
#            question_logits : shape = [rebuild_batch_size, rebuild_max_time, logits_size]
#            response_logits : shape = [rebuild_batch_size, rebuild_max_time, logits_size]
#        """
#        with tf.variable_scope('Discriminator') as d_scope:
#            def get_discriminator_reward(batch):
#                """
#                current_question_length : shape = []
#                current_response_length : shape = []
#                current_question_ids : shape = [rebuild_max_time]
#                current_response_ids : shape = [rebuild_max_time]
#                current_question_logits : shape = [rebuild_max_time, logits_size]
#                current_response_logits : shape = [rebuild_max_time, logits_size]
#                """
#                current_question_length = question_length[batch]
#                current_response_length = response_length[batch]
#                current_question_ids = question_ids[batch]
#                current_response_ids = response_ids[batch]
#                current_question_logits = question_logits[batch]
#                current_response_logits = response_logits[batch]
#                
#                question_slice_indices = tf.where(tf.equal(current_question_ids, hparams.eos_id))
#                question_slice_end = tf.cast(tf.scan(lambda y,x: y+x, question_slice_indices), tf.int32)
#                response_slice_indices = tf.where(tf.equal(current_response_ids, hparams.eos_id))
#                response_slice_end = tf.cast(tf.scan(lambda y,x: y+x, response_slice_indices), tf.int32)
#                
#                def rebuild_sentence(batch):
#                    
#                    
#                = tf.while_loop(lambda size, *_: tf.less(size, size(question_slice_indices)), 
#                                lambda size, rebuild_sentence_length, rebuild_sentence_id, rebuild_sentence_logits: 
#                                    (tf.add(size, 1),
#                                     tf.concat([rebuild_sentence, tf.expand_dims(tf.slice(question_slice_indices), axis=0)], axis=0),
#                                     loop_vars=[tf.constant(0), ],
#                                    shape_invariants=[tf.TensorShape([]), tf.TensorShape([None,None])],
#                             name='get_discriminator_reward')
#                
#                
#                response_slice_indices = tf.where(tf.equal(current_response_ids, hparams.eos_id))
#                
#                current_question_length
#                
#                with tf.variable_scope('Encoder_question') as scope:
#                    ques_output, ques_state = self.build_encoder(
#                            hparams, question_ids, embedding=self.embedding,
#                            length=question_length, time_major=False, scope=scope)
#                    
#                    
#                    
#         
#                return tf.add(batch, 1)
#            _, _, question_ids, response_ids, question_logits, response_logits\
#             = tf.while_loop(lambda batch, *_: tf.less(batch, rebuild_batch_size), get_discriminator_reward,
#                             loop_vars=[batch],
#                             shape_invariants=[batch.get_shape()],
#                             name='get_discriminator_reward')
         
        with tf.variable_scope('Discriminator') as d_scope:
            with tf.variable_scope('Encoder_question') as scope:
                    ques_output, ques_state = self.build_encoder(
                            hparams, question_ids, embedding=self.embedding,
                            length=question_length, time_major=False, scope=scope)
            #Build Response Encoder
            with tf.variable_scope('Encoder_response') as scope:
                resp_output, resp_state = self.build_encoder(
                        hparams, response_ids, initial_state=ques_state, embedding=self.embedding,
                        length=response_length, time_major=False, scope=scope)
                reward_logits = tf.nn.softmax(tf.layers.dense(resp_output, 2))
            self.discriminator_params = d_scope.trainable_variables()
            
        sample_indices = tf.squeeze(tf.where(tf.equal(tf.squeeze(tf.slice(reward_labels, [0,0], [-1,1])), 0)))
        target_indices = tf.squeeze(tf.where(tf.not_equal(tf.squeeze(tf.slice(reward_labels, [0,0], [-1,1])), 0)))
        
        self.sample_response_length = tf.gather(response_length, sample_indices)
        self.target_response_length = tf.gather(response_length, target_indices)
        self.sample_response_ids = tf.gather(response_ids, sample_indices)
        self.target_response_ids = tf.gather(response_ids, target_indices)
        self.sample_response_logits = tf.gather(response_logits, sample_indices)
        self.target_response_logits = tf.gather(response_logits, target_indices)
        self.sample_reward_labels = tf.gather(reward_labels, sample_indices)
        self.target_reward_labels = tf.gather(reward_labels, target_indices)
        self.sample_reward_logits = tf.gather(reward_logits, sample_indices)
        self.target_reward_logits = tf.gather(reward_logits, target_indices)
        
        self.discriminator_reward_labels = reward_labels
        self.discriminator_reward_logits = reward_logits
        self.discriminator_response_length = response_length
        self.discriminator_response_ids = response_ids
            
        self.question_length, self.question_ids, self.question_logits = question_length, question_ids, question_logits
        
        return self.sample_response_length, self.sample_response_ids, self.sample_response_logits, self.sample_reward_labels, self.sample_reward_logits
                
    def build_encoder(self, hparams, source, initial_state = None, embedding = None, length = None, time_major=False, scope = None):
        """Create encoder."""
        if embedding != None:
            source = tf.nn.embedding_lookup(self.embedding, source)
        cell = self.create_rnn_cell(hparams, hparams.num_layers, hparams.num_units)
        
#        cell_fw = self.create_rnn_cell(hparams, hparams.num_layers, hparams.num_units)
#        cell_bw = self.create_rnn_cell(hparams, hparams.num_layers, hparams.num_units)
#        initial_states_fw = [cell_fw.zero_state(self.batch_size, tf.float32) for cell_fw in cells_fw]
#        initial_states_bw = [cell_bw.zero_state(self.batch_size, tf.float32) for cell_bw in cells_bw]
#        outputs, _, _ =　tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
#                                                                    initial_states_fw=initial_states_fw,
#                                                                    initial_states_bw=initial_states_bw, dtype=tf.float32)
        
        
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell, source, initial_state=initial_state, dtype=scope.dtype, sequence_length=length, time_major=time_major, scope=scope)
        return encoder_outputs, encoder_state
    
    def build_latent_encoder(self, hparams, passing_flag, latent_state_input, state_input, scope, cell=None):
        if cell == None:
            cell = self.create_rnn_cell(hparams, hparams.num_layers, hparams.num_units)
        if passing_flag != None:
            initial_state = tf.cond(passing_flag, lambda: latent_state_input, lambda: cell.zero_state(self.batch_size, dtype=tf.float32))
        else:
            initial_state = latent_state_input
        latent_output, latent_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(state_input, axis=1), initial_state=initial_state, dtype=scope.dtype)
        response_flag_logits = tf.nn.softmax(tf.layers.dense(latent_output, 2))
        return response_flag_logits, latent_state, cell
    
    def build_emotion_transfer(self, hparams, passing_flag, emotion_state_input, state_input, scope):
        cell = self.create_rnn_cell(hparams, hparams.num_layers, hparams.num_emotion_units)
        if passing_flag != None:
            initial_state = tf.cond(passing_flag, lambda: emotion_state_input, lambda: cell.zero_state(self.batch_size, dtype=tf.float32))
        else:
            initial_state = emotion_state_input
        state_input = tf.layers.dense(state_input, hparams.num_emotion_units)
        emotion_output, emotion_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(state_input, axis=1), initial_state=initial_state, dtype=scope.dtype)
        emotion_output = tf.layers.dense(emotion_output, hparams.num_emotion_units*2, activation=tf.nn.leaky_relu)
        emotion_output = tf.layers.dense(emotion_output, hparams.num_emotion_units, activation=tf.nn.leaky_relu)
        emotion_output_logits = tf.layers.dense(emotion_output, self.emotion_size, name="transfer_emotion_classifier")
        return emotion_output_logits, emotion_state
    
    def build_decoder(self, hparams, sentence_target_id, sentence_input_length, sentence_target_length, encoder_outputs, encoder_state, latent_state, emotion_output):
        """Create decoder."""
        eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.eos_token)), tf.int32, name = 'eos_id')
        emotion_id = tf.cast(self.vocab_table.lookup(emotion_output), tf.int32, name = 'emotion_id')
        start_tokens = tf.reshape(emotion_id, [-1])
        # maximum_iteration: The maximum decoding steps.
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
        else:
            max_encoder_length = tf.reduce_max(sentence_input_length)
            maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * 2.0))
            
        with tf.variable_scope("output_projection", reuse=tf.AUTO_REUSE):
            output_layer = layers_core.Dense(self.vocab_size, use_bias=False, name="output_projection")
            
        cell, init_state = self.build_decoder_cell(hparams, sentence_input_length, encoder_outputs, latent_state)
        
        # Training
        if self.training:
            max_time = tf.shape(sentence_target_id)[1]
            with tf.name_scope("teacher_forcing_decoder"):
                decoder_input = tf.concat([tf.expand_dims(start_tokens, axis=1), sentence_target_id], axis = 1)
                decoder_emb_inp = tf.nn.embedding_lookup(self.embedding, decoder_input)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, sentence_target_length)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, init_state, output_layer = output_layer)
                decoder_outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory = True)
                
                decoder_outputs_logit = decoder_outputs.rnn_output
                decoder_outputs_id = decoder_outputs.sample_id
                decoder_state = decoder_state.cell_state
            
            with tf.name_scope("sample_decoder"):
                helper_sample = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding, start_tokens=start_tokens, end_token=eos_id)
                decoder_sample = tf.contrib.seq2seq.BasicDecoder(cell, helper_sample, init_state, output_layer = output_layer)
                output_sample, output_sample_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder_sample, swap_memory=True, maximum_iterations = max_time)
            
                output_sample_logit = output_sample.rnn_output
                output_sample_id = output_sample.sample_id
                output_sample_state = output_sample_state.cell_state
            
            
#            if self.mode == 'adv-train':
#                decoder_state = output_sample_state
#            else:
            decoder_state = decoder_state
            
            
#            decoder_state = output_sample_state
    
        # Inference
        else:
            if hparams.beam_width > 0:
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell, embedding=self.embedding, start_tokens=start_tokens, end_token=eos_id,
                    initial_state=init_state, beam_width=hparams.beam_width, 
                    output_layer=output_layer, length_penalty_weight=hparams.length_penalty_weight)
            else:
#                    helper = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding, start_tokens, eos_id)
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embedding, start_tokens, eos_id)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, init_state, output_layer=output_layer)
            # Dynamic decoding
            decoder_outputs, decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=maximum_iterations, swap_memory=True)
                
            if hparams.beam_width > 0:
                decoder_state = tuple([tf.squeeze(tf.slice(
                        decoder_state.cell_state.cell_state[index],
                        [tf.shape(decoder_state.cell_state.cell_state[index])[0]-1,0,0],[1,1,-1]),[0]) for index in range(hparams.num_layers)])
                logits = tf.no_op()
                sample_id = tf.slice(tf.transpose(decoder_outputs.predicted_ids, [0,2,1]), [0,0,0], [-1,-1,-1])
            else:
                decoder_state = decoder_state.cell_state
                logits = decoder_outputs.rnn_output
                sample_id = decoder_outputs.sample_id
                
            return logits, sample_id, decoder_state, logits, sample_id
        return decoder_outputs_logit, decoder_outputs_id, decoder_state, output_sample_logit, output_sample_id
                    
    def build_decoder_cell(self, hparams, source_sequence_length, encoder_outputs, input_init_state):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        num_units = hparams.num_units
        beam_width = hparams.beam_width
        if not self.training and beam_width > 0:
            with tf.name_scope('Tile_batch'):
                source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length,multiplier=beam_width)
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                input_init_state = tf.contrib.seq2seq.tile_batch(input_init_state,multiplier=beam_width)
            beam_size = self.batch_size * beam_width
        else:
            beam_size = self.batch_size
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, encoder_outputs, memory_sequence_length=source_sequence_length)
        cell = self.create_rnn_cell(hparams, hparams.num_layers, hparams.num_units)
        # Only generate alignment in greedy INFER mode.
        alignment_history = (not self.training and beam_width == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism, attention_layer_size=num_units, alignment_history=alignment_history, name="attention")
        init_state = cell.zero_state(beam_size, tf.float32).clone(cell_state=input_init_state) if hparams.pass_hidden_state else cell.zero_state(beam_size, tf.float32)
        return cell, init_state
    
    def create_rnn_cell(self, hparams, num_layers, num_units):
        """Create multi-layer RNN cell."""
        cell_list = []
        for i in range(num_layers):
            #Create a single RNN cell
            if hparams.rnn_cell_type == 'LSTM':
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True)
            else:
                single_cell = tf.contrib.rnn.GRUCell(num_units)
            if hparams.keep_prob < 1.0:
                single_cell = tf.contrib.rnn.DropoutWrapper(cell = single_cell, input_keep_prob = hparams.keep_prob)
            cell_list.append(single_cell)

        if len(cell_list) == 1:
            return cell_list[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cell_list)