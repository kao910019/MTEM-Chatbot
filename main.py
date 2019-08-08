# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2

from Model.emotion_seqgan import EMSeqGAN

def Execute_Trainer(target = ""):
    
    config_proto = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
    
    config_proto.gpu_options.allow_growth = True
    
    # When "Already exists: Resource ...... TmpVar" error happen.
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
        
    with tf.Session(target = target, config = config_proto) as sess:
        model = EMSeqGAN(sess, training = True)
        model.train()
    
if __name__ == "__main__":
    Execute_Trainer()
    
    
        
        
