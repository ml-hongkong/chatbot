# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pprint
import _pickle as cPickle
from model import DialogueModel
from utils import TextLoader, UNK_ID, PAD_ID
from glob import glob

checkpoint = "/tmp/model.ckpt"

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logdir", "log", "Log directory [log]")
flags.DEFINE_float("temperature", 0.5, "temperature")
FLAGS = flags.FLAGS

def main(_):
  config = cPickle.load(open(FLAGS.logdir + "/hyperparams.pkl", 'rb'))
  pp.pprint(config)

  try:
    # pre-trained chars embedding
    emb = np.load("./data/emb.npy")
    chars = cPickle.load(open("./data/vocab.pkl", 'rb'))
    vocab_size, emb_size = np.shape(emb)
    data_loader = TextLoader('./data', 1, chars)
  except Exception:
    data_loader = TextLoader('./data', 1)
    emb_size = config["emb_size"]
    vocab_size = data_loader.vocab_size

  checkpoint = FLAGS.checkpoint + '/model.ckpt'

  model = DialogueModel(batch_size=1, max_seq_length=data_loader.seq_length,
                        vocab_size=vocab_size, pad_token_id=0, unk_token_id=UNK_ID,
                        emb_size=emb_size, memory_size=config["memory_size"],
                        keep_prob=config["keep_prob"], learning_rate=config["learning_rate"],
                        grad_clip=config["grad_clip"], temperature=config["temperature"], infer=True)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(init)

    if len(glob(checkpoint + "*")) > 0:
      saver.restore(sess, checkpoint)
    else:
      print("No model found!")
      return

    ## -- debug --
    #np.set_printoptions(threshold=np.inf)
    #for v in tf.trainable_variables():
    #  print(v.name)
    #  print(sess.run(v))
    #  print()
    #return

    while True:
      try:
        input_ = input('in> ')
      except EOFError:
        print("\nBye!")
        break

      input_ids, input_len = data_loader.parse_input(input_)

      feed = {
        model.input_data: np.expand_dims(input_ids, 0),
        model.input_lengths: [input_len]
      }

      output_ids, state = sess.run([model.output_ids, model.final_state], feed_dict=feed)

      print(data_loader.compose_output(output_ids[0]))

if __name__ == "__main__":
  tf.app.run()
