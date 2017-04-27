import tensorflow as tf
import model
import pprint
import _pickle as cPickle
from glob import glob
import math
import sys
import numpy as np
from utils import TextLoader, UNK_ID
from model import DialogueModel

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("num_epochs", 25, "Epoch to train [25]")
flags.DEFINE_integer("memory_size", 300, "Memory size [300]")
flags.DEFINE_integer("emb_size", 300, "The dimension of embedding matrix [300]")
flags.DEFINE_integer("batch_size", 32, "The size of batch [32]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.001]")
flags.DEFINE_float("keep_prob", 0.5, "Dropout rate [0.5]")
flags.DEFINE_float("grad_clip", 5.0, "Grad clip [5.0]")
flags.DEFINE_integer("temperature", 5, "temperature [5]")
flags.DEFINE_string("checkpoint", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("logdir", "log", "Log directory [log]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(FLAGS.__flags)
  emb = None

  try:
    # pre-trained chars embedding
    emb = np.load("./data/emb.npy")
    chars = cPickle.load(open("./data/vocab.pkl", 'rb'))
    vocab_size, emb_size = np.shape(emb)
    data_loader = TextLoader('./data', FLAGS.batch_size, chars)
  except Exception:
    data_loader = TextLoader('./data', FLAGS.batch_size)
    emb_size = FLAGS.emb_size
    vocab_size = data_loader.vocab_size

  model = DialogueModel(batch_size=FLAGS.batch_size, max_seq_length=data_loader.seq_length,
                        vocab_size=vocab_size, pad_token_id=0, unk_token_id=UNK_ID,
                        emb_size=emb_size, memory_size=FLAGS.memory_size,
                        keep_prob=FLAGS.keep_prob, learning_rate=FLAGS.learning_rate,
                        grad_clip=FLAGS.grad_clip, temperature=FLAGS.temperature,
                        infer=False)

  summaries = tf.summary.merge_all()

  init = tf.global_variables_initializer()

  # save hyper-parameters
  cPickle.dump(FLAGS.__flags, open(FLAGS.logdir + "/hyperparams.pkl", 'wb'))

  checkpoint = FLAGS.checkpoint + '/model.ckpt'
  count = 0

  saver = tf.train.Saver()

  with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    sess.run(init)

    if len(glob(checkpoint + "*")) > 0:
      saver.restore(sess, checkpoint)
      print("Model restored!")
    else:
      # load embedding
      if emb is not None:
        sess.run([], { model.embedding: emb })
      print("Fresh variables!")

    current_step = 0
    count = 0

    for e in range(FLAGS.num_epochs):
      data_loader.reset_batch_pointer()
      state = None

      # iterate by batch
      for _ in range(data_loader.num_batches):
        x, y, input_lengths, output_lengths = data_loader.next_batch()

        if (current_step + 1) % 10 != 0:
          res = model.step(sess, x, y, input_lengths, output_lengths, state)
        else:
          res = model.step(sess, x, y, input_lengths, output_lengths, state, summaries)
          summary_writer.add_summary(res["summary_out"], current_step)
          loss = res["loss"]
          perplexity = np.exp(loss)
          count += 1
          print("{0}/{1}({2}), perplexity {3}".format(current_step + 1,
                                                      FLAGS.num_epochs * data_loader.num_batches,
                                                      e,
                                                      perplexity))
        state = res["final_state"]

        if (current_step + 1) % 2000 == 0:
          count = 0
          summary_writer.flush()
          save_path = saver.save(sess, checkpoint)
          print("Model saved in file:", save_path)

        current_step = tf.train.global_step(sess, model.global_step)

    summary_writer.close()
    save_path = saver.save(sess, checkpoint)
    print("Model saved in file:", save_path)

if __name__ == "__main__":
  tf.app.run()
