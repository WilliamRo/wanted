import tensorflow as tf
import console


console.suppress_logging()
console.start('Gradients (of tf.scan output to value outside) inside tf.scan')

v = tf.constant([1., 2, 3])
w = tf.constant(10.)
mascot = tf.placeholder(tf.float32)

wv = tf.scan(lambda _, x: w * x, v, initializer=mascot)

def fn(_, x):
  # TODO: since x is generated by another scan operator, and w is defined
  #       outside that scan op, the gradients op will produce an error
  #       saying 'Cannot use `../f_count_1` as input to `../f_count` since they
  #       are in different while loops.'
  grad = tf.gradients(x, w)[0]
  return grad

grads = tf.scan(fn, wv, initializer=mascot)

with tf.Session() as sess:
  # TODO: what we want here is to see '>> grads = [10., 10., 10.]'
  console.eval_show(grads, name='grads')

console.end()




