import tensorflow as tf
import console


console.suppress_logging()
console.start('00 in tf.scan')

u, v = tf.constant(2.), tf.constant(4.)
elems = tf.constant([1., 2. ,3.])
tr_flag = tf.placeholder(dtype=tf.bool)


def fn(a, x):

  @tf.custom_gradient
  def update_r(r, u, x, v):
    def grad(dy):
      dr_r = tf.cond(tr_flag, lambda: dy * u * 0, lambda: dy * u)
      return dr_r, dy * r, dy * v, dy * x
    r = r * u + x * v
    return r, grad

  r = update_r(a, u, x, v)
  return r

rs = tf.scan(fn, elems, initializer=0.)
r = rs[-1]
dr3_dv = tf.gradients(r, v)

truncation_switch = False

with tf.Session() as sess:
  val_rs = sess.run(rs)
  print('>> rs = {}'.format(val_rs))

  val_dr3_dv = sess.run(dr3_dv, feed_dict={tr_flag: truncation_switch})
  print('>> dr3_dv = {}'.format(val_dr3_dv))

console.end()
