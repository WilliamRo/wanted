import tensorflow as tf
import console


console.suppress_logging()
console.start('00 in tf.while_loop')

u, v = tf.constant(2.), tf.constant(4.)
elems = tf.constant([1., 2. ,3.])

def cond(i, _):
  return i < elems.shape[0]

def body_tr(i, r):

  @tf.custom_gradient
  def update_r(r, u, x, v):
    def grad(dy):
      # dr_r = dy * u
      dr_r = tf.cond(u < 0, lambda: dy * u, lambda: dy * u * 0)
      return dr_r, dy * r, dy * v, dy * x
    r = r * u + x * v
    return r, grad

  r = update_r(r, u, elems[i], v)
  return i + 1, r

r = tf.while_loop(cond, body_tr, (0, 0.))

"""
r = 0
r1 = 0 * 2 + 1 * 4 = 4
r2 = 4 * 2 + 2 * 4 = 16
r3 = 12 * 2 + 3 * 4 = 44

r3 = r2 * u + x[3] * v
dr3/dv = x[3] + u * dr2/dv
       = x[3] + u * (x[2] + u * dr1/dv)
       = x[3] + u * (x[2] + u * 1)  
       = x[3] + u * (2 + 2)  
       = 3 + 2 * (2 + 2) = 11
       
If we want to truncate the gradient, say, let dr2/dv = 0
Thus dr3/dv = x[3] + u * dr2/dv = x[3] = 3.0
"""
dr3_dv = tf.gradients(r, v)


with tf.Session() as sess:
  val_i, val_r = sess.run(r)
  print('>> r = {}'.format(val_r))

  val_dr3_dv = sess.run(dr3_dv)
  print('>> dr3_dv = {}'.format(val_dr3_dv))

console.end()
