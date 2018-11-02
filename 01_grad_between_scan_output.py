import tensorflow as tf
import console


console.suppress_logging()
console.start('Gradients between the outputs of tf.scan')

u, v, w = tf.constant(2.), tf.constant(4.), tf.constant(10.)
elems = tf.constant([1., 2. ,3.])

def fn(pre_outputs, x):
  # Get previous s. pre_outputs = (pre_y, pre_s)
  pre_s = pre_outputs[1]

  new_s = pre_s * u + x * v
  y = w * new_s
  return y, new_s

ys, ss = tf.scan(fn, elems, initializer=(0., 0.))
last_y, last_s = ys[-1], ss[-1]
dy_ds = tf.gradients(last_y, last_s)

# TODO: the problem is dy_ds = [None]

with tf.Session() as sess:
  evaluate = lambda tensor: sess.run(tensor)
  show = lambda name, tensor: print('>> {} = {}'.format(
    name, evaluate(tensor)))

  show('val_ss', ss)
  show('val_ys', ys)
  show('last_s', last_s)
  show('last_y', last_y)

  show('dy/ds', dy_ds)

console.end()


"""
new_s = pre_s * u + x * v
new_s = pre_s * 2 + x * 4

s = 0
s1 = 0 * 2 + 1 * 4 = 4
s2 = 4 * 2 + 2 * 4 = 16
s3 = 12 * 2 + 3 * 4 = 44

r3 = r2 * u + x[3] * v
dr3/dv = x[3] + u * dr2/dv
       = x[3] + u * (x[2] + u * dr1/dv)
       = x[3] + u * (x[2] + u * 1)  
       = x[3] + u * (2 + 2)  
       = 3 + 2 * (2 + 2) = 11
"""
