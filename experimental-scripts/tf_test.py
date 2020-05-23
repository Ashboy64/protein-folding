import tensorflow as tf
import tensorflow_probability as tfp

# x = tf.Variable(10*tf.random.uniform(shape=[3]))
# print(x)
#
# loss_fn = lambda: tf.math.reduce_sum(x**2)
# losses = tfp.math.minimize(loss_fn,
#                            num_steps=100,
#                            optimizer=tf.optimizers.Adam(learning_rate=0.1))
#
# # In TF2/eager mode, the optimization runs immediately.
# print("optimized value is {} with loss {}".format(x, losses[-1]))

x = tf.Variable([1.0, 0.0, 7.0])
y = tf.Variable([1.0, 0.0, 4.0])

z = tf.reduce_sum(x * y)

if tf.cond(tf.math.equal(z, tf.constant(0.0)), lambda: True, lambda: False):
    print("Here")

