import tensorflow as tf
import numpy as np
def sample_point(k):
    num_points = 100
    #振幅
    amplitude =np.random.uniform(low=1.0, high=5.0)
    #相位
    phase = np.random.uniform(low=0, high=np.pi)

    x = np.linspace(-5, 5, num_points)
    y = amplitude * np.sin(x + phase)
    #抽取k个数据点
    sample = np.random.choice(np.arange(num_points), size=k)
    return (x[sample], y[sample])
tf.reset_default_graph()
num_hidden = 64
num_classes = 1
num_features = 1

X = tf.placeholder(tf.float32, shape=[None,num_features])
Y = tf.placeholder(tf.float32, shape=[None,num_classes])

w1 = tf.Variable(tf.random_uniform([num_features, num_hidden]))
b1 = tf.Variable(tf.random_uniform([num_hidden]))

w2 = tf.Variable(tf.random_uniform([num_hidden, num_classes]))
b2 = tf.Variable(tf.random_uniform([num_classes]))

#1层
z1 = tf.matmul(X, w1) + b1
a1 = tf.nn.tanh(z1)

z2 = tf.matmul(a1, w2) + b2
Yhat = tf.nn.tanh(z2)

#使用均方误差作为损失函数
loss_function = tf.reduce_mean(tf.square(Yhat - Y))
#使用ADAM optimizer
optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss_function)
init = tf.global_variables_initializer()
if __name__ == '__main__':
    num_epochs = 100
    num_samples = 50
    num_tasks = 2
    num_iterations = 10
    mini_batch = 10
    with tf.Session() as sess:
        sess.run(init)
        for e in range(num_epochs):
            for task in range(num_tasks):
                old_w1, old_b1, old_w2, old_b2 =sess.run([w1, b1, w2, b2])

                x_sample, y_sample = sample_point(num_samples)
                for k in range(num_iterations):
                    for i in range(0, num_samples, mini_batch):

                        x_minibatch = x_sample[i:i + mini_batch]
                        y_minibatch = y_sample[i:i + mini_batch]
                        train = sess.run(optimizer, feed_dict={X:x_minibatch.reshape(mini_batch,1),Y:y_minibatch.reshape(mini_batch,1)})
                    new_w1, new_b1, new_w2, new_b2 = sess.run([w1, b1, w2, b2])
                    epsilon = 0.1
                    # 执行元更新，
                    # 即 theta = theta + epsilon * (theta_star - theta)
                    updated_w1 = old_w1 + epsilon * (new_w1 - old_w1)
                    updated_b1 = old_b1 + epsilon * (new_b1 - old_b1)

                    updated_w2 = old_w2 + epsilon * (new_w2 - old_w2)
                    updated_b2 = old_b2 + epsilon * (new_b2 - old_b2)
                    # 使用新参数更新模型参数
                    w1.load(updated_w1, sess)
                    b1.load(updated_b1, sess)

                    w2.load(updated_w2, sess)
                    b2.load(updated_b2, sess)
                    if e%10 == 0:
                        loss = sess.run(loss_function, feed_dict={X:x_sample.reshape(num_samples, 1), Y:y_sample.reshape(num_samples, 1)})
                        print('Epoch {}:Loss:{}\n'.format(e, loss))
                        print('Updated Model Parameter Theta\n')
                        print('Sampling Next Batch of Tasks\n')
                        print('-------------------------------\n')