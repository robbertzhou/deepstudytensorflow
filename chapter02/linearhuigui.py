import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()
train_x = np.linspace(-1,1,100)
train_y = 2 *train_x +np.random.randn(*train_x.shape) * 0.3
# plt.plot(train_x,train_y,'ro',label='Original data')
# plt.legend()
# plt.show()

#正向模型
X = tf.compat.v1.placeholder("float")
Y = tf.compat.v1.placeholder("float")
W = tf.Variable(tf.compat.v1.random_normal([1]),name="weight")
b = tf.Variable(tf.zeros([1]),name="bias")
z = tf.multiply(X,W) + b

globa_step = tf.Variable(0,name="global_step",trainable=False)
cost = tf.reduce_mean(tf.square(Y-z))
learning_ate = 0.01
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_ate).minimize(cost,globa_step)

init = tf.compat.v1.global_variables_initializer()
#定义学习参数
training_epochs = 20
display_step = 2

savedir = "log/"
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),max_to_keep=1)
plotdata = {"batchsize":[],"loss":[]}
def moving_average(a,w = 10):
    if(len(a) < w):
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]


#训练模型
with tf.compat.v1.Session() as sess:
    sess.run(init)
    kpt = tf.train.latest_checkpoint(savedir)
    if kpt != None:
        saver.restore(sess,kpt)
    #向模型输入数据
    while globa_step.eval() / len(train_x) < training_epochs:
        step = int(globa_step.eval()/len(train_x))
        for (x,y) in zip(train_x,train_y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            loss = sess.run(cost,feed_dict={X:train_x,Y:train_y})
        plotdata["batchsize"].append(globa_step.eval())
        plotdata["loss"].append(loss)

    print("cost=",sess.run(cost,feed_dict={X:train_x,Y:train_y}),"w=",sess.run(W),"b=",sess.run(b))
    #显示模型
    plt.plot(train_x,train_y,'ro','Original data')
    plt.plot(train_x,sess.run(W)*train_x + sess.run(b),label="Fitted line")
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"],'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()
