import tensorflow as tf

i_s=4
k_s=3
k_d=i_d=1
k_n=1
pad=1

f=open("in.txt")
f=f.readlines()
# w = f[1].split('\n')[0].split(',')[:-1]
# w = [float(i) for i in w]
# print(w)
w = [1 for i in range(0,k_s*k_s*k_d*k_n)]
w = tf.constant(w, shape=[k_s,k_s,k_d,k_n], dtype=tf.float32)
inputs = f[0].split('\n')[0].split(',')[:-1]
inputs = [float(i) for i in inputs]

inputs = [i for i in range(0,i_s*i_s*i_d)]
# print(inputs)
inputs = tf.constant(inputs, shape=[1,i_s,i_s,i_d], dtype=tf.float32)



##########convolution
ans = tf.nn.conv2d(inputs,w,[1,pad,pad,1],'VALID')
ans = tf.contrib.layers.flatten(ans)

#######maxpool
# ans = tf.nn.avg_pool(inputs,ksize=[1,k_s,k_s,1],strides=[1,pad,pad,1], padding='VALID')

sess = tf.InteractiveSession()

print(sess.run(ans))


