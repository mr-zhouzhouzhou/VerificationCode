import tensorflow as tf


acc_img=[[ True , True  ,True  ,True],
         [ True , True  ,True  ,True],
         [ True,  True,  True  ,True],
         [ True , True  ,True  ,True],
         [ True,  True,  True  ,True],
         [ True , True  ,True  ,True],
         [ True,  True,  True  ,True],
         [ True , False , True , True],
         [ True , True , True , True],
         [ True , True , True , True]]
ca=tf.cast(acc_img, tf.float32)
acc_img=tf.reduce_min(ca,axis=1)
mean=tf.reduce_mean(acc_img)

with tf.Session() as sess:
    print(sess.run([ca,mean,acc_img]))