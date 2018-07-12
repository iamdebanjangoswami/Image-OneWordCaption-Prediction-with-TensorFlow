import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split 
import tensorflow as tf
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/fashion-mnist_train.csv')
data.info()

data.head()

cols =  data.columns
cols

X = data[cols[1:]].values
Y = data['label']

Y.value_counts().plot('bar');

Y = pd.get_dummies(Y).values.tolist()
test = pd.read_csv('../input/fashion-mnist_test.csv')
test = test[cols[1:]].values

image_size = 28
label_size = 10
hidden_size =1024
learning_rate =0.001
batch_size =256
split = 0.2

trainX, validationX, trainY, validationY = train_test_split(X,Y,test_size=split,random_state=42)
validationX.shape

#Define Placeholder conv2d
x_input = tf.placeholder(tf.float32, shape=(None,image_size*image_size))
y_labels = tf.placeholder(tf.float32, shape=(None,label_size))
dropout = tf.placeholder(tf.bool)

x_image = tf.reshape(x_input,[-1,image_size,image_size,1])

conv1 = tf.layers.conv2d(inputs=x_image, kernel_size=[5,5], filters=32, padding='same',activation= tf.nn.elu)
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

conv2 = tf.layers.conv2d(inputs=pool1, kernel_size= [5,5], filters= 64, padding='same',activation= tf.nn.elu)
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])

hidden = tf.layers.dense(inputs=flatten,units=hidden_size,activation=tf.nn.elu)
dropouts = tf.layers.dropout(inputs=hidden,rate= 0.65,training= dropout)

y_output = tf.layers.dense(inputs=dropouts,units=label_size)


loss = tf.reduce_min(tf.losses.softmax_cross_entropy(logits=y_output,onehot_labels=y_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
#optimizer = tf.train.AdamOptimizer(0.03).minimize(loss),global_step=tf.train.get_global_step()
prediction=tf.argmax(y_output,1)

correct_prediction = tf.equal(prediction, tf.argmax(y_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    

sess =  tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

l = len(trainX) - len(trainX)%batch_size

for i in range(0,l,batch_size):
    #input_batch, labels_batch = sess.run(data.repeat().batch(512).make_one_shot_iterator().get_next())
    #print(input_batch.shape,labels_batch.shape,type(input_batch),type(labels_batch),input_batch[0])
    #print(labels_batch.dtype,labels_batch[0])
    feed_dict = {x_input: trainX[i:i+batch_size], y_labels:trainY[i:i+batch_size], dropout: True}
    feed_dict_test = {x_input: trainX[i:i+batch_size], y_labels: trainY[i:i+batch_size], dropout: False}
    
    #plt.imshow(np.reshape(input_batch[5],(28,28)),cmap='gray');
    if (i//batch_size)%50 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict_test)
        print("Step %d, training batch accuracy %.3f"%(i//batch_size, train_accuracy))
    sess.run(optimizer,feed_dict=feed_dict)
    
print("The end of training!")

print("Validation accuracy: %.3f"%accuracy.eval(feed_dict={x_input: validationX[:200], y_labels: validationY[:200], dropout: False}))

predictions = sess.run(prediction, feed_dict={x_input:validationX[:512] ,dropout: False})

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
classes

n = 32
fig = plt.figure(figsize=(10,10))
for i,p in enumerate(predictions[:n]):
    ax = fig.add_subplot(8,8,i+1)
    plt.title(f'''\n\n {classes[p]}''')
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax.imshow(np.reshape(validationX[:512][i],(28,28)),cmap='gray');
    
    v = pd.Series([validationY[i].index(1) for i in range(512)])
#p = pd.Series([classes[predictions[i]] for i in range(512)])

# Create a confusion matrix on training data.
plt.figure(figsize=(10,10))
cm = tf.confusion_matrix(v.values,predictions)
cm_out = sess.run(cm)

# Normalize the confusion matrix so that each row sums to 1.
cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_out, annot=True, xticklabels=classes, yticklabels=classes);
plt.xlabel("Predicted");
plt.ylabel("True");
plt.title('Confusion Matrix Fashion Dress');

test_preds = []
test_preds.extend(sess.run(prediction, feed_dict={x_input:test,dropout: False}))
print(len(test_preds))
sess.close()
test_preds[:10]

fig = plt.figure(figsize=(10,10))
tv = test[100:116]
for i,p in enumerate(test_preds[100:116]):
    ax = fig.add_subplot(4,4,i+1)
    plt.title(f''' \n {classes[p]}''')
    ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax.imshow(np.reshape(tv[i],(28,28)),cmap='gray');



