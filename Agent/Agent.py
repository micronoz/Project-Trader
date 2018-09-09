
# coding: utf-8

# In[22]:


import tensorflow as tf
import numpy as np
from trade_sim import Market
import tensorflow.contrib as contrib
import os
from multiprocessing import Process, SimpleQueue
import random


# In[30]:


class Agent:
    def __init__(self, numberOfCurrencies, timeFrame, sess, initialPortfolio=10000.0):
        self._s = sess
        self.inputT = tf.placeholder(shape=[None, numberOfCurrencies, timeFrame, 3], dtype=tf.float32)
        self.conv1 = tf.layers.conv2d(inputs=self.inputT, filters=2, kernel_size=[1,3], activation=tf.nn.relu)
      #  self.conv1 = tf.nn.depthwise_conv2d(self.inputT, [1,3,3,4], [1,1,1,1], 'SAME')
        #self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=150, kernel_size=[1,8], activation=tf.nn.relu)
        #self.conv3 = tf.layers.conv2d(inputs = self.conv2, filters=200, kernel_size=[1,41], activation=tf.nn.relu)
        #self.conv4 = tf.layers.conv2d(inputs=self.conv3, filters=30, kernel_size=[1,1] , activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=30, kernel_size=[1,48], activation=tf.nn.relu)
        self.final = tf.layers.dense(self.conv2, 20000, activation=tf.nn.relu)
        self.hidden0 = tf.layers.dense(self.final, 1000, activation=tf.nn.relu)
        self.hidden = tf.layers.dense(self.hidden0, 2000, activation=tf.nn.relu)
        self.final2 = tf.layers.dense(self.hidden, 1)
        self._allocate = tf.nn.softmax(self.final2, axis=1)
        
        self.priceChanges = tf.placeholder(shape=[None, numberOfCurrencies, 1], dtype=tf.float32)
        
        #self.loss = -tf.matmul(tf.matrix_transpose(tf.nn.leaky_relu(tf.log(self.priceChanges), alpha=10)),tf.reshape(self._allocate, [-1, numberOfCurrencies, 1]))
        self.averageLoss = tf.reduce_mean(tf.matmul(tf.matrix_transpose(self.priceChanges), 
                                             tf.scalar_mul(tf.constant(initialPortfolio), 
                                               tf.reshape(self._allocate, [-1, numberOfCurrencies, 1]))))

        self.loss = tf.exp(tf.reduce_sum(tf.multiply(-tf.log(self.priceChanges), tf.reshape(self._allocate, [tf.shape(self._allocate)[0], numberOfCurrencies, 1]))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self._train = self.optimizer.minimize(self.loss)
        
    def act(self, observation):
        return self._s.run(self._allocate, feed_dict={self.inputT: observation})
    
    def train_step(self, obs, prices):
        batch_feed = {self.inputT : obs,
                     self.priceChanges: prices
                     }
        _, lossValue = self._s.run([self._train, self.averageLoss], feed_dict=batch_feed)
        return lossValue


# In[31]:


def importData(simulator):
    testSim = simulator
    q = SimpleQueue()
    jobs = []
    PERIOD_SIZE = 50
    BATCH_SIZE = 100
    BATCH_COUNT = 1
    BATCH_OFFSET = 100
    dates = testSim.getAllDates()
    index = list(range(BATCH_COUNT))
    feed = []
    threads = 16

    running = False
    count = 0
    while 1:
        if count < threads:
            for i in random.sample(index, threads-count if len(index) >= threads-count else len(index)):
                p = Process(target=testSim.processTimePeriod, args=(q, PERIOD_SIZE, dates, BATCH_SIZE * (i + BATCH_OFFSET) + PERIOD_SIZE, BATCH_SIZE))
                jobs.append(p)
                p.start() 
                index.remove(i)
        count = 0
        for p in jobs:
            if not p.is_alive():
                p.terminate()
                jobs.remove(p)
            else:
                count += 1
        while not q.empty():
            print('Getting')
            feed.append(q.get())
        if count == 0 and len(index) == 0: 
            break
    return feed


# In[32]:


def main():
    testSim = Market(['EUR','USD'], os.path.abspath('/mnt/disks/ProcessedData'))
    seeds = [3, 5, 7]
    with tf.Session() as sess:
        tf.set_random_seed(seeds[1])
        test1 = Agent(len(testSim.currencies), 50, sess)
        
        feed = importData(testSim)
        sess.run(tf.global_variables_initializer())
        prices = []
        batches = []
        
        
            
        for episode in range(10000):
            print("Episode: {}".format(episode))
            index = list(range(len(feed)))
            loss = 0
            count = len(feed)
            while len(index) != 0:
                for i in random.sample(index, 1):
                    with tf.device('/gpu:0'):
                        loss += test1.train_step(feed[i][0], feed[i][1])
                    index.remove(i)
            print(loss/count)
            #print("Loss: {}", loss/count)
 
#         for index in range(len(allDates)):
#             obs.append(testSim.processTimePeriod(50, allDates[index]))
#             prices.append(testSim.processTimePeriod(50, allDates[index]))
            
#             i = index
#             while obsSingle is None and pricesSingle is None:
#                 obsSingle = testSim.processTimePeriod(50, allDates[i])
#                 pricesSingle = testSim.processTimePeriod(50, allDates[i])
#                 i += 1
#         for _ in range(20000):
#             test1.train_step(obs, prices)
#         print(test1.act(obs))
#         print(prices)
#         print(test1._s.run(test1.loss, feed_dict={test1.inputT: obs, test1.priceChanges:prices}))
        


# In[33]:


if __name__ == "__main__":
    main()

