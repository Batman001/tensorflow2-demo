# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


# 划分数据集
# 60%(15,000)用于训练 40%（10,000）用于验证(validation)
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, val_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

# 数据格式
train_examples_batch, train_label_batch = next(iter(train_data.batch(10)))
print("数据的样式为：")
print(train_examples_batch)
print(train_label_batch)


# 文本嵌入过程
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])

# build the model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.summary()

# 优化器和损失函数
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

# train the model
history = model.fit(train_data.shuffle(1000).batch(512), epochs=20,
                    validation_data=val_data.batch(512), verbose=1)

# evaluate the model
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
    print("在测试集验证的结果为：\n %s: %.3f" %(name, value))


