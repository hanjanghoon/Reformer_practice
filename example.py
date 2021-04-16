from Reformers import Reformer_Model
import torch
import tensorflow as tf

model_tf = Reformer_Model(
    vocab_size= 20000,    #vocab
    emb_size = 512,
    depth = 1,            #layer   수
    max_seq_len = 1024,
    heads = 8,
    causal = True,        # 논문의 causal 위함.
    bucket_size = 16,     
    num_hash = 4,         # multi-round hash
    ff_chunks = 256       # chunked feed forward 위함

)

# 모듈 테스트 용 코드입니다.. 직접 인코더 디코더 모델을 구현하지 못했습니다ㅜㅜ

x = tf.random.uniform((4, 1024))
y = tf.random.uniform(
    (4,1024), minval=1, maxval=2, dtype=tf.dtypes.int32, seed=None, name=None
)
'''
optimizer = tf.keras.optimizers.Adam(learning_rate=6.25e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

#tf.config.experimental_run_functions_eagerly(True)
model_tf.compile(optimizer=optimizer, loss=loss, metrics=[metric])
#tf.executing_eagerly()

history = model_tf.fit(x, y, 
                        epochs=2, 
                        batch_size=1,
                        )
'''
#x = tf.random.uniform((2, 8192))
#model_tf.build(input_shape=(1,8192))
#model_tf.summary()
#y = model_tf(x)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_optimizer=(tf.keras.optimizers.Adam(learning_rate=1e-3))

epochs=10
for i in range(epochs):
    loss = model_tf.train_step(x,y,loss_object,train_loss,accuracy_object,train_optimizer)
    template = '에포크: {}, 손실: {}, 정확도: {}'
    print (template.format(i,train_loss.result(),accuracy_object.result()*100))

