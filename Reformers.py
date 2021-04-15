import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Dense, Layer, LayerNormalization, Dropout
import math
 
#######주요 구현.
#########LSH Self-Attention
#########Reversible Residual Connection #직접 backward 연결 중
#########Chunked Feed Forward


def get_index_with_sorting(val, idx, dim=-1):

    idx = tf.broadcast_to(idx, val.shape)
    
    off=tf.range(val.shape[0])*val.shape[1]
    off=tf.reshape(off,[-1,1])
    off=tf.broadcast_to(off, val.shape)

    
    #차원 맞춰주기!
    return tf.sort(val, axis=dim), tf.gather(tf.reshape(idx,[-1]), tf.argsort(val, axis=dim)+off, axis=dim)


def reordering(val, idx, chunknum=None, flag=False):
    batch_size= val.shape[0]
    seq_length = val.shape[1]

    off=tf.range(idx.shape[0])*seq_length
    off=tf.reshape(off,[-1,1])
    off=tf.broadcast_to(off, idx.shape)
    
    output=tf.gather(tf.reshape(val,[-1,val.shape[-1]]), idx+off)
    if flag:
        output=tf.reshape(output, (batch_size, chunknum, -1, output.shape[-1]))


    return output


class Revnet_Layers(tf.keras.Model):
    

    def __init__(self,layers):
       
        super(Revnet_Layers, self).__init__()
        self.r_layers = layers

    def call(self, h, training=True):
        
        for layer in self.r_layers:
            h = layer(h, training=training)
        return h
    #dkw
    def custom_backward(self, x, y, dy, training=True):

        for i in reversed(range(len(self.r_layers))):
            layer = self.r_layers[i]         
            y, dy= layer.custom_backward(y, dy, training=training)

        return dy



class Revnet(tf.keras.Model):

    def __init__(self,f_layer,g_layer):
        super(Revnet, self).__init__()

        self.axis = -1        
        self.f = f_layer
        self.g = g_layer

    def call(self, x, training=True, concat=True):

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
        
        f_x2 = self.f(x2, training=training)
        y1 = f_x2 + x1
        g_y1 = self.g(y1, training=training)
        y2 = g_y1 + x2
        
        output = tf.concat([y1, y2], axis=self.axis)
        return output

    def custom_backward(self, y, dy, training=True):
        dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)
        del dy

        with tf.GradientTape(persistent=True) as tape:
            
            y = tf.identity(y)
            tape.watch(y)
            y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
            del y

            #gy1 역전파.
            gy1 = self.g(y1, training=training)
            #graident target, trinable variable ,output 요게 gradient임.
            #gy1 gradient 계산.
            grads_combined = tape.gradient(gy1, [y1] + self.g.trainable_variables, output_gradients=dy2)
            
            #activation.
            x2 = y2 - gy1
            del y2, gy1
            
            dx1 = dy1 + grads_combined[0]#y1.grad
            del dy1

    
            fx2 = self.f(x2, training=training)
            x1 = y1 - fx2
            del y1, fx2
            #x2.grad 
            grads_combined = tape.gradient(fx2, [x2] + self.f.trainable_variables, output_gradients=dx1)
            #쪼개서 꺼내기.
            dx2 = dy2 + grads_combined[0]

            del tape

        x = tf.concat([x1, x2], axis=self.axis)
        dx = tf.concat([dx1, dx2], axis=self.axis)

        return x, dx


class Feed_Forward(tf.keras.Model):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.linear1 = Dense(emb_size*4,activation='relu')#transformers 참조
        self.linear2 = Dense(emb_size)

    def call(self, inputs):
        output = self.linear1(inputs)
        output = self.linear2(output)
        return output


class LSHAttention(tf.keras.Model):
    def __init__( self, bucket_size , num_hash  ,causal = False):
        super(LSHAttention, self).__init__()
        self.dropout = Dropout(0.1)
        self.causal = causal
        self.num_hash = num_hash #num_hash
        self.bucket_size = bucket_size

    def look_forward(self, x):
            x_forward = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)#한칸 땡김
            return tf.concat([x, x_forward], axis=2)

    def undo_sorting(self,sorted_qkv, sorted_logits, arg_sort_index , undo_sort_index):
            qkv = reordering(sorted_qkv, undo_sort_index)
            _, logits = get_index_with_sorting(arg_sort_index, sorted_logits, dim=-1)
            return qkv, logits

    
    def call(self, qk, v):
        batch_size, seq_length, dim = qk.shape

        #버켓사이즈로 나눠줌.. 그래서 버켓 갯수 대략 구함. 버켓 사이즈가 파라미터.
        bucket_nums = seq_length // self.bucket_size #이거 논문에 나온거임. bucket_size= seq/bucket_num
        chunknum= bucket_nums * self.num_hash
        r_size = bucket_nums//2
        qk = self.dropout(qk)
        #hash 만큼 해서 돌아.. LSH.. #batch , (hash*seq_len) 값은 bucket 값. 이걸로 sorting.
        # LSH 
        #맵핑 키 모양 잡자.
        r_shape = (batch_size,dim,self.num_hash,r_size)#공식임.
        #hash random key. broad cast는 expand랑 똑같음.
        # qk vector 태워서 보내려고 차원 맞추는거임.
        R = tf.broadcast_to(tf.random.normal(r_shape), (batch_size, dim, self.num_hash , r_size))
        
        #torch에도 있음. 태워서 보낸다.
        xR = tf.einsum('btf,bfhi->bhti', qk, R)

        #공식.
        xR = tf.concat([xR, -xR], axis=-1)
        # 해쉬 별로 안겹치게 하려고 offset 줌.
        # bucket수 128개
        hash_off = bucket_nums*tf.range(self.num_hash,dtype=tf.int64)
        #offset 더해주기
        hash_off = tf.reshape(hash_off, (1, -1, 1))

        
        # hash 여러개라서 안겹치게 함.. 나중에 sorting 함, + Multi-round
        buckets = tf.math.argmax(xR, axis=-1)
        buckets = tf.reshape(buckets + hash_off, (batch_size, -1,))

        # sorting 할거임. 위치정보 반영해서 정렬 seq_length * hashes 여기에 bucket 인덱스 들어갈거야.. + 원래 위치 인덱스랑.
        bucket_index = tf.expand_dims(tf.range(self.num_hash * seq_length), axis=0)#sorting 하고 되돌릴 인덱스 저장하기 위해서 인덱스 만듬. # unsqueeze대신 expand dim씀.
        #참고로 원래 버켓 값보다 많이 큼.
        scaled_bucket = seq_length * buckets + tf.cast((bucket_index % seq_length), tf.int64)#bucket값이 겹치지 않게 해야함. 그리고 bucket 값 다음으로 seq index 값도 반영해서 정렬해아하기 때문에 offset 이렇게 만듬
        
        #sorting 해주기.
        #sort key val은 기본적으로 sort와 그전 index를 가져다줌.
        sorted_scaled_bucket, arg_sort_index = get_index_with_sorting(scaled_bucket, bucket_index, dim=-1)#순서대로 버켓 넘 순으로 정렬된것과, argsort 인덱스. 처음이 원래 순서대로임.. 왜냐하면 bucket값이 0인 애들이 초반에 그 위치에 있기 때문.
        _, undo_sort_index = get_index_with_sorting(arg_sort_index, bucket_index, dim=-1) #기존 index를 반대로 넣어서 돌아가는 index를 만듬. undo sort를 곱하면 원래 순서로 돌아감.
        

        #stop graident = detach. #이런건 학습안함. 알고리즘이라.
        scaled_bucket,sorted_scaled_bucket= tf.stop_gradient(scaled_bucket), tf.stop_gradient(sorted_scaled_bucket)
        arg_sort_index,undo_sort_index= tf.stop_gradient(arg_sort_index), tf.stop_gradient(undo_sort_index)

        
        
        
        #우리는 bucket 순서대로. 정렬할건데 그래서 qk 도 그렇게 해야함.
        #그런데 아까 인덱스 바꾸는게 있었음. 이렇게하면 bucket 값 기준으로 qk값들이 정렬됨
        h_arg_sort_idx = (arg_sort_index % seq_length)#index 이긴 한대.. 한꺼번에 hash 다합쳐서 했기 때문에 나눠줌 
        
        # 원래 qk 버켓값으로 정렬한거 반영해서 바꿔줌.
        sorted_qk = reordering(qk, h_arg_sort_idx,chunknum,flag=True)
        sorted_v = reordering(v, h_arg_sort_idx,chunknum,flag=True)
        
        sorted_q_idx = sorted_kv_idx = tf.reshape(h_arg_sort_idx, (batch_size, chunknum, -1)) #인덱스
        
        #s_bucket 자체는 가중된 bucket값을 가짐. 그래서 다시 나눠줘야 원래 버켓값으로 감. 어자피 sort 했으니까 괜찮. 

        #논문에 나와잇는 부분.. key가 normalization 없는걸 방지하기 위해 nomalization 해줘야함. 
        sorted_q = sorted_qk  #TensorShape([2, 512, 128, 64])
        sorted_k = sorted_qk / tf.norm(sorted_qk,  ord=2, axis=-1, keepdims=True)

        # 논문에 나온대로 sort한 이후 k,v 에 대하여 자신과 전 chunk에 대해서 어텐션을 수행
        #q는 안해도 자기가 기준이니까.
        sorted_k = self.look_forward(sorted_k)#실제 값
        sorted_v = self.look_forward(sorted_v)
        sorted_kv_idx = self.look_forward(sorted_kv_idx)#이것은 인덱스.
        #sorted_kv_buckets = look_one_back(sorted_kv_buckets)#이것은 버켓값

        # 어텐션 식.
        attn_weight = tf.einsum('bhie,bhje->bhij', sorted_q, sorted_k) * (sorted_q.shape[-1] ** -0.5)
        
        #자신은 안넣는게 맞음 (예외처리 안함) 일단 appedix 공식대로 구현
        self_mask = sorted_q_idx[:, :, :, None] == sorted_kv_idx[:, :, None, :]
        attn_weight = tf.math.multiply(attn_weight, (1-tf.cast(self_mask, tf.float32))) + (tf.cast(self_mask, tf.float32)) * (- 1e5)
        del self_mask
    
        # Causal masking 논문에 언급..다음꺼 못보게 decoding 에서 쓰임.
        if self.causal:
            mask = sorted_q_idx[:, :, :, None] < sorted_kv_idx[:, :, None, :] # sorted_q_idx의 값은 인덱스임.. 원래 위치를 의미함. 따라서 이걸로 판단 쿼리 보다 크면 안하면 됨. mask 해준다.
            #마스크 하는 방법.
            attn_weight = tf.math.multiply(attn_weight, (1-tf.cast(mask, tf.float32))) + (tf.cast(mask, tf.float32))*(-1e9)#마스크는 매우 작게.       
            del mask


        # Softmax. 어텐션 공식대로 dk는 곱해졌음.
        attn_weight_logsumexp = tf.math.reduce_logsumexp(attn_weight, axis=-1, keepdims=True)
        attn_weight = tf.exp(attn_weight - attn_weight_logsumexp) 
        attn_weight = self.dropout(attn_weight)

        #sorted_v아까.. 이전 chunk까지해서 128임.
        sorted_qkv = tf.einsum('buij,buje->buie', attn_weight, sorted_v)#가중치값 v에 곱해줌. batch, chunk 수 , chunksize, dim 
        sorted_qkv = tf.reshape(sorted_qkv, (batch_size, -1, sorted_qkv.shape[-1]))#batch는 살리고, chunk 다 합침.= hash * seq_len
        
        
        #요것은 어텐션 합친 값임. 이게 왜필요하냐면 multiround hash 가중치 주려구..
        sorted_logits = tf.reshape(attn_weight_logsumexp, (batch_size, -1,))

        
        #순서 다시 바까주기. undosort.
        sorted_qkv, sorted_logits = tf.stop_gradient(sorted_qkv), tf.stop_gradient(sorted_logits)
        qkv, logits = self.undo_sorting(sorted_qkv, sorted_logits, arg_sort_index ,undo_sort_index)

        #요기가 multi-round hash임..
        #이제 해시로 쪼갬. 어텐션 총합별로 각 hash갑에 대한 가중치를 구함. ratio
        qkv = tf.reshape(qkv, (batch_size, self.num_hash, seq_length, qkv.shape[-1]))#b,nh,sl,dim
        logits = tf.reshape(logits, (batch_size, self.num_hash, seq_length, 1))# 이건 합이라서 dim 없음.
        ratio = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))#이건 그냥 소프트맥스 공식.
        
        
        #reduced sum. 줄 합치는거임. 그냥 torch.sum
        out = tf.reduce_sum(qkv * ratio, axis=1)
        return out, buckets

class MH_LSHAttention(tf.keras.Model):#multi-head + lsh_attention
    def __init__(self, emb, heads = 8, bucket_size = 64, num_hash = 8, causal = False, **kwargs):
        super(MH_LSHAttention, self).__init__()
    
        self.emb = emb
        self.headnum = heads
        self.attn = LSHAttention(bucket_size=bucket_size, num_hash = num_hash , causal=causal, **kwargs)
        self.lk = Dense(emb, use_bias = False)
        self.lv = Dense(emb, use_bias = False)
        self.lout = Dense(emb)
    
    def call(self, inputs):
        b, t, e, h = *inputs.shape, self.headnum#batch, seq_length, emb, head
        qk = self.lk(inputs)#linear 이건 그냥 self attention
        v = self.lv(inputs)

        def split_heads(v):
            return tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3])
        
        qk = tf.reshape(split_heads(qk), (b * h, t, -1)) # head 쪼개기. 512->8*64 #multihead attetntion... 
        v = tf.reshape(split_heads(v), (b * h, t, -1)) 


        chunked_list = list(map(lambda x: tf.split(x, self.headnum, axis=0), (qk,v)))# 멀티 어텐션 쪼개기.
        split_attn_outs = [self.attn(*chunk_input) for chunk_input in zip(*chunked_list)]#각각 어텐션 태우기
            

        attn_out = tf.concat([attn_out for (attn_out, _) in split_attn_outs], axis=0)#멀티어텐션 합치기
        out = tf.reshape(split_heads(attn_out), (b, t, e))#차원 바까주기.

        return self.lout(out)


class Chunk_feedforward_Normalization(Layer):
    def __init__(self, normalization, f, chunk_num):
        super(Chunk_feedforward_Normalization, self).__init__()
        
        self.norm = normalization()
        self.ffn = f
        self.chunk_num=chunk_num

    def call(self, inputs):
        outputs = self.norm(inputs)
        outputs = self.ffn(outputs)
        
        chunks = tf.split(outputs, self.chunk_num, axis= -2)
        return tf.concat([self.ffn(c) for c in chunks], axis = -2)
    


class Reformer(tf.keras.Model):
    def __init__(self, emb_size, depth, max_seq_len, heads , bucket_size , num_hash, ff_chunks , causal = False):
        super().__init__()
        self.emb_size = emb_size
        self.depth = depth
        
        #걍 feed forward
        ff_caller = lambda: Feed_Forward(emb_size)

        # lsh attention.. 핵심임.
        lsh_caller = lambda: MH_LSHAttention(emb_size, heads, bucket_size, num_hash, causal = causal)
    
        layers = []

        for _ in range(depth):
            f = lsh_caller()
            #chunk feed forward
            g = Chunk_feedforward_Normalization(LayerNormalization, ff_caller(),ff_chunks)
            #이것은 pytorch. reversible block code 참조.
            layers.append(Revnet(f, g))
        
        #요것도 참조.
        self.model_layers = Revnet_Layers(layers)

    def call(self, x):
        x = tf.concat([x, x], axis = -1)# x1, x2 하려고..
        x = self.model_layers(x)
        return tf.stack(tf.reduce_sum(tf.split(x, 2, axis=-1), axis=0))

class Reformer_Model(tf.keras.Model):
    def __init__(self, vocab_size, emb_size, depth, max_seq_len, heads , bucket_size , num_hash , ff_chunks ,  causal = False):
        super().__init__()
        self.token_emb = Embedding(vocab_size, emb_size)
        self.pos_emb = Embedding(max_seq_len, emb_size)
        self.reformer = Reformer(emb_size, depth, max_seq_len, heads = heads, bucket_size = bucket_size, num_hash = num_hash, ff_chunks = ff_chunks,  causal = causal)
        self.to_logits = Dense(vocab_size)

    def call(self, inputs):
        #axial positional embedding은 구현 못했습니다 ㅜㅜ
        #간이 임베딩 사용.
        inputs = self.token_emb(inputs) + self.pos_emb(tf.range(inputs.shape[1]))
        r_out = self.reformer(inputs) #batch, seq , dim
        output= self.to_logits(r_out)
        return output #batch, seq, vacab 
    