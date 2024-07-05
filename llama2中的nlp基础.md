# 总览

## 参数量计算
[以13B为例](https://medium.com/@saratbhargava/mastering-llama-math-part-1-a-step-by-step-guide-to-counting-parameters-in-llama-2-b3d73bc3ae31)

    LlamaForCausalLM(
    (model): LlamaModel(
        (embed_tokens): Embedding(32000, 5120)
        (layers): ModuleList(
        (0-39): 40 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
            (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
        )
        )
        (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
    )

模型参数：

    args = ModelArgs(
        dim=5120,
        n_layers=40,
        n_heads=40,
        n_kv_heads=None,
        vocab_size=32000,   #64793  #50257
        multiple_of=256,
        norm_eps=1e-5,
        max_seq_len=1024,
        dropout=0.0
    )

### Embedding Block
   
    (embed_tokens): Embedding(32000, 5120)
    #词汇表嵌入的参数量
    args.vocab_size * args.dim= 32,000 x 5,120 = 163,840,000

### Attention block
13B用的 Multi-head attention (MHA)，70B版本用的是 Grouped-query attention (GQA)，attention参数以MHA为例。13B有40个attention heads，每个头有128维。

    (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (k_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (v_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )

    #W_Q的参数量=5120 x (128 x 40)=26,214,400
    # W_O, W_K, 和 W_V参数量一样，加上最后有一个线性层W_O 
    #所以参数量是3x5120 x (128 x 40) + 5120 x 5120 = 104,857,600  1亿
### MLP Block
有三个线性层，gate_proj、up_proj 和 down_proj 是三个线性变换层，用于将输入张量 x 映射到不同的表示空间。它们分别用于产生门控信号、升维和降维操作。
1. gate_proj
作用：gate_proj（门控投影）通常用于在 MLP 中引入非线性和控制信息流。在某些模型设计中，如 Gated Linear Networks 或其他使用门控机制的架构，它可能会起到控制下一层激活的重要作用。它的输出可能用作后续计算的门控信号。
2. up_proj
作用：up_proj（上升投影）层通常用于扩展输入特征的维度。在 Transformer 的 MLP 中，这个阶段称为“展开”阶段，通过增加维度，可以增加模型的表示能力和处理更复杂模式的能力。
3. down_proj
作用：down_proj（下降投影）层作用是将之前扩展的特征维度“缩减”回较低的维度，通常与输入维度相同，这样可以将处理过的信息重新整合，并继续传递给网络的下一部分。

    (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
            (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
            (act_fn): SiLUActivation()
            )
    out = down_proj( act_fn(gate_proj(input)) x up_proj(input) ).
    # mlp参数量= 3 x 5120 x 13,824 = 212,336,640    2亿

### [RMS Norm layers](https://arxiv.org/abs/1910.07467)

    (input_layernorm): LlamaRMSNorm()
    (post_attention_layernorm): LlamaRMSNorm()
    per_layer_rms_norm_ parameters = 2 x 5120 and pre_lm_head_rms_norm_parameters = 5120.

### LM Head

     (lm_head): Linear(in_features=5120, out_features=32000, bias=False)
      lm_head_parameters = 5,120 x 32,000 = 163,840,000 1亿

## 加和

    otal parameters = embed_parameters + num_layers x (attn_module_parameters + mlp_block_parameters + per_layer_rms_norm_ parameters) + pre_lm_head_rms_norm_parameters + lm_head_parameters

    Substituting the respective values:

    Total parameters = 163,840,000 + 40 x ( 104,857,600 + 212,336,640 + 5,120 x 2) + 5, 120 + 163,840,000 = 13,015,864,320

## [训练过程中的显存](https://zhuanlan.zhihu.com/p/624740065)


## 位置编码，RoPE（旋转位置编码）
[Transformer升级之路：2、博采众长的旋转式位置编码 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/8265)
不同于RNN、CNN等模型，对于Transformer模型来说，位置编码的加入是必不可少的，因为纯粹的Attention模块是无法捕捉输入顺序的，即无法区分不同位置的Token。为此我们大体有两个选择：
- 想办法将位置信息融入到输入中，这构成了绝对位置编码的一般做法；
- 想办法微调一下Attention结构，使得它有能力分辨不同位置的Token，这构成了相对位置编码的一般做法。
### 绝对位置编码
[一般来说，绝对位置编码会加到输入中：在输入的第k个向量xk中加入位置向量pk变为xk+pk，其中pk只依赖于位置编号k](https://spaces.ac.cn/archives/8130)
#### 训练式
训练式是**将位置编码当作可训练参数**，比如最大长度为512，编码维度为768，那么就初始化一个512×768的矩阵作为位置向量，让它随着训练过程更新。现在的BERT、GPT等模型所用的就是这种位置编码。事实上它还可以追溯得更早，比如2017年Facebook的[《Convolutional Sequence to Sequence Learning》](https://papers.cool/arxiv/1705.03122)就已经用到了它。
对于这种训练式的绝对位置编码，一般的认为它的**缺点是没有外推性**，即如果预训练最大长度为512的话，那么最多就只能处理长度为512的句子，再长就处理不了了。当然，也可以将超过512的位置向量随机初始化，然后继续微调。
#### 三角式
三角函数式位置编码，一般也称为Sinusoidal位置编码，是Google的论文[《Attention is All You Need》](https://papers.cool/arxiv/1706.03762)所提出来的一个显式解：
很明显，三角函数式位置编码的**特点是有显式的生成规律**，因此可以期望于它有一定的外推性。另外一个使用它的理由是：由于sin(α+β)=sinαcosβ+cosαsinβ
以及cos(α+β)=cosαcosβ−sinαsinβ，这表明位置α+β的向量可以表示成位置α和位置β的向量组合，这提供了表达相对位置信息的可能性。但很奇怪的是，现在我们很少能看到直接使用这种形式的绝对位置编码的工作，原因不详。
#### 递归式 
基于递归模型的位置编码也具有**比较好的外推性**，同时它也比三角函数式的位置编码有更好的灵活性（比如容易证明三角函数式的位置编码就是FLOATER的某个特解）。但是很明显，递归形式的位置编码牺牲了一定的并行性，可能会**速度瓶颈**。

### 相对位置编码
#### 经典式
相对位置编码起源于Google的论文[《Self-Attention with Relative Position Representations》](https://papers.cool/arxiv/1803.02155)，华为开源的NEZHA模型也用到了这种位置编码，后面各种相对位置编码变体基本也是依葫芦画瓢的简单修改。
#### XLNET式
XLNET式位置编码其实源自Transformer-XL的论文[《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》](https://papers.cool/arxiv/1901.02860)，只不过因为使用了Transformer-XL架构的[XLNET](https://papers.cool/arxiv/1906.08237)模型并在一定程度上超过了BERT后，Transformer-XL才算广为人知，因此这种位置编码通常也被冠以XLNET之名。
#### T5式
T5模型出自文章[《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://papers.cool/arxiv/1910.10683)，里边用到了一种更简单的相对位置编码。思路依然源自展开式(7)
，如果非要分析每一项的含义，那么可以分别理解为“输入-输入”、“输入-位置”、“位置-输入”、“位置-位置”四项注意力的组合。如果我们认为输入信息与位置信息应该是独立（解耦）的，那么它们就不应该有过多的交互，所以“输入-位置”、“位置-输入”两项Attention可以删掉，而piWQW⊤Kp⊤j
实际上只是一个只依赖于(i,j)的标量，我们可以直接将它作为参数训练出来。
这个设计的思路其实也很直观，就是比较邻近的位置（0～7），我们需要比较得精细一些，所以给它们都分配一个独立的位置编码，至于稍远的位置（比如8～11），我们不用区分得太清楚，所以它们可以共用一个位置编码，距离越远，共用的范围就可以越大，直到达到指定范围再clip。
### [RoPE](https://spaces.ac.cn/archives/8265)
在RoPE中，我们的出发点就是“通过绝对位置编码的方式实现相对位置编码”，这样做既有理论上的优雅之处，也有实践上的实用之处，比如它可以拓展到线性Attention中就是主要因为这一点。
#### 复数的旋转特性
考虑复平面这个几何情形，乘以一个复数，可以同时带来两种变换的效果。
1. 长度的缩放（通过改变模长）。
2. 旋转（通过改变幅角）。
乘以一个模为 1 的复数时，不会导致缩放，只会产生旋转。这样的复数就称为旋转子（rotor），旋转子提供了“纯”旋转动作的数学表示，它可以将复数旋转任意角度。一般而言，将复数旋转角度 θ的旋转子定义为：

## Attention
### k-v cache
[大模型推理性能优化之KV Cache解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/630832593)
[key和value用来计算点积attention。](https://medium.com/@joaolages/kv-caching-explained-276520203249)k-v cache 用于在token生成阶段且仅用在decoder中，比如decoder模型GPT或encoder-decoder模型中的decoder部分如T5，像bert这种encoder模型没有KV cache。GPT类模型一次推理只输出一个token，输出token会与输入tokens 拼接在一起，然后作为下一次推理的输入，这样不断反复直到遇到终止符。即在推理过程中，每 step 内，输入一个 token序列，经过Embedding层将输入token序列变为一个三维张量[b, s, h]，经过一通计算，最后经logits层将计算结果映射至词表空间，输出张量维度为[b, s, vocab_size]。当前轮输出token与输入tokens拼接，并作为下一轮的输入tokens，反复多次。可以看出第
i+1轮输入数据只比第i轮输入数据新增了一个token，其他全部相同！因此第i+1轮推理时必然包含了第i轮的部分计算。KV Cache的出发点就在这里，缓存当前轮可重复利用的计算结果，下一轮计算时直接读取缓存结果，就是这么简单，不存在什么Cache miss问题。
但是，kv cache可能会占用很大一部分内存，成为长上下文生成的瓶颈，尤其是对于大型语言模型。

[llama2源码中，generate过程中每次新的查询 (Q)是新token](https://github.com/meta-llama/llama/issues/151)。为什么新的查询 (Q)是新token，而不是新token加上之前生成的所有token？因为每次生成过程中只需要为新加入的 token 计算查询（Q），而不需要为整个序列重新计算查询。这是因为序列中之前的部分已经在之前的生成步骤中被处理过，并且这些部分没有发生变化。假设我们使用一个 Transformer 模型来生成文本，起始输入为 "The quick brown fox"，我们希望继续生成文本。初始生成步骤:输入序列是 "The quick brown fox"。对于这个序列，模型计算每个单词的 Q, K, V。基于这些计算，模型生成一个新的词，比如 "jumps"。下一个生成步骤:现在序列变成了 "The quick brown fox jumps"。在这个步骤中，对于 "The quick brown fox" 部分的 Q, K, V 已经在前一步计算过了，因此不需要重新计算。模型只需要为新加入的词 "jumps" 计算 Q, K, V。这些新计算的值将用来与之前缓存的 K, V 值一起，计算新的输出词的概率。

1. [KV Cache节省了Self-Attention层中哪部分的计算？](https://zhuanlan.zhihu.com/p/630832593)
    KV Cache 主要节省了对已经计算过的键（Key）和值（Value）的重复计算。键（Keys）和值（Values）的重复计算：在没有使用 KV Cache 的情况下，每次输入序列更新时（例如，在文本生成或语音识别的递增模式中），模型需要为整个输入序列重新计算 Keys 和 Values，即使大部分序列元素已经在之前的步骤中处理过。这种重复计算尤其在长序列处理中非常低效。
    通过使用 KV Cache：只需为新添加到序列中的元素（通常是一个或几个 token）计算新的 Keys 和 Values。之前计算的 Keys 和 Values 被存储（缓存）起来，并可以直接在后续的注意力计算中重用。这意味着对于大部分序列，模型不需要重新计算 Keys 和 Values，只需将新计算的部分添加到现有的缓存中。
2. KV Cache对MLP层的计算量有影响吗？
    KV Cache 主要设计用于优化 Transformer 模型中的自注意力（Self-Attention）层的计算，而不直接影响多层感知器（MLP）层的计算量。MLP层主要进行基于前一层输出的密集矩阵运算，并不涉及对序列中不同时间步长的数据进行长期缓存。因此，KV Cache 的使用与MLP层的运算量没有直接关系。
3. KV Cache对block间的数据传输量有影响吗？
   在不使用KV Cache的情况下，每个新的生成步骤都可能需要重新从头开始计算整个序列的键（Key）和值（Value），这涉及到大量的数据再次通过模型的多个层传输。使用KV Cache后，仅需要为新增加的序列部分计算键和值，并将其与已缓存的数据合并。这减少了因重复计算而需要在模型层之间传输的数据量。
4. 为什么降低KV Cache的大小如此重要？
   众所周知，一般情况下LLM的推理都是在GPU上进行，单张GPU的显存是有限的，一部分我们要用来存放模型的参数和前向计算的激活值，这部分依赖于模型的体量，选定模型后它就是个常数；另外一部分我们要用来存放模型的KV Cache，这部分不仅依赖于模型的体量，还依赖于模型的输入长度，也就是在推理过程中是动态增长的，当Context长度足够长时，它的大小就会占主导地位，可能超出一张卡甚至一台机（8张卡）的总显存量。

    在GPU上部署模型的原则是：能一张卡部署的，就不要跨多张卡；能一台机部署的，就不要跨多台机。这是因为“卡内通信带宽 > 卡间通信带宽 > 机间通信带宽”，由于“木桶效应”，模型部署时跨的设备越多，受设备间通信带宽的的“拖累”就越大，事实上即便是单卡H100内SRAM与HBM的带宽已经达到了3TB/s，但对于Short Context来说这个速度依然还是推理的瓶颈，更不用说更慢的卡间、机间通信了。

    所以，减少KV Cache的目的就是要实现在更少的设备上推理更长的Context，或者在相同的Context长度下让推理的batch size更大，从而实现更快的推理速度或者更大的吞吐总量。当然，最终目的都是为了实现更低的推理成本。

    要想更详细地了解这个问题，读者可以进一步阅读《FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness》、《A guide to LLM inference and performance》、《LLM inference speed of light》等文章，
### Attention
![Alt text](images/image-2.png)
#### MHA（Multi-head Attention）
标准的多头注意力机制，包含h个Query、Key 和 Value 矩阵。所有注意力头的 Key 和 Value 矩阵权重不共享。后面的MQA、GQA、MLA，都是围绕“如何减少KV Cache同时尽可能地保证效果”这个主题发展而来的产物。
#### MQA（Multi-Query Attention，Fast Transformer Decoding: One Write-Head is All You Need）
多查询注意力的一种变体，也是用于自回归解码的一种注意力机制。与MHA不同的，MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。
#### GQA（Grouped-Query Attention，Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints）
分组查询注意力，GQA将查询头分成G组，每个组共享一个Key 和 Value 矩阵。GQA-G是指具有G组的grouped-query attention。GQA-1具有单个组，因此具有单个Key 和 Value，等效于MQA。若GQA-H具有与头数相等的组，则其等效于MHA。
[Grouped-Query Attention，GQA](https://arxiv.org/pdf/2305.13245)，7b和13b模型并没有增加GQA，Llama2新加入。GQA共享key和value对，在推理的时候可以减少kv cache处理的sequence中token


## FFN
在 Transformer 模型中，MLP层通常位于自注意力层之后，用于对从注意力层传入的每个位置的数据进行进一步的非线性变换。这一层通常包括以下几个步骤：
线性变换：首先对数据应用一个线性变换（通常是一个全连接层）。
激活函数：应用一个非线性激活函数，如 ReLU 或 GELU。
第二个线性变换：进行另一个线性变换。
可能的正则化：如 Dropout。


- SwiGLU激活函数：在前馈神经网络（FFN）使用SwiGLU 激活函数替换了Transformer中的 ReLU 激活函数来提升性能
- RMSNorm



## SFT指令微调

## 并行训练

参考资料
- [【llm大语言模型】一文看懂llama2(原理,模型,训练) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/651248009)
- 解析论文：https://zhuanlan.zhihu.com/p/644671690

