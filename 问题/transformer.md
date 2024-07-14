https://dongnian.icu/llm_interview_note/#/02.大语言模型架构/1.attention/1.attention
# 1. Attention
## Attention机制和传统的Seq2Seq模型有什么区别？
Attention机制和传统的Seq2Seq（Sequence-to-Sequence）模型在处理序列数据方面有显著的区别，主要体现在以下几个方面：

传统的Seq2Seq模型
结构
- 编码器-解码器架构：传统的Seq2Seq模型由一个编码器（Encoder）和一个解码器（Decoder）组成。编码器将输入序列编码为一个固定长度的上下文向量（Context Vector），然后解码器根据这个上下文向量生成输出序列。

工作原理
1. 编码阶段：编码器接受输入序列，并通过RNN（如LSTM或GRU）逐步处理每个输入元素，最后将整个序列压缩成一个**固定长度**的上下文向量。
2. 解码阶段：解码器根据这个固定长度的上下文向量，通过RNN逐步生成输出序列。

局限性
- 固定长度的上下文向量：由于所有信息都被压缩到一个固定长度的向量中，当输入序列非常长时，信息可能会丢失，导致模型性能下降。
- 长距离依赖问题：对于长序列，RNN在捕捉长距离依赖时可能表现不佳，因为信息在长序列中传递时可能会衰减或被遗忘。

Attention机制的Seq2Seq模型
结构
- 编码器-解码器架构+Attention机制：在传统的编码器-解码器架构中加入Attention机制。编码器仍然生成一系列隐藏状态向量（每个输入元素对应一个隐藏状态），而解码器在生成每个输出元素时，都会动态计算一个加权和，以选择性地关注输入序列中的不同部分。

工作原理
1. 编码阶段：编码器处理输入序列并生成一系列**隐藏状态向量**。
2. 解码阶段：每次生成输出元素时，**解码器会根据当前生成的状态和所有的隐藏状态向量计算注意力权重（Attention Weights）**。这些权重表示解码器应该关注输入序列中的哪些部分。然后，通过加权和计算出一个上下文向量，解码器使用这个上下文向量和之前生成的输出元素来生成下一个输出元素。

优点
1. 动态上下文向量：每个解码步骤都有一个动态计算的上下文向量，能够捕捉输入序列中的不同部分的信息。
2. 改进长距离依赖捕捉：Attention机制允许解码器在每一步都直接访问编码器的所有隐藏状态，从而更好地处理长距离依赖问题。
3. 提高模型性能：在翻译、文本生成等任务中，加入Attention机制的Seq2Seq模型通常表现更好，因为它能够更灵活地选择输入信息。

总结
传统Seq2Seq模型：依赖固定长度的上下文向量来捕捉输入序列的信息，可能会导致长序列的信息丢失和长距离依赖问题。
Attention机制的Seq2Seq模型：通过引入Attention机制，解码器在每个时间步可以动态关注输入序列的不同部分，显著提升了模型**处理长序列**和捕捉**长距离依赖**的能力，提高模型性能。

## 目前主流的attention方法有哪些？
- Scaled Dot-Product Attention: 这是Transformer模型中最常用的Attention机制，用于计算查询向量（Q）与键向量（K）之间的相似度得分，然后使用注意力权重对值向量（V）进行加权求和。
- Multi-Head Attention: 这是Transformer中的一个改进，通过同时使用多组独立的注意力头（多个QKV三元组），并在输出时将它们拼接在一起。这样的做法允许模型在不同的表示空间上学习不同类型的注意力模式。
- Relative Positional Encoding: 传统的Self-Attention机制在处理序列时并未直接考虑位置信息，而相对位置编码引入了位置信息，使得模型能够更好地处理序列中不同位置之间的关系。
- Transformer-XL: 一种改进的Transformer模型，通过使用循环机制来扩展Self-Attention的上下文窗口，从而处理更长的序列依赖性。


## Attention - token mixer, FFN - channel mixer？
[MLP-mixer这篇paper提出的抽象是最好的。类比到transformer，attention就是token-mixing，ffns就是channel-mixing](https://arxiv.org/pdf/2105.01601)

Attention主要起到的是token-mixer的作用，早期很多工作表明Attention换成MLP-mixer效果也还行，甚至MetaFormer说换成Pooling就可以（某些任务），
所以可以预估Attention本身带来的是弱非线性，因此就需要额外的FFN来补足它的非线性能力。

Token mixer和channel mixer是用于描述Transformer和类似架构中的不同操作的术语。它们反映了如何在不同维度上对输入数据进行处理和混合。让我们详细探讨一下这两个概念。

Token Mixer
定义
Token mixer（令牌混合器）是指在处理输入序列时，通过某种机制对序列中的各个元素（或“令牌”）进行混合或交互。其目的是捕捉序列中不同元素之间的关系和依赖性。

典型示例
自注意力机制（Self-Attention）：自注意力机制是Token mixer的经典示例。在Transformer中，自注意力机制允许每个令牌关注序列中所有其他令牌，从而捕捉全局依赖关系。
Attention

这里，查询（Query）、键（Key）和值（Value）向量分别从输入序列的各个token计算而来。
功能
捕捉长距离依赖：Token mixer能够有效地捕捉序列中远距离元素之间的依赖关系。
全局信息交互：通过混合序列中的所有令牌，实现全局信息的交互和融合。
Channel Mixer
定义
Channel mixer（通道混合器）是指在处理输入数据的不同通道（或特征维度）时，通过某种机制对各通道进行混合或交互。其目的是捕捉输入数据在特征维度上的关系和依赖性。

典型示例
前馈神经网络（Feed-Forward Network, FFN）：在Transformer的每一层中，前馈神经网络是一个典型的Channel mixer。FFN对每个令牌的特征向量进行独立处理，通过非线性变换混合不同的特征维度。
FFN
功能
特征变换和增强：Channel mixer通过非线性变换混合和重新组合特征，提高了模型对输入数据特征的表达能力。
增加非线性表示：通过激活函数（如ReLU），Channel mixer增加了模型的非线性表示能力。
实际应用
在实际的Transformer模型中，Token mixer和Channel mixer协同工作，以实现对输入数据的全面处理：

自注意力机制（Token mixer）：处理输入序列中的令牌，捕捉序列元素之间的关系。
前馈神经网络（Channel mixer）：处理输入数据的特征维度，对每个令牌的特征向量进行非线性变换和增强。
这种协同工作确保了模型能够有效地捕捉和表示输入数据在不同维度上的复杂依赖关系和特征。

总结
Token mixer：处理输入序列中的令牌，捕捉序列中不同元素之间的关系和依赖性。典型示例是自注意力机制。
Channel mixer：处理输入数据的特征维度，捕捉特征之间的关系和依赖性。典型示例是前馈神经网络（FFN）。
两者结合使得Transformer模型在处理复杂序列数据时具有强大的表达能力和灵活性。

## self-attention 和 target-attention的区别？
应用场景：
Self-Attention：用于序列的内部信息交互，例如Transformer模型中的所有层。
Target-Attention：用于解码器在生成输出时参考编码器的输出，通常在Seq2Seq模型中使用。

注意力的对象：
Self-Attention：每个元素对同一个序列中的所有其他元素进行注意力计算。
Target-Attention：解码器的每个生成步骤对编码器的输出进行注意力计算。

功能：
Self-Attention：捕捉序列内部的全局依赖关系。
Target-Attention：在生成输出时，选择性地关注输入序列的不同部分，帮助解码器生成准确的输出。
总结
Self-Attention：用于序列内部的全局信息交互和依赖关系捕捉，是Transformer模型的核心机制。
Target-Attention：transformer中encoder- decoder部分。用于解码器在生成输出时参考编码器的输出，是Seq2Seq模型中重要的机制。

# 2. Transformer
## attention与全连接层的区别何在？
**Attention和全连接最大的区别就是Query和Key，而这两者也恰好产生了Attention Score这个Attention中最核心的机制。**
而在Query和Key中，我认为Query又相对更重要，因为Query是一个锚点，Attention Score便是从过计算与这个锚点的距离算出来的。
任何Attention based algorithm里都会有Query这个概念，但全连接显然没有。
全链接层可没有什么Query和Key的概念，只有一个Value，也就是说给每个V加一个权重再加到一起（如果是Self Attention，加权这个过程都免了，因为V就直接是从raw input加权得到的。）

## [为什么transformer里面的自注意力总是被魔改，但里面的FFN魔改很少？](https://www.zhihu.com/question/646160485/answer/3470469940)
1. 自注意力机制的灵活性和重要性
信息交互与建模能力：
- 自注意力机制负责捕捉输入序列中各元素之间的相互关系，并将这些关系编码到输出表示中。这种全局信息交互的特性使得它在处理不同任务时具有高度的灵活性和可调整性。

研究热点和创新空间：
- **自注意力机制的设计对模型性能的影响非常大**，因此它成为许多研究者关注的热点。例如，提出不同的注意力计算方法（如相对位置编码、稀疏注意力等）可以显著改变模型的性能和效率。
- 由于注意力机制在捕捉**长距离依赖关系**方面的独特优势，许多研究集中在如何优化这一过程以提高模型的性能。

复杂度和效率优化：
- 传统的全连接自注意力**计算复杂度较高（O(n^2)）,是输入序列长度的平方，**尤其是在处理长序列时会导致计算资源的瓶颈。为了解决这个问题，研究者提出了许多优化方案，如Linformer、Performer等，它们通过稀疏化或近似的方法降低计算复杂度。

2. FFN的相对稳定性
- 结构简单和功能单一：
FFN的结构相对简单，通常包括两个线性变换和一个激活函数。**它的主要功能是对每个位置的特征进行独立的非线性变换，增强模型的表达能力，不涉及位置间的交互**。
由于其功能单一且明确，FFN在模型中的角色相对固定，不需要频繁调整。
- 性能影响较小：
虽然FFN也对模型性能有影响，但相比于自注意力机制，它的调整空间和对整体性能的影响相对较小。因此，研究者通常更关注对模型性能有更大提升潜力的自注意力机制。
成熟的设计和应用

FFN有两个常见变体，一个是LLAMA用的GLU，一个是稀疏化的MoE，目前三者都有一席之地。
**FFN不是不能改，而是FFN本身已经足够简单，而且足够有效，你改的话只能往复杂里改，还不一定有效**，何必呢。

相反，Attention部分虽然有很多魔改工作，但多数都是ChatGPT出来之前的结果，
大部分工作目前看来已经过时，在LLM的今天，主流的Attention形式依然是最早Transformer的scaled-dot product形式
（顶多加了个RoPE，换一下GQA、MQA），所以Attention才是几乎没变化的那个。

## [如何理解 Transformers 中 FFNs 的作用？](https://www.zhihu.com/question/622085869/answer/3518358912)，一般都会从非线性变换的角度来回答，attention 中也有 softmax，也是非线性，那 FFN 还是必须的么？

1. **FFN 设计的初衷，其实就是为模型引入非线性变换。**

虽然 Transformers 论文的名字叫《Attention is All your Need》，但是实际上， FFN and ResNet are also your need.
研究人员发现 FFN 和 ResNet 的 Skip Connection 无论去掉哪一个，模型都会变得不可用，具体可以看《One Wide Feedforward Is All You Need》 这篇论文。
所以说 Attention, FFN, ResNet 可以认为是 Transformers 架构的三驾马车，缺一不可 。

2. 
那接着问，attention 中也有 softmax，也是非线性，那 FFN 还是必须的么？ 

大多数人开始产生自我怀疑，开始从别的角度回答 FFN 的作用。就比如用Transformers 原始论文中的解释： FNN 可以看作用 1x1 的卷积核来进行特征的升维和降维。
其实这么追问是个陷阱，用来了解一下候选人对 Transformers 细节的把握情况。这个陷阱其实会引出另外一个问题：**attention 是线性运算的还是非线性运算的？**

全局来看，对于x来说是非线性运算。因为仔细看一下 Attention 的计算公式，其中确实有一个针对 q 和 k 的 softmax 的非线性运算。
但是对于 value 来说，并没有任何的非线性变换。**所以每一次 Attention 的计算相当于是对 value 代表的向量进行了加权平均，虽然权重是非线性的权重。**
这就是 FFN 必须要存在的原因，或者说更本质的原因是因为 FFN 提供了最简单的非线性变换。
线性变换无法处理一些非线性的特征，恰如当年马文明斯基给神经网络判的死刑，只需要加个非线性变换的激活函数就能起死回生。

## transformer中multi-head attention中每个head为什么要进行降维？

在Transformer的Multi-Head Attention中，对每个head进行降维是为了增加模型的表达能力和效率。
每个head是独立的注意力机制，它们可以学习不同类型的特征和关系。通过使用多个注意力头，Transformer可以并行地学习多种不同的特征表示，从而增强了模型的表示能力。

然而，在使用多个注意力头的同时，注意力机制的计算复杂度也会增加。原始的Scaled Dot-Product Attention的计算复杂度为
$O(d^2)$，其中d是输入向量的维度。如果使用h个注意力头，计算复杂度将增加到$O(hd^2)$。这可能会导致Transformer在处理大规模输入时变得非常耗时。

为了缓解计算复杂度的问题，Transformer中在每个head上进行降维。在每个注意力头中，输入向量通过线性变换被映射到一个较低维度的空间。这个降维过程使用两个矩阵：一个是查询（Q）和键（K）的降维矩阵
$W_q$和$W_k$，另一个是值（V）的降维矩阵$W_v$.
​通过降低每个head的维度，Transformer可以在保持较高的表达能力的同时，大大减少计算复杂度。降维后的计算复杂度为
$(hd_1^2)$，其中$d_1$是降维后的维度。通常情况下，$d_1$会远小于原始维度d，这样就可以显著提高模型的计算效率。

## [transformer的点积模型做缩放的原因是什么？为什么scaled是维度的根号，不是其他的数？](https://www.zhihu.com/question/339723385)
使用缩放的原因是为了控制注意力权重的尺度，以避免在计算过程中出现梯度爆炸的问题。

解决方法就有两个:
- 像NTK参数化那样，在内积之后除以$\sqrt{d_K}$，使q⋅k的方差变为1，对应这样softmax之后也不至于变成one hot而梯度消失了，这也是常规的Transformer如BERT里边的Self Attention的做法
- 另外就是不除以​	$\sqrt{d_K}$，但是初始化q,k的全连接层的时候，其初始化方差要多除以一个d，这同样能使得使q⋅k的初始方差变为1，T5采用了这样的做法。