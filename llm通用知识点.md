# 学习资料
https://datawhalechina.github.io/llm-cookbook/#/

# 名词解释
zero shot prompting
```

```

few shot prompting

In Context Learning

Instruct

linguistic和semantic有什么不一样？
```
Linguistic（语言学）：
Linguistic关注语言的结构、规则和形式，以及语言如何被使用和理解。
它涉及到语音、语法、语义、语用等语言层面的特征和规律。
Linguistic分析通常包括词法分析、句法分析、语义分析等，用于理解和描述语言的结构和特征。
Semantic（语义学）：
Semantic关注语言中词语、短语和句子的含义和解释。
它研究词语和句子的意义，以及它们在不同上下文中的解释和使用。
Semantic分析涉及词义的表示、句子的语义解释、逻辑推理等，用于理解语言的含义和语义关系。
```

# 问题：
## 1.对LLM的通用理解
[参考博客文章，LLM综述](https://zhuanlan.zhihu.com/p/597586623 "悬停显示")
1. [LLM学习到了什么知识？](#llm学习到了什么知识)
2. [LLM如何存取知识？](#llm如何存取知识)
```
上面2个问题可以阅读论文[BERTnesia: Investigating the capture and forgetting of knowledge in BERT]
https://arxiv.org/pdf/2106.02902
```
3. [如何修改LLM存储的知识？](#如何修改llm存储的知识)
4. [当LLM越来越大时会发生什么？](#当llm越来越大时会发生什么)
5. [FLOPs是什么](#FLOPs是什么)


## 1.对LLM的通用理解
### llm学习到了什么知识
* **语言类知识：**
  包括语法、词法、词性、语义等信息。<mark>浅层语言知识比如词法、词性、句法等，存储在Transformer的底层和中层；抽象的语言知识比如语义，存储在Transformer的中层和高层结构中</mark>。
* **事实性知识（factual knowledge）**
  包括世界上真实发生的事件，如俄乌战争等。这些知识分布在Transformer的中高层，尤其聚集在中层。随着Transformer的层数增加，能够学习到的事实知识成<mark>指数级增加</mark>。
  ```
  这里的底层、中层、高层是什么还没有弄清楚
  ```
### LLM如何存取知识？
可以参考论文[BERTnesia: Investigating the capture and forgetting of knowledge in BERT](#https://arxiv.org/pdf/2106.02902)
 1. 知识存在哪里？
  位置：语言类知识存在低、中层；事实性知识存在中、高层
模型参数：
1. 知识用什么形式存储？
### 如何修改LLM存储的知识？
知识编辑通常包含三个基本的设定：知识新增、知识修改和知识删除。假设某条世界知识存在某些FFN节点的参数里。
1. <mark>从训练数据的源头修正知识</mark>
  方式：
  优点：
  缺点：
1. <mark>对LLM进行一次fine-turning修正知识</mark>
  方式：
  优点：
  缺点：
1. <mark>直接修改LLM里某些知识对应的模型参数修正知识</mark>
  方式：
  优点：
  缺点：
* [LLM知识编辑可以参考比赛](https://tianchi.aliyun.com/competition/entrance/532182/information)
* [LLM知识编辑相关的论文可以参考](https://github.com/zjunlp/KnowledgeEditingPapers?spm=a2c22.12281978.0.0.1d05648eOJdWNN)

### 当llm越来越大时会发生什么
### FLOPs是什么
FLOPs 是指在执行某项任务，如模型推理时，所进行的浮点运算次数（Floating Point Operations）。在深度学习领域，特别是在模型性能评估时，FLOPs 是一个重要的指标，用来衡量模型在单次前向传递中需要多少计算资源。FLOPs 通常用于评估模型的复杂性和效率，尤其是在资源受限的环境中。
#### 如何计算 FLOPs？
计算 FLOPs 需要详细分析模型的每一层所执行的运算。以下是一些主要步骤和组件的通常计算方法：
- 线性层（全连接层）：
FLOPs 通常是输入特征数乘以权重矩阵的大小（即输入特征数乘以输出特征数）。如果有偏置，则额外加上输出特征数。
计算公式：
FLOPs=(input_features × output_features) + output_features（如果有偏置）
- 卷积层：对于卷积层，FLOPs 取决于卷积核的大小、输入通道数、输出通道数、输出特征图的维度。
计算公式：
FLOPs=kernel_height × kernel_width × input_channels × output_height × output_width × output_channels
- 激活函数：
激活函数如 ReLU 或 Sigmoid 的 FLOPs 通常等于处理的元素数量（即神经元数量）。
- 归一化层（如 Batch Normalization）：
归一化层的 FLOPs 通常也等于处理的元素数量，因为每个元素需要执行一定数量的乘法和加法操作。
残差连接和其他元素操作：
这些操作的 FLOPs 通常基于元素级的加法或乘法。
#### 综合计算模型的 FLOPs
要计算整个模型的 FLOPs，你需要将所有层的 FLOPs 相加。对于复杂的模型，这通常需要通过编程自动化完成，尤其是对于有大量层的深度网络。
