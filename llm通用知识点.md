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
5. 为什么LLM都是decoder结构？．https://spaces.ac.cn/archives/9529


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
   
