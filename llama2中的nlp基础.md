# 总览

## 参数量计算
[以13B为例](https://medium.com/@saratbhargava/mastering-llama-math-part-1-a-step-by-step-guide-to-counting-parameters-in-llama-2-b3d73bc3ae31)
![Alt text](image.png)

    args = ModelArgs(
        dim=5120,
        n_layers=32,
        n_heads=40,
        n_kv_heads=None,
        vocab_size=32000,   #64793  #50257
        multiple_of=256,
        norm_eps=1e-5,
        max_seq_len=1024,
        dropout=0.0
    )

### Embedding Block
   
    #词汇表嵌入的参数量
    args.vocab_size * args.dim= 32,000 x 5,120 = 163,840,000

### Attention block
13B用的 Multi-head attention (MHA)，70B版本用的是 Grouped-query attention (GQA)，attention参数以MHA为例。13B有40个attention heads，每个头有128维。

    #W_Q的参数量=5120 x (128 x 40)=26,214,400
    # W_O, W_K, 和 W_V参数量一样，加上最后有一个线性层W_O 


## RoPE
[Transformer升级之路：2、博采众长的旋转式位置编码 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/8265)

## Attention
### k-v cache
  - [大模型推理性能优化之KV Cache解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/630832593)
### 分组查询注意力（Grouped-Query Attention）
[Grouped-Query Attention，GQA](https://arxiv.org/pdf/2305.13245)，7b和13b模型并没有增加GQA，Llama2新加入。GQA共享key和value对，在推理的时候可以减少kv cache处理的sequence中token
  ![Alt text](image-1.png)
### rotary embeddings
  - [复数参考资料：复数基础与二维空间旋转 - 何雨龙 - 博客园 (cnblogs.com)/让研究人员绞尽脑汁的Transformer位置编码 - 科学空间|Scientific Spaces (kexue.fm)](https://www.cnblogs.com/noluye/p/11964513.html)
## FFN
- SwiGLU激活函数：在前馈神经网络（FFN）使用SwiGLU 激活函数替换了Transformer中的 ReLU 激活函数来提升性能
- RMSNorm


参考资料
- https://www.mlpod.com/494.html
- [【llm大语言模型】一文看懂llama2(原理,模型,训练) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/651248009)
- 解析论文：https://zhuanlan.zhihu.com/p/644671690

