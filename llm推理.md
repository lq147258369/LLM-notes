

# 部署和推理

- [大模型推理一般是自回归型的任务，往往是显存密集型的任务；](https://zhuanlan.zhihu.com/p/655557420)
- 大模型推理主要是考虑延迟和吞吐量；
- 大模型推理非常的耗费显存，除了模型占用显存外，kv cache本身也会占用大量的显存；
- 大模型太大的时候，可能会遇到单机无法存下，这时候需要分布式；

FLOPS

[注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。](https://zhuanlan.zhihu.com/p/376925457)

计算公式：
对卷积层：(K_h * K_w * C_in * C_out) * (H_out * W_out)
对全连接层：C_in * C_out
FLOPs

注意s小写，是floating point operations的缩写（s表示复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度

ps:FLOPs 是模型推理时间的一个参考量，但并不能百分百表示该模型推理时间的长短，
因为乘法和加法计算不一样，乘法的时间一般是加法时间的四倍，但现在有很多优化卷积
层的计算算法，可能把乘法计算时间缩为加法的两倍不等，所以FLOPs只是个估量的指标，
不是决定推理时间长短的指标。即FLOPs越小并不代表着模型推理时间越短
## 性能指标计算
[几个推理时需要关注的几个关键的指标，即延迟(latency)、吞吐量(throughput)和推理成本(tokens/$)。](https://www.birentech.com/Research_nstitute_details/22.html)延迟是指前后连续产生关联token之间的时间，吞吐量是每秒能够产生的新token数，推理成本则是考虑设备成本与运营成本之后使用单位成本(1美元)可以产生的token数。

https://arthurchiao.art/blog/llm-inference-speed-zh/
https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f

## [投机采样（Speculative Decoding）](https://www.zhihu.com/question/588122011)

## [多头美杜莎机制](https://www.birentech.com/Research_nstitute_details/24.html)
