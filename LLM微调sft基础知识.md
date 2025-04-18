# 微调
有监督微调
微调的核心目标在于实现知识的精细化灌输与指令系统的精确匹配，所以SFT的重点是学习样式和指令，而非知识注入。当前实践中，微调主要分为全参数微调和部分参数微调。


## [微调基础知识](https://zhuanlan.zhihu.com/p/682604566)

微调是指在已经预训练好的大型语言模型基础上，使用特定的数据集进行进一步的训练，使模型适应特定任务或领域。微调主要目的是，完成知识注入、指令对齐。目前微调的方式一般分为以下方式

### 指令微调  （instruction tuning）
在进行指令微调的时候，会将Instruction（指令） 以及对应的answer拼接成文本（在拼接过程中一般会加入【USER】、【BOT】等角色，同时会加入开始、结束的special token，这样可以转换成一个chat式任务）。如一个翻译任务

instruction：
> 【USER】：将下列内容翻译成英语：｛待翻译文本｝

answer:
>【BOT】：{翻译结果}

拼接后的文本：
> <bos_token>【USER】：将下列内容翻译成英语：｛待翻译文本｝<special token>【BOT】：{翻译结果} <eos_token>

将拼接文本采用预训练任务的方式进行自回归预测，**和预训练的区别在于loss的计算**，同样使用Cross-Entropy作为loss，在指令微调的时候只会计算answer部分，Instruction部分通过设置ignore_index隐掉。在上面的案例中，我们只会计算 “【BOT】：” 之后的loss。


## 微调样本
[大模型微调如何组织训练样本？](https://zhuanlan.zhihu.com/p/641562439)
SFT的重点是学习样式，而非知识注入，所以SFT的样本在于其质量而非数量，少量但精良的样本往往胜过大批中低品质的样本，实现同样甚至更优的微调效果。高质量的样本远比大量中低质量的样本效果好，一般1万左右的样本数量就有较好的效果。 Meta 在《LIMA: Less Is More for Alignmen》  这篇论文论述了高质量微调数据的重要性，因此我们应该花更多的时间去提升样本质量，而不是追求样本的数量。那这样就会带来一个新的问题，如何评估微调样本的质量，这个问题可以作为一个单独的话题去讨论。对于微调样本质量评估，一般需要评估样本多样性、answer质量。
### 微调数据制作

### 评估微调样本质量
1. 样本多样性（Sample Diversity）：
指令多样性：考察样本中指令的覆盖范围是否广泛，是否包含了各类任务类型、不同难度级别以及多样化的指令结构和表达方式，确保模型在微调后能应对多种复杂情境。
内容多样性：检查样本中提供的文本内容是否涵盖了不同主题、文体、长度以及语境，以避免模型在特定领域或文本类型上过拟合，确保其具备良好的泛化能力。
2. 答案质量（Answer Quality）：
准确性（Accuracy）：评估答案是否准确无误地响应了给定指令和内容，是否忠实反映了任务要求，且不包含事实性错误、逻辑矛盾或语义模糊。
完备性（Completeness）：考察答案是否全面覆盖了指令所要求的所有任务点，尤其对于多步骤或复合任务，答案应完整体现所有必要的操作结果。
简洁性与清晰度（Conciseness & Clarity）：衡量答案是否言简意赅、表达清晰，避免冗余信息或含糊表述，确保模型在微调后生成的输出易于理解和使用。
3. 一致性（Consistency）：
内部一致性：检查同一指令对不同内容的处理结果是否保持一致，即模型在相似情境下应给出相似的答案。
外部一致性：对比样本答案与已知的知识库、专家判断或公认的基准结果，确保答案符合领域共识和常识。
4. 难度适配（Difficulty Calibration）：
难易程度分布：分析样本集中简单、中等、复杂任务的比例，确保微调数据集包含不同难度级别的样本，有助于模型逐步提升处理复杂指令的能力。
5. 噪声控制（Noise Reduction）：
标签错误检查：识别并剔除标注错误或不一致的样本，确保答案与指令、内容间的映射关系正确无误。
数据清洗：去除重复样本、无关内容或低质量文本，提升数据集的整体纯净度。



## 微调方法
微调方法分为全参数微调（Full Fine-tuning）、部分参数微调（Repurposing）典型全微调方法的如：SFT，部分微调的方法包括：LoRA、Adapter、Prefix-tuning、P-tuning、Prompt-tuning 、Freeze-tuning等。如果在资源充足的情况下，建议使用SFT进行全量微调。部分参数微调的方法不稳定，在有的场景下效果不理想。
### 添加特定的任务层

针对不同任务，添加特定任务层，如分类任务，在模型最后一层添加softmax层。典型的案例如reward模型的实现。    

目前大模型具备超强生成能力，我们可以通过生成式的方式去解决判别式任务，如多目标的文本分类问题，我们可以采用指令微调方式去解决，效果非常好。甚至我们在7B、3B的base模型上，去生成一个复杂json结构（包含多层结构的标签）依然work。
### SFT训练-Trick
受GPT论文的影响，目前大模型通用训练模式是三阶段训练模式，第一阶段pre-train，第二阶段是SFT，第三阶段是RLHF。通过三阶段训练分别得到base模型以及chat模型，chat模型是在base模型基础进行通用任务的SFT以及RLHF，使模型具备了对话能力、推理能力、用户偏好对齐、以及其他的NLU的能力。如果我们想在实际业务场景中使用大模型，一般还需要进行领域数据的微调。下面分享在领域数据SFT一些Trick。

### 训练模式选择
在进行领域任务的SFT的时候我们通常会有以下训练模式进行选择，根据领域任务、领域样本情况、业务的需求我们可以选择合适的训练模式。
模式一：基于base模型+领域任务的SFT；
模式二：基于base模型+领域数据 continue pre-train +领域任务SFT；
模式三：基于base模型+领域数据 continue pre-train +通用任务SFT+领域任务SFT；
模式四：基于base模型+领域数据 continue pre-train +通用任务与领域任务混合SFT；
模式五：基于base模型+领域数据 continue pre-train（混入SFT数据） +通用任务与领域任务混合SFT；
模式六：基于chat模型+领域任务SFT；模式六：基于chat模型+领域数据 continue pre-train +领域任务SFT

- 在资源充足的情况下，如只考虑领域任务效果，建议选择模式二；
- 在资源充足的情况下，如考虑模型综合能力，建议选择模式五；
- 在资源不允许的情况下，我会考虑模式六；

a.是否需要continue pre-train
大模型的知识来自于pre-train阶段，如果你的领域任务数据集与pre-train的数据集差异较大，比如你的领域任务数据来自公司内部，pre-train训练样本基本不可能覆盖到，那一定要进行continue pre-train。
如果你的领域任务数据量较大（token在1B以上），并只追求领域任务的效果，不考虑通用能力，建议进行continue pre-train。

b.是选择chat模型 还是base模型
如果你有一个好的base模型，在base模型基础进行领域数据的SFT与在chat模型上进行SFT，效果上差异不大。基于chat模型进行领域SFT，会很容导致灾难性遗忘，在进行领域任务SFT之后，模型通用能力会降低，如只追求领域任务的效果，则不用考虑。如果你的领域任务与通用任务有很大的相关性，那这种二阶段SFT会提升你的领域任务的效果。如果你既追求领域任务的效果，并且希望通用能力不下降，建议选择base模型作为基座模型。在base模型上进行多任务混合训练，混合训练的时候需要关注各任务间的数据配比。

在训练数据方面，Base模型是基于海量语料库进行的无监督学习。它从大量文本中学习语言模式和知识，而不需要人工标注或监督。相比之下，Chat模型则是在指令微调的有监督学习下进行训练的。这意味着它使用人工标注的数据集进行训练，以便更好地理解和响应特定指令。

在应用场景上，Base模型主要用于无监督学习任务，如文本分类、情感分析、摘要生成等。这些任务主要关注文本内容的理解和处理，而不需要对特定指令做出响应。相反，Chat模型则主要用于指令学习任务，如问答系统、对话生成、智能客服等。在这些任务中，模型需要理解和响应人类的指令，以提供准确和有用的信息。

在模型特性上，Base模型预训练之后没有做任何调整。它提供了基本的语言理解和生成能力，但可能需要针对特定任务进行微调或优化。而Chat模型则是在Base模型上进行微调的版本，它通过指令微调和人工反馈强化学习等方法，使模型更加符合人类的价值观和指令要求。

总之，Base和Chat是两种不同的大模型，它们在训练数据、应用场景和模型特性上有所区别。Base主要用于无监督学习任务，而Chat则专注于指令学习任务。在模型特性上，Chat通常在Base上进行微调，以更好地适应特定任务的需求。

根据以上区别，在选择基座模型时也要考虑数据量和任务差别难度，对于训练数据量少的，任务和基座大模型比较优秀能力接近的选chat模型。对于训练数据量比较大，或任务与chat版本的相似的能力比较差，选择base版本。

另一种说法是**base模型可以更方便做知识注入**，而chat版本是做过对其的，不好做知识注入。所以基于base的SFT可以做的上限更高，更方便做知识的注入，而**基于chat模型的SFT是做的样式学习或者指令学习**。但是base也存在没有对其的风险，输出可能和希望有差距，需要更多的调优和对齐。

## 关于微调的一些新的尝试：
全参数SFT+LoRA微调模式：尝试了将全参数SFT与LoRA进行结合，具体微调的方式：前10%-30% step 采用全参数SFT的方式，后面的step采用LoRA的方式，比单纯的LoRA要更加稳定，比全部采用全量参数SFT更加节省资源。该方式动机，通常来讲，大模型微调的时候，前面step中，模型参数变化最快，loss也是下降的最快，后面step模型参数变化幅度越来越小，loss变化幅度变小，逐渐收敛。因此，可以在微调的最开始step采用全参数SFT，让模型能够尽快的学习到指令，后面采用LoRA的方式，让模型能够更好的遵循指令。全参数SFT与LoRA 训练step配比，可以依据自己的tokens来定。尝试LoRA的升级版本：DoRA，DoRA: Weight-Decomposed Low-Rank Adaptation（http://arxiv.org/abs/2402.09353），目前还没有看出DoRA较LoRA的特别大的优势，后面还需做更多的实验进行观察。


## qwen微调
https://qwen.readthedocs.io/en/latest/training/SFT/example.html


