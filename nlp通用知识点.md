# 为何使用 Logits？

在神经网络和尤其是在自然语言处理任务中，logits 是模型最后一层的输出，未经过归一化（例如 softmax 函数）处理的原始预测值。这些值通常用于计算损失函数，并在后续步骤中转换成概率。

详细解释
当使用如 BERT 这样的预训练模型进行分类任务时，最后一个全连接层（通常称为分类头）会输出每个类别的一个得分。这些得分是：

线性的：它们直接来自于模型最后的线性层，没有应用任何非线性激活函数。
未归一化的：这意味着输出值可能是任意范围内的实数，并不直接代表概率。
多维的：对于多分类任务，每个样本的输出是一个向量，其中每个元素对应一个类别的得分。
使用 logits 而不是直接输出概率的原因包括：
- 数值稳定性：在计算损失函数（如交叉熵损失）时，直接从 logits 到概率的转换（通过 softmax）可以避免数值不稳定问题，如下溢或指数爆炸。
- 效率：在某些情况下，合并 softmax 和交叉熵损失到单个操作中（比如 PyTorch 中的 nn.CrossEntropyLoss），可以减少计算步骤和提高数值稳定性。
- 灵活性：提供 logits 允许用户根据具体需求选择合适的归一化方法（softmax、sigmoid等）或直接使用 logits 进行操作。
>logits例子：假设样本数量：3，类别数量：4
logits = torch.tensor([
    [2.0, 1.0, -1.0, 0.5],
    [0.5, -0.5, 1.0, 2.0],
    [-1.5, 1.0, 2.5, 0.0]
])
对于第一个样本，模型认为它最可能属于第一个类别（得分最高为2.0）。
对于第二个样本，模型认为它最可能属于第四个类别（得分最高为2.0）。
>对于第三个样本，模型认为它最可能属于第三个类别（得分最高为2.5）。

# torch模型保存
## 状态字典(state_dict)
**一个状态字典就是一个简单的 Python 的字典，其键值对是每个网络层和其对应的参数张量。** PyTorch 中，一个模型(torch.nn.Module)的可学习参数(也就是权重和偏置值)是包含在模型参数(model.parameters())中的。模型的状态字典只包含带有可学习参数的网络层（比如卷积层、全连接层等）和注册的缓存（batchnorm的 running_mean）。优化器对象(torch.optim)同样也是有一个状态字典，包含的优化器状态的信息以及使用的超参数。

    #打印模型的状态字典
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
## 模型的保存与加载
[PyTorch | 保存和加载模型](https://zhuanlan.zhihu.com/p/82038049)
[torch原文](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
提供哪些模型信息可以复现模型
- 模型代码：
（1）包含了我们如何定义模型的结构，包括模型有多少层/每层有多少神经元等等信息；
（2）包含了我们如何定义的训练过程，包括epoch batch_size等参数；
（3）包含了我们如何加载数据和使用；
（4）包含了我们如何测试评估模型。
- 模型参数：提供了模型代码之后，对方确实能够复现模型，但是运行的参数需要重新训练才能得到，而没有办法在我们的模型参数基础上继续训练，因此对方还希望我们能够把模型的参数也保存下来给对方。
（1）包含model.state_dict()，这是模型每一层可学习的节点的参数，比如weight/bias；
（2）包含optimizer.state_dict()，这是模型的优化器中的参数；
（3）包含我们其他参数信息，如epoch/batch_size/loss等。
- 数据集：
（1）包含了我们训练模型使用的所有数据；
（2）可以提示对方如何去准备同样格式的数据来训练模型。
- 使用文档：
（1）根据使用文档的步骤，每个人都可以重现模型；
（2）包含了模型的使用细节和我们相关参数的设置依据等信息。

### 只保存模型参数字典（推荐）
torch.save大致来说，它是把每一个Python object使用pickle进行保存，然后打包成一个zip压缩文件，同时对pytorch内部的tensor存储、内存共享等细节进行了部分优化。    

    #保存
    torch.save(the_model.state_dict(), PATH)
    #读取
    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))

### 保存整个模型

    # 保存模型
    PATH = "entire_model.pt"
    # PATH = "entire_model.pth"
    # PATH = "entire_model.bin"
    torch.save(model, PATH)
    #读取
    the_model = torch.load(PATH)


## torch模型格式
[Pytorch格式 .pt .pth .bin .onnx 详解](https://zhuanlan.zhihu.com/p/620688513)
### .pt .pth格式
是 PyTorch 的默认保存格式。
### .bin格式
.bin文件是一个二进制文件，可以保存Pytorch模型的参数和持久化缓存。.bin文件的大小较小，加载速度较快，因此在生产环境中使用较多。通常由 Hugging Face 的 Transformers 库使用，主要用于保存预训练模型的权重。

### .onnx格式（Open Neural Network Exchange）
可以通过PyTorch提供的torch.onnx.export函数转化为ONNX格式，这样可以在其他深度学习框架中使用PyTorch训练的模型。

|格式|优点|缺点|怎么选择格式
|----|-----|-----|-----|
|.pt 或 .pth 格式| <li>原生支持： 直接由 PyTorch 提供支持，无需额外转换。<li> 灵活性： 可以选择保存整个模型（包括架构）或仅状态字典（推荐）。 <li>易于加载： 直接使用 PyTorch API 加载。|<li> 依赖 PyTorch： 加载模型需要 PyTorch 环境，不适合非 Python 环境或非 PyTorch 框架。<li> 版本依赖性： 保存的模型可能依赖特定版本的 PyTorch，不同版本间可能存在兼容问题。|如果在 PyTorch 环境中工作且无需将模型迁移到其他框架， 使用 .pt 或 .pth 格式是最方便的。
|.bin 格式| <li>优化存储： 通常用于存储模型权重，文件大小适中。<li>跨框架使用：虽然主要由 Transformers 库使用，但理论上可用于其他环境，只要适当处理权重数据。|<li>需要模型架构信息： 通常只保存权重，加载时需要先定义好模型架构。<li>库依赖性： 主要由 Hugging Face 库使用，与库的特定功能强相关。|如果你使用 Hugging Face 的 Transformers， .bin 文件是标准选择。
|.onnx 格式|<li>框架无关性： 可以在支持 ONNX 的任何平台上加载和运行，如 TensorFlow, Caffe2, Microsoft's CNTK,等。<li>广泛支持： 许多硬件加速器和优化工具支持 ONNX，便于部署。<li>性能优化： 可以利用 ONNX 运行时进行优化，提高模型执行效率。|<li>转换过程： 需要从原始框架（如 PyTorch）转换为 ONNX，这个过程可能出现兼容性问题，尤其是对于复杂的自定义层或特殊的操作。<li>更新滞后： ONNX 格式可能不总是能够支持最新的层或操作。|如果需要在不同的平台或框架之间迁移模型， 或需要在特定设备上部署模型，如使用 TensorFlow Serving，或者需要模型在多种硬件上运行，则选择 .onnx 格式会更合适。
