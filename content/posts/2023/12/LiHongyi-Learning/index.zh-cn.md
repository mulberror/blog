---
title: "「李宏毅机器学习」课程听课记录"
subtitle: ""
description: ""
slug: 333538
date: 2023-12-20T15:37:19+08:00
lastmod: 2023-12-20T15:37:19+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ["机器学习","深度学习","Transformer","ChatGPT","预训练","回归","强化学习"]
# 分类
categories: ["学习"]
# 合集(如果下面这一行注释掉，就不会显示系列为空了)
# collections: [""]
# 从主页面中去除
hiddenFromHomePage: false
# 从搜索中去除
hiddenFromSearch: false

lightgallery: false

# 否开启表格排序
table:
  sort: false
toc:
  enable: true
  auto: true
expirationReminder:
  enable: false
  # ...
code:
  copy: true
  # ...
edit:
  enable: false
  # ...
math:
  enable: true
  # ...
mapbox:
  accessToken: ""
  # ...
share:
  enable: true
  # ...
comment:
  enable: true
  # ...
library:
  css:
    # someCSS = "some.css"
    # 位于 "assets/"
    # 或者
    # someCSS = "https://cdn.example.com/some.css"
  js:
    # someJS = "some.js"
    # 位于 "assets/"
    # 或者
    # someJS = "https://cdn.example.com/some.js"
seo:
  images: []
  # ...
---

## ChatGPT 原理剖析

ChatGPT 的社会化：

1. 学会文字接龙
2. 人类引导文字接龙方向
3. 模仿人类喜好
4. 用增强式学习向模拟老师学习

### 预训练（Pre-train）

ChatGPT 真正在做的事情本质上是**文字接龙**，将其看成一个函数 $f(x)$，其中的 $x$ 自变量可以是用户输入的一个语句，得到的函数就是接下来会回答的每一种答案的概率。每次产生答案的过程是从这样一个分布中进行**随机取样**。

ChatGPT 的输入还会包括同一则对话中的历史记录，这样就实现了和上文的联系。做文字接龙的模型就是语言模型。

ChatGPT 背后的关键技术：预训练（Pre-train）或称自监督式学习（Self-supervised Learning）、基石模型（Foundation Model）。

ChatGPT，Chat：聊天，G：Generative，P：Pre-trained，T：[Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA)

> 监督式学习：人类提供大量的成对的资料，让机器自己寻找函数 $f$。监督式学习所提供的资料较为有限，ChatGPT 所使用的技术可以无痛制造成对资料。
> 预训练：在多种语言上做预训练后，只要教某一个语言的某一个任务（finetune过程），就可以自动学会其他语言的同样任务。
> 增强式学习（Reinforcement Learning, RL）：简而言之是人类告诉机器给出的答案是好还是不好。（较为节省人力，且适用于人类都不知晓答案的时候）

### ChatGPT 带来的研究问题

1. 如何提出需求，对 ChatGPT 进行“催眠”（学术界叫做 Prompting），
2. 对于模型的调整（会不会因为某一个调整，然后导致其他的更多问题，），这个研究题目就是叫做**Neural Editing**。
3. 检测 AI 生成的物件。如何用模型侦测一段文字是不是 AI 生成的。
4. 安全性问题。清空 ChatGPT 的记录，研究题目“Machine Unlearning”。

### 和 ChatGPT 玩文字冒险游戏

将文字游戏的一个开始文本输入到 ChatGPT 中，然后将 ChatGPT 生成的文字冒险游戏叙述输入到 Midjourney 中再生成游戏插图。（很有意思的一个游戏过程）

## Regression（回归问题）

[知乎 # 机器学习-回归问题(Regression)](https://zhuanlan.zhihu.com/p/127972563)

举例应用：

- 预测股票市场
- 自动驾驶
- 视频推荐系统
- 课堂提及：预测宝可梦的战斗力 CP 值（Combat Power），假如说我们研究的问题是需要计算一只宝可梦在进化后的 CP 值，我们现在的已知量是当前宝可梦的一些数据。

解决该回归问题的步骤：

1. 构建模型（也就是得到一组函数 $f(x_i)=y$）
线性模型（lLinear model）：$y=b+\sum w_i x_i$，其中的取出的 $x_i$ 叫做 feature，$w_i$ 叫做 weight，$b$ 叫做 bias。

2. 计算 Goodness of Function。
我们有了实际的十个用于训练的宝可梦数据 $\{ (x^i, \hat{y}^i)\}$
需要定义一个**损失函数（Loss Function）** $L$，损失函数的输入为一个函数，输出为当前输入函数的好坏值（坏的值），$L(f)=L(w, b)$。损失函数有多种类型，课程以其中一种为例，$L(f)=\sum (\hat{y}^n-(b+w\cdot x^n_{cp}))^2$，也就是估测误差的平方和。

3. Best Function
找到**损失函数**最小的一个函数 $f^*$，$f^*=arg \ min_f L(f)$，如果是以上面那一个为损失函数的话，那么可以用线性代数进行求解。
另外一种办法叫做**梯度下降（Gradient Descent）**，适用于任何**可微分**的损失函数：
假设我们现在考虑的是 $L(w)$ 的损失函数，首先随机选取一个 $w_0$ 作为当前 $w$，然后计算其在 $w_0$ 处的微分值 $\frac{dL}{dw}|_{w=w_0}$，如果当前这个微分值是负数，那么就需要增加当前 $w$，反之同理。其中的改变量为 $\eta \cdot \frac{dL}{dw}|_{w=w_0}$，其中这个参数 $\eta$ 叫做 learning rate。梯度下降就是进行这样的一个迭代过程。**如果是多元问题，那么就用偏微分来求解。**
对于多元函数来说是需要考虑极值和最值的区别的，如果是线性的损失函数，那么是不需要考虑的，因为损失函数的图像是一个凸包。

>简单来说，其实就是构建一个带参的模型函数去拟合给定的数据集，然后有一个类似于范数一样的函数 $L$（损失函数）来评估，可以用梯度下降或者数理计算的方式来得到参数的最佳值。（模型可能会出现过拟合的情况）
>Regularization：修改损失函数为 $L(f)=\sum (\hat{y}^n-(b+w\cdot x^n_{cp}))^2 + \lambda \sum (w_i)^2$，当后面这个越接近 0，这个函数是越平滑的，函数对输入的改变越不敏感，可以一定程度上防止干扰。 那么就需要考虑这个函数的平滑程度。

## Classification（分类）

有好几种方法 Perceptron 和 SVM。
主要分成几个阶段，类似于 regression 的过程。

- 创建 model function 模型函数。
- 对于每一个 class 进行 Maximum Likelihood 找到对应的模型函数 model function（k 维的一个正态分布）。
- 然后使用贝叶斯公式计算出出现了当前这个 object x 时，属于 class 1 的概率。

有一个优化的过程就是将所有 class 计算概率的正态分布的矩阵设置为一样的，这样可以大大的提高 classifier 的正确性。
如果不是用 k 维进行考虑，而是将 k 维拆开后单独考虑最后乘在一起，这样的一个模型叫做 Naive Bayes Classifier（较为初级的贝叶斯分类器）。
将计算概率的贝叶斯公式如果写成

$$\frac{1}{1+exp(-z)}
$$

其中 $z=wx+b$ 的形式（$w,b$ 中间有矩阵关系的推导）。

## 生成式学习的两种策略

> 生成语音：http://arxiv.org/abs/2210.02303
> 生成声音：https://arxiv.org/abs/2301.12503

### 各个击破

各个击破的模型的专业术语：Autoregressive(AR) model

### 一次到位

专业术语：Non-autoregressive (NAR) model

## 对于大语言模型的两种不同的期待

让机器学会读题目描述 **(Instruction Learning)** 和题目范例 **(In-context Learning)**。

### 成为通才

The Natural Language Decathlon：arxiv.org/abs/1806.08730
Ask Me Anything：arxiv 1506.07285
Is ChatGPT A Good Translator：arxiv 2301.08745
How Good Are GPT Models at Machine Translation? arxiv 2302.09210

### 给预训练模型进行改造

#### 加 Head

加入一个 Head:youtu.be/gh0hewYkjgo

#### 微调(Finetune)

跑梯度下降

#### Adapter

在语言模型中加入一些插件，微调 Adapter 中的参数。
BERT 模型的 Adapter: https://adapterhub.ml/
https://arxiv.org/abs/2210.06175
如果有了 Adapter 那么就几乎不需要去微调语言模型中的参数了。

### In-context Learning（上下文学习）

给大语言模型一些提前的示例。
Rethink the Role of Demonstration: arxiv 2202.12837，大语言模型本来就可以做情感分析，只是需要需要被指出需要做情感分析。
**所以是不是当前的语言模型存在着瓶颈？需要更加强大的基础模型，有没有可以完美类似于人类大脑的自我成长的模型，需要了解神经学。**
arxiv 2212.10559, arxiv 2211.15661, arxiv 2303.03846
[知乎 【论文解读】in-context learning到底在学啥？](https://zhuanlan.zhihu.com/p/484999828)

### Instruction-tuning

简单的来说就是人类做题过程中的审题过程。
arxiv 2110.08207
FLAN(Finetuned Language Net): arxiv 2109.01652

### Chain of Thought (CoT) Prompting

在给机器材料的同时，除了问题的描述和答案以外，给出一些求得问题的**过程**。
一般会有一个标志性的语句，比如说 Think about it step by step.
论文：In-context 下的 arxiv 2201.119037
arxiv 2205.11916 告诉机器 lets's think step by step. (Zero-shot-CoT)
Self-consistency (2203.11171)
arxiv 2205.10625

### 机器自己寻找 Prompt (相当于给机器催眠)

- Using reinforcement learning(arxiv 2206.03931)
构建一个 Reward Function，告诉语言模型当前给出的 output 的得分（是好还是坏）。
- Using an LM to find Promt(arxiv 2211.01910)
这难道不就是 In-context learning 吗？

## 大模型+大资料

Scaling Laws for Neural Language Models: arxiv 2001.08361
模型大小感觉可以简单理解为模型中的参数个数，资料就是用于给模型学习矫正参数的总数据大小。

### 大模型的 Emergent Ability

[知乎# 大模型的涌现能力(Emergent Abilities of LLM)](https://zhuanlan.zhihu.com/p/609339534)
arxiv.org/pdf/2206.07682
需要观察语言模型中间给出的过程，如果用对和错来评判某个答案的话，我们并不知道中间过程的变化。
scratchpad: arxiv abs 2112.00114
Language Models (Mostly) Know What Theyt Know: arxiv abs 2207.05221
在做 Chain of thought 中这种现象会非常明显
>但是注意图表下面的横轴，横轴并不是均匀变化的。

模型自己知不知道自己在瞎掰（Calibration）：机器预测下面的一个字或者词的几率。arxiv abs 2207.05221，只有在模型足够大的情况下，才会有 Calibration。
Inverse Scaling Benchmark, arxiv abs 2211.0201 （老师上课：一知半解吃大亏），小模型瞎猜，中模型一知半解，大模型会有点逻辑，这一点和人类从小到大学习的过程有点相似。

### 资料的重要性

When Do You Need Billions of Words of Pretraining Data? arxiv 2011.04946
需要同时使用语言知识和世界知识。

#### Data Preparation

Scaling Language Models: Method, Analysis & Insights from Training Gopher: arxiv abs 2112.11446

1. 过滤有害内容
2. Text Extration: 去除 HTML tag（保留文本内容，项目符号）
3. Quality Filtering，筛选高质量资料
4. 去除重复资料
5. Test-set Filtering，保证实验的严谨（训练资料中不能用测试资料）。
去除重复资料重要性，Deduplicating Training Data Makes Language Models Better, arxiv abs 2107.06499

#### 在固定的运算资源中选择模型变大还是资料量变多

问题：选择**小模型大资料**还是**大模型小资料**，或者是如何寻找模型和资料之间的平衡点。（人的大脑和看进去的知识来类比，学而不思，思而不学）
arxiv abs 2203.15556: 讨论了关于这个问题的最佳比例，也就是模型和资料大小的比例关系。
Instruction-tuning: arxiv abs 2210.11416
Instruct GPT: arxiv abs 2203.02155

### KNN LM (K 近邻)

和一般的语言模型有点不一样。
arxiv abs 1911.00172
简单的过程可以概括为，比如说我们的资料是这样的格式 $x\rightarrow y$，那么我们就把 $x$ 放到模型中输出一个向量 $v$，那么现在的对应关系就是 $v \rightarrow y$，那么我们如果提出一个问题 $q_x$，我们计算出问题对应的向量 $q_v$，然后和每一个 $v$ 计算一个距离函数，然后对这个得到的距离进行一个概率的分布的计算。
KNN LM 得出答案太花时间了。

### Retrieval Enhanced Transformer

proceedings.mlr.press/v162/borgeaud22a.html
一个擅长记忆的模型，就比如人在开卷考试的时候查询资料。

### GPT-4

技术细节：论文中并没有更多的技术细节，还是使用了 Transformer 架构（但是给了个 style），Reinforcement Learning Human Feedback (RLHF) 来 finetune。
应用：

- 会更多的语言
- Inverse Scalling 测试达到了 100% 正确率
- Calibration curve: GPT 知道自己不知道（当 GPT 的信心和答案的正确率几乎成 1:1 的关系）
- 影像输入（如何加入一个 Adapter），老师给的猜想：Caption Generation，OCR，Image Encoder（转变成一个新的语言或者 Kosmos）转变成向量

## 图像生成模型

现在影像生成的模型都有一个共同的套路，会给一个额外的输入，也将相当于是给 SD 中输入的 tag，我们需要根据这些额外的输入，从原先设定要的一些分布中采样出向量，输入到模型中才可以产生需要的图片。

- VAE: youtu.be/8zomhgKrsmQ, encoder 和 decoder 相互还是独立的。
- Flow-based: 具体看课程参考资料（说实话，有点没看懂，大概是能够神奇的构造一个能够求逆的矩阵），往 encoder（给图片编码）输入一个图片输出一个向量，将很多图片输入后这些向量的分布是一个 normal distribution，相当于 encoder 和 decoder 恰好为互逆操作。
- Diffusion: 主要操作是 denoise (可视为 decoder) 和 addnoise (可视为 encoder)
- GAN: 构造了一个 discriminator 计算给出的 distribution 和计算出来的 distribution 的差距？还是生成图片的差距？**可以看老师给的 GAN 的系列讲解** 感觉还是分布的差距
有 VAE+GAN (arxiv abs 1512.09300), Flow+GAN (arxiv abs 1705.08868), Diffusion+GAN (arxiv abs 2206.02262)

### Diffusion Model

Denoising Diffusion Probabilistic Models (DDPM): arxiv abs 2006.11239

#### 生成图片的过程

1. 先从分布（可能是高斯分布）采样出一个大部分是杂声的向量，这个向量的维度必须和你要生成的图片的大小一致）。
2. 然后会有一个 denoise 的 net-work，每一次 denoise 就会去除掉一定程度的噪声，这个步骤也被称为 Reverse Processs。（就像是雕塑的时候，最终的成品已经在材料中了，做的工作就是去掉不需要的部分）
Denoise 中还是输入一个现在的迭代次数

#### Denoise

 Denoise 中输入一个带杂讯的图片和一个迭代次数，然后会由 Noise Predicter 在中间生成一个**预测的杂讯**，然后在原先的图片中减去掉这个预测的部分，得到一个去掉一部分杂讯的图片。
 为什么不直接生成一个去掉一部分杂讯的图片，可能预测杂讯和预测图片的难度是不一样的。
 训练的过程就是给训练资料加入完全随机的噪音，这样的过程叫做 Forward Process (Diffusion Process)。

#### Text-to-Image

laion.ai/blog/laion-5b

### Stable Diffusion（这一部分没有特别的明白，需要多次学习）

#### Framework

1. 首先是由一个 Text Encoder 将输入的文本转换为一个向量。
2. 将杂讯和上面的向量输入到 Generation Model 中，得到一个中间产物。
3. 将这个中间产物输入到 decoder 中迭代除去中间的杂讯。
SD: arxiv abs 2112.10752

#### DALL-E series

arxiv abs 2204.06125, arxiv abs 2102.12092

#### Imagen

imagen.research.google
arxiv abs 2205.11487

#### 文字处理 Text-Encoder

arxiv abs 2205.11487
文字的 Encoder 的大小变化对图像质量的影响较大，Diffusion Model 的大小变化对图像质量的影响较小。

#### FID

这个是用于评价影像生成的好坏的一个指标。
arxiv abs 1706.08500
将所有的资料图片和机器生成的图片都丢进到一个训练好的 CNN 模型中，然后将真实的图片和生成的图片分开，计算两组图片之间的距离，假设每一组图片都是一个高斯分布，然后计算这两个分布的距离，这个距离只要越小就说明生成的质量越好。

#### Contrastive Language-Image Pre-Training (CLIP)

arxiv abs 2103.00020
将输入的文本用 Text-Encoder 转换为一个向量，用 Image-Encoder 转换为一个向量，如果是要对应的，那么这两个向量就需要很接近。
