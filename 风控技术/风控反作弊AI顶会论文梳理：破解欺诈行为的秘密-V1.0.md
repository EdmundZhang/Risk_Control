> 持续整理和解读风控反作弊方面的顶会论文，可以关注、收藏。  
> 评选标准：
> 
> 1.  有代码、方便复现的论文优先考虑
> 2.  对账号、营销薅羊毛、支付风控、广告反作弊等场景问题有直接启发的论文优先考虑

| 名称  | 方向  | 论文数 | 有代码论文数 |
| --- | --- | --- | --- |
| WWW | Multiple Fields  | 16  | 7   |
| KDD | Data Mining | 8   | 3   |
| CIKM | Data Mining | 8   | 5   |
| AAAI | Artificial Intelligence | 5   | 4   |
| ICDM | Data Mining | 3   | 3   |
| WSDM | Data Mining  | 2   | 1   |
| VLDB | Data Bases  | 1   | 1   |
| IJCAI | Artificial Intelligence | 1   |     |
| NeurIPS | Machine Learning | \-  |     |
| ICML | Machine Learning | \-  |     |

- 已解读论文  
    [从虚假点赞到恶意评论：FRAUDAR算法如何一路斩妖除魔？](https://zhuanlan.zhihu.com/p/687094360)  
    SliceNDice - 2024年3月  
    FlowScope - 2024年4月

# 代码简单解读(模板)

- 适用场景
- 算法效果
- 论文创新的点
- 算法的优化目标
- 算法迭代过程
- 算法在风控场景的应用

> - 适用场景
> - 算法效果
> - 论文创新的点
> - 算法的优化目标
> - 算法迭代过程
> - 算法在风控场景的应用

# WWW：International World Wide Web Conference

1.  **SliceNDice: Mining Suspicious Multi-attribute Entity Groups with Multi-view Graphs** *2019年*

> **算法原理**：捕捉具备大量共同属性的用户，比如在同一时间和地点创建的帐户、宣传相同言论和转发类似文章的账户。  
> **输入**：实体+属性1...+属性N  
> **算法功能**：根据共同属性构造边，合并各属性构造的子图，形成Multi-View Graph.随后挖掘异常关联子图（"dense" subnetwork）.  
> **实验数据集大小**：Snapchat advertiser platform（23W实体 × 12属性），发现2435异常实体，precision 89%

- [Paper](https://arxiv.org/abs/1908.07087)
- [Code](https://github.com/hamedn/SliceNDice)

2.  **Spam Review Detection with Graph Convolutional Networks** *2019年*

> **算法原理**：这篇论文中，作者构建了两种图结构：异构图Xianyu Graph和同构图Comment Graph。  
> **输入**:异构图 Xianyu Graph：  
> **节点**由用户（U）、物品（I）和评论（E）组成。  
> 同构图 Comment Graph：节点由评论（E）组成。  
> **边**表示相似的评论。通过构建近似KNN图算法，找到每个评论的K个最近邻居，从而形成同构图。  
> **算法功能**：  
> 使用异构GCN对Xianyu Graph进行embedding  
> 采用归纳GCN对Comment Graph进行embedding  
> 采用TextCNN对评论文本进行embedding  
> 一起输入到GCN-based Anti-Spam（GAS）模型，进行有监督分类。

- [Paper](https://arxiv.org/abs/1908.10679 "阿里的论文 CIKM 2019")
- [Code](https://github.com/safe-graph/DGFraud)

3.  **Heterogeneous Graph Neural Networks for Malicious Account Detection** *2018年*

> 算法功能：考虑设备聚合和活动聚合，构建图结构embedding进行分类。本质是图卷积网络的一种变体  
> 效果：GBDT+Graph 方法与 GBDT+Node2Vec 方法相比效果相似，而 GCN 的效果优于 GBDT+Graph 和 GBDT+Node2Vec。  
> 作者提出的 GEM 方法在多种方面都优于 GCN，因为它处理了异构设备类型，而 GCN 只能处理同质图，无法区分图中不同类型的节点。  
> GEM 还使用每种节点类型的聚合操作，而不是标准化操作，因此更好地模拟了底层的聚合模式。

- [Paper](https://dl.acm.org/doi/10.1145/3269206.3272010 "CIKM 2018")
- [Code](https://github.com/safe-graph/DGFraud)

4.  **FlowScope: Spotting Money Laundering Based on Graphs** *2020年*

> 输入：A（转账人）->M(中间人)->C(收款人)  
> 算法功能：找到其中可能涉及洗钱的团伙  
> 实验数据集大小：-

- [Paper](https://shenghua-liu.github.io/papers/aaai2020cr-flowscope.pdf)
- [Code](https://github.com/BGT-M/spartan2)

5.  **FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System** *2019年*

- Rui Wen, Jianyu Wang and Yu Huang

> - 适用场景  
>     检测应用商店中的**虚假评论、传播错误信息**等欺诈行为，需要少量的恶意样本输入。训练过程是**半监督**，检测是纯**无监督**。  
>     设计了一个双层GCN网络，用于从未标记的数据中检测出欺诈用户。  
>     类似的算法包括 **Fraudar**、**NetWalk**和**FraudEagle**等。
> - 算法效果  
>     FdGars在召回率上表现出色，达到了0.958+，优于其他基线方法。  
>     从腾讯公司收集了82,542名用户在2018年8月1日至3日期间发布的302,097条评论数据作为实验数据集。其中，将8月1日的评论作为训练集，2日和3日的评论作为测试集。  
>     基线方法：论文对比了包括逻辑回归（LR）、随机森林（RF）、DeepWalk和LINE在内的几种流行的分类和图结构基础方法在欺诈检测任务上的性能。实验结果表明，FdGars方法在召回率和精确率上均优于其他方法，且在实际应用中具有更高的计算效率。
> - 论文创新的点  
>     首次将图卷积网络（GCN）应用于这一领域。
> - 算法的优化目标  
>     **恶意样本特征**：作者提取了评论内容和评论者行为方面的特征。评论内容特征包括相似评论数量、特殊字符数量、评论长度、符号数量与长度的比例以及正则表达式黑名单。评论者行为特征包括连续评论天数和登录设备数量。  
>     **恶意样本提取**：基于行为特征，设计了一个标签函数来分类评论者。如果评论者在指定期间连续评论天数超过阈值且登录设备数量也超过阈值，则将其分类为欺诈者。通过多次数据分析，确定了连续天数和登录设备数量的阈值，以便识别高度可疑的欺诈者。  
>     **图构建**：评论的用户作为图中的节点，如果两位评论者评论了相同的App，则认为存在一条管理边。按以上方法将评论行为构成可挖掘的图结构。  
>     **半监督欺诈检测**：使用两层GCN进行半监督欺诈检测。

> - 算法在风控场景的应用  
>     **个人看法**：看着挺麻烦，对场景依赖也有点强，不确定ROI。

- [Paper](https://dl.acm.org/citation.cfm?id=3316586)
- [Code](https://github.com/safe-graph/DGFraud)

6.  **CatchCore: Catching Hierarchical Dense Subtensor** *2019年*

> - 适用场景  
>     **无监督**的方式检测多维数据（张量）中的层次密度子张量，即异常或者欺诈行为。并且能够识别出异常行为和有趣模式，如周期性攻击和动态研究人员组。
> - 算法效果  
>     CatchCore在检测密集子张量和异常模式方面优于顶级竞争对手的准确性。同时，CatchCore还成功地识别了一个具有强烈相互作用的层次结构研究人员合作组。  
>     （CC：CatchCore，D：D-Cube，M：M-Zoom，CS：CrossSpot）  
>     与D-Cube、M-Zoom、CrossSpot和CP分解（CPD）进行对比，\*\*  
>     **注入密度子张量检测**：论文描述了如何生成一个随机均匀的3维张量R，并在这个张量中注入了一个具有不同密度的200×300×100的密集子张量。CatchCore能检测到最佳基线1/10密度的子结构。（？？）  
>     **BeerAdvocate数据集**：CatchCore最优秀  
>     **TCP数据包网络入侵检测**：检测到Neptune攻击的分层行为模式
> - 论文创新的点  
>     现有方法通常是平面和独立的检测最密集的子张量，假设这些子张量是互斥的。许多现实世界的张量通常表现出层次结构。CatchCore可以有效地发现**层次结构的密集子张量**。  
>     CatchCore的**时空复杂度**均随张量的各个方面**线性增长**。  
>     论文通过实验展示了CatchCore具有随着输入规模的增长而线性或亚线性扩展的能力。作者通过改变一个因素（如元组的数量、属性的数量或属性的基数）来测量CatchCore的可扩展性，并发现CatchCore在元组数量和属性数量上呈线性增长，在属性基数上呈亚线性增长。这表明CatchCore的复杂性在定理7中所述的过于悲观。
> - 算法的优化目标  
>     最大化子张量的Entry-Plenum度量
> - 算法迭代过程
> - 算法在风控场景的应用

- [Paper](https://shenghua-liu.github.io/papers/pkdd2019-catchcore.pdf)
- [Code](https://github.com/wenchieh/catchcore/tree/master)

7.  **No Place to Hide: Catching Fraudulent Entities in Tensors** *2019年*

> - 适用场景  
>     论文原理：为了最大化利润，欺诈者必须在多次欺诈中共享或复用不同的资源，例如假账户、IP地址和设备ID。欺诈活动由于资源的共享，往往在张量中形成密集块。（ISG+D-Spot方案）
> - 算法效果  
>     理论上证明了D-Spot找到的子图在密度上至少是最优子图的一半，而传统方法只能保证1/N的密度。  
>     在实际图中，D-Spot的运行速度比现有的最先进算法快11倍。  
>     在Amazon数据集上，ISG+D-Spot准确地检测到了同步行为，即大量欺诈用户在短时间内为一小群产品创建大量假评论的行为。  
>     在Yelp数据集上，ISG+D-Spot在检测串通餐馆方面的准确性最高，因为它对ISG应用了更高的理论界限。  
>     在DARPA和AirForce数据集上，ISG+D-Spot通过为每个IP地址分配特定的可疑分数，有效地识别了恶意IP地址和恶意连接。
> - 论文创新的点  
>     传统的密集块检测方法无法有效识别**在张量的所有维度上不具有高密度**但在**子集维度上具有高密度**的密集块，即“隐藏密集块”（hidden-densest blocks）。
> - 算法的优化目标  
>     ISG能够将张量中的复杂关系转换为图结构。  
>     点：待检测的对象，比如欺诈用户。  
>     **成对值共享 构造边**：如果两个节点在某个维度有相同的值，则具有一条边。  
>     **自我值共享 构造边**：对于节点ui，如果它在某个维度上多次出现相同的值a，那么具备一条边。
> - 算法迭代过程  
>     **信息共享图（ISG）**：将张量中的值共享转换为加权边或节点（实体）的表示。  
>     **D-Spot算法**：用于在ISG上快速并行地找到多个最密集的子图。  
>     总体上，复杂度与边的个数成线性关系。
> - 算法在风控场景的应用

- [Paper](https://arxiv.org/pdf/1810.06230.pdf)
- [Code](https://proceedings.mlr.press/v71/ram18a/ram18a.pdf)

8.  HiDDen: Hierarchical Dense Subgraph Detection with Application to Financial Fraud Detection

- [Paper](https://www.public.asu.edu/~hdavulcu/SDM17.pdf)

9.  FairPlay: Fraud and Malware Detection in Google Play

- [Paper](https://arxiv.org/abs/1703.02002)

10. Fraud Transactions Detection via Behavior Tree with Local Intention Calibration

- [Paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403354)

11. Identifying Anomalies in Graph Streams Using Change Detection

- [Paper](https://www.mlgworkshop.org/2016/paper/MLG2016_paper_12.pdf)

12. HitFraud: A Broad Learning Approach for Collective Fraud Detection in Heterogeneous Information Networks

- [Paper](https://arxiv.org/abs/1709.04129)

13. Friend or Faux: Graph-Based Early Detection of Fake Accounts on Social Networks

- [Paper](https://arxiv.org/abs/2004.04834)

14. Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection (WWW 2021)

- Yang Liu, Xiang Ao, Zidi Qin, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He
- [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3442381.3449989)

15. Financial Defaulter Detection on Online Credit Payment via Multi-view Attributed Heterogeneous Information Network (WWW 2020)

- Qiwei Zhong, Yang Liu, Xiang Ao, Binbin Hu, Jinghua Feng, Jiayu Tang, Qing He
- [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3366423.3380159)

16. Friend or Faux: Graph-Based Early Detection of Fake Accounts on Social Networks (WWW 2020)

- Adam Breuer, Roee Eilat, Udi Weinsberg
- [\[Paper\]](https://arxiv.org/abs/2004.04834)

# KDD：ACM SIGKDD Conference on Knowledge Discovery and Data Mining

1.  \[**Best Paper**\]2026-FRAUDAR: Bounding Graph Fraud in the Face of Camouflage

- 论文解读: [从虚假点赞到恶意评论：FRAUDAR算法如何一路斩妖除魔？](https://zhuanlan.zhihu.com/p/687094360)
- Bryan Hooi-Carnegie Mellon University
- [Papers](https://www.semanticscholar.org/paper/FRAUDAR%3A-Bounding-Graph-Fraud-in-the-Face-of-Hooi-Song/2852982175beeb92e14127277cb158c4cb31f5c5)
- [Code](https://github.com/safe-graph)

2.  \[**Best Paper**\]**Adversarial Attacks on Neural Networks for Graph Data** *2018年*
> - 适用场景  
>     论文提出了一种针对图数据的神经网络模型的对抗性攻击方法，名为Nettack。
> - 算法效果  
>     Nettack通过针对节点特征和图结构生成对抗性扰动，显著降低了节点分类的准确性。实验表明，即使只进行少量扰动，也会导致目标节点被错误分类。此外，这些攻击具有迁移性，能够成功应用于其他先进的节点分类模型，并且即使在对图的知识有限的情况下也能成功。
> - 论文创新的点
> - 算法的优化目标
> - 算法迭代过程
> - 算法在风控场景的应用
- Daniel Zügner-Technical University of Munich;
- [Papers](https://arxiv.org/pdf/1805.07984.pdf)
- [Codes](https://github.com/danielzuegner/nettack)

3.  **Collective Opinion Spam Detection: Bridging Review Networks and Metadata** *2015*

> - 适用场景  
>     识别**虚假评论**、**可疑用户**以及**被虚假评论影响的产品**。
> - 算法效果
> - 论文创新的点  
>     算法结合了元数据（文本、时间戳、评分）和关系数据（评论网络）。 
>     可以无缝集成标签信息，适用于半监督场景
> - 算法的优化目标
> - 算法迭代过程
> - 算法在风控场景的应用

- Shebuti Rayana, Leman Akoglu
- [\[Paper\]](https://www.andrew.cmu.edu/user/lakoglu/pubs/15-kdd-collectiveopinionspam.pdf)
- [\[Code\]](https://github.com/safe-graph/UGFraud)

4.  \[**Best Paper**\]**Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping** *2012*
    - Thanawin Rakthanmanon, University of California, Riverside;
    - [Papers](https://www.semanticscholar.org/paper/Searching-and-Mining-Trillions-of-Time-Series-under-Rakthanmanon-Campana/c5a8d4145dc3530d437c93112b790bddb1927b59)
5.  \[**Best Paper**\]**A probabilistic framework for semi-supervised clustering** *2004*
    - Sugato Basu University of Texas at Austin;
    - [Papers](https://www.semanticscholar.org/paper/A-probabilistic-framework-for-semi-supervised-Basu-Bilenko/01d5bf24c0b35d9a234d534bf69924fa16201dee)
6.  \[**Best Paper**\]**Pattern discovery in sequences under a Markov assumption** *2002*
    - Darya Chudova & Padhraic Smyth, University of California, Irvine
    - [Papers](https://www.semanticscholar.org/paper/Pattern-discovery-in-sequences-under-a-Markov-Chudova-Smyth/cf640511fb17f544b99a80470b018b7ea0d7b7b4)
7.  **Intention-aware Heterogeneous Graph Attention Networks for Fraud Transactions Detection** *2021*
- Can Liu, Li Sun, Xiang Ao, Jinghua Feng, Qing He, Hao Yang
- [\[Paper\]](https://dl.acm.org/doi/10.1145/3447548.3467142)

8.  **Live-Streaming Fraud Detection: A Heterogeneous Graph Neural Network Approach** *2021*
- Haishuai Wang, Zhao Li, Peng Zhang, Jiaming Huang, Pengrui Hui, Jian Liao, Ji Zhang, Jiajun Bu
- [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3447548.3467065)

# CIKM：ACM Conference on Information and Knowledge Management

1.  **Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters** *2020*
> 综述：半监督图异常检测
> 
> - 适用场景  
>     虚假评论检测、金融欺诈检测、移动欺诈检测等。这些场景通常涉及大量的图结构数据，其中欺诈者可能通过伪装行为来逃避传统的检测机制。
> - 算法效果  
>     优于GNN
> - 论文创新的点
> - 算法的优化目标
> - 算法迭代过程
> - 算法在风控场景的应用

- Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, Philip S. Yu
- [\[Paper\]](https://arxiv.org/abs/2008.08692)
- [\[Code\]](https://github.com/YingtongDou/CARE-GNN)

2.  **BotSpot: A Hybrid Learning Framework to Uncover Bot Install Fraud in Mobile Advertising** *2020年*

> 综述：**有监督**算法  
> 适用场景:检测移动广告中的机器人安装欺诈行为。部属在Mobvista的线上环境中。  
> 算法效果：结果显示其在 Recall@90% Precision 评估指标上至少比其他竞争基线方法高出2.2%（第一个数据集）和5.75%（第二个数据集）
> 
> - 论文创新的点
> - 算法的优化目标
> - 算法迭代过程
> - 算法在风控场景的应用

- Tianjun Yao, Qing Li, Shangsong Liang, Yadong Zhu
- [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3340531.3412690)
- [\[Code\]](https://github.com/akakeigo2020/CIKM-Applied_Research-2150)

3.  **Spam Review Detection with Graph Convolutional Networks** *2019*

> **算法类型：**  
> 半监督学习  
> **适用场景：**  
> 垃圾评论  
> **具体算法应用效果：**  
> 在Xianyu数据集上，与TextCNN+MLP基线模型相比，F1提高了4.33分，并且在固定90%精度阈值的情况下，召回率提高了16.16%。  
> **论文创新点：** 1. 结合异构图和同构图的GCN模型，用于捕获评论的局部和全局上下文信息。2. 引入了时间相关的采样策略，以提高训练效率并减少内存消耗。3. 采用了TextCNN模型来提取评论文本的嵌入，并将这些嵌入与图神经网络模型集成，形成一个端到端的分类框架。  
> **算法输入：**  
> 算法的输入包括用户的评论、用户特征、被评论项目的特征以及评论的文本内容。  
> **算法优化的目标：**
> **算法迭代过程：**  
算法首先使用异构图GCN模型来提取用户、项目和评论的特征，然后利用同构图GCN模型来捕获评论之间的全局关系。此外，算法还包括了文本分类模型TextCNN，用于将评论文本转换为嵌入向量，并与图神经网络模型集成，形成一个端到端的分类框架。  
**算法在风控场景的应用：**

- Ao Li, Zhou Qin, Runshi Liu, Yiqun Yang, Dong Li
- [\[Paper\]](https://arxiv.org/abs/1908.10679)
- [\[Code\]](https://github.com/safe-graph/DGFraud)

4.  **Key Player Identification in Underground Forums Over Attributed Heterogeneous Information Network Embedding Framework** *2019年*
    - Yiming Zhang, Yujie Fan, Yanfang Ye, Liang Zhao, Chuan Shi

> **算法类型：** 该算法属于半监督学习，因为它结合了有标签和无标签的数据进行训练。  
> **适用场景：** 在线地下论坛挖掘，这些论坛被网络犯罪分子用来交换知识和交易非法产品或服务，对网络安全构成威胁。

> **具体算法应用效果：** 在来自不同地下论坛（例如Hack Forums和Nulled）的数据集上进行了广泛的实验，与替代方法相比，iDetective在关键人物识别方面表现出色。具体的性能提升没有在摘要中明确提及。

> **论文创新点：**
> 
> 1.  提出了一种新的地下论坛用户特征表示方法，使用属性异构信息网络（AHIN）来表示用户之间的丰富语义关系。
> 2.  提出了Player2Vec模型，有效地在AHIN中学习节点（即用户）表示，以便识别关键人物。
> 3.  开发了iDetective系统，利用Player2Vec来自动识别地下论坛中的关键人物，帮助执法机构和网络安全从业者制定有效的干预措施。

> **算法输入：**  
> 算法的输入包括地下论坛的用户配置文件、发布的主题、回复和评论等数据。

> **算法迭代过程/优化停止条件：**  
> 算法首先将构建的AHIN映射到一个由多个单视图属性图组成的多视图网络，然后使用图卷积网络（GCN）学习每个单视图属性图的嵌入，最后使用注意力机制根据不同单视图属性图学习到的不同嵌入来融合最终表示。优化过程的具体迭代次数或停止条件未在摘要中提及。

> **算法在风控场景的应用：**  
> 在风险控制场景中，iDetective系统可以自动化监视地下论坛，识别关键人物，从而帮助执法机构和网络安全从业者更好地理解网络犯罪生态系统，并制定有效的对策来打击网络犯罪。

- [Paper](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf)
- [Code](https://github.com/safe-graph/DGFraud)

5.  **Heterogeneous Graph Neural Networks for Malicious Account Detection** *2018年*  ***target***
> **算法类型：**  
> 该算法属于**半监督学习**(要具体研究一下)  
> **适用场景：**  
> 批量创建恶意账户  
> **具体算法应用效果：**  
> GEM在支付宝上部署为真实系统，每天能够检测数万个恶意账户。  
> **论文创新点：**
> 
> 1.  提出了一种新颖的基于异构图神经网络的图表示方法，捕捉攻击者的两个基本弱点（设备聚集和活动聚集）
> 2.  为了适应不同类型的设备，提出了一种注意力机制来学习不同类型设备的重要性，并使用求和操作符来建模每种类型节点的聚合模式。
> 3.  这是首次将图神经网络方法应用于欺诈检测问题，并精心构建了图。  
>     **算法输入：**  
>     算法的输入包括账户和设备的数据，以及它们之间的活动关系（例如，账户在设备上的注册或登录行为）。  
>     **算法迭代过程/优化停止条件：**  
>     **算法在风控场景的应用：**  
>     GEM系统每天针对新注册的账户运行，构建包含账户和相关设备的图，并预测在每周末新注册的账户的风险。该系统能够以高置信度检测到恶意账户，同时保持对正常用户的最小干扰

- Ziqi Liu, Chaochao Chen, Xinxing Yang, Jun Zhou, Xiaolong Li, and Le Song
- [Paper](https://arxiv.org/abs/2002.12307)
- [Code](https://github.com/safe-graph/DGFraud)

6.  **Explainable Graph-based Fraud Detection via Neural Meta-graph Search** *2022*

- Zidi Qin, Yang Liu, Qing He, Xiang Ao
- [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3511808.3557598)

7.  **Dual-Augment Graph Neural Network for Fraud Detection** *2022*

- Qiutong Li, Yanshen He, Cong Xu, Feng Wu, Jianliang Gao, Zhao Li
- [\[Paper\]](https://dl.acm.org/doi/10.1145/3511808.3557586)

8.  **BRIGHT - Graph Neural Networks in Real-time Fraud Detection** *2022*

- Mingxuan Lu, Zhichao Han, Susie Xi Rao, Zitao Zhang, Yang Zhao, Yinan Shan, Ramesh Raghunathan, Ce Zhang, Jiawei Jiang
- [Papers](https://arxiv.org/abs/2205.13084)

# AAAI：AAAI Conference on Artificial Intelligence
1.  **Cash-Out User Detection Based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism** *2019年*

>**算法类型：**
有监督学习

>**适用场景：**
识别套现用户

>**具体算法应用效果：**
数据集大小：使用了两个真实数据集，一个是包含188万用户的“十天数据集”，另一个是包含516万用户的“一个月数据集”。该算法在AUC指标上相较基线模型有1-2%的提升。

>**论文创新点：**
**算法输入：**
**算法迭代过程/优化停止条件：**
模型使用随机梯度下降（SGD）
**算法在风控场景的应用：**

- Binbin Hu, Zhiqiang Zhang, Chuan Shi, Jun Zhou, Xiaolong Li, Yuan Qi
- [\[Paper\]](https://aaai.org/ojs/index.php/AAAI/article/view/3884)
- [\[Code\]](https://github.com/safe-graph/DGFraud)

2.  **GeniePath: Graph Neural Networks with Adaptive Receptive Paths** *2019年*
**算法类型：**
半监督学习：GeniePath在有监督的设置下工作，特别是在图分类和节点分类任务中。

**适用场景：**
- 图数据表示学习：适用于各种类型的图数据，包括社交网络、引用网络、生物网络和交易网络等。
- 风控场景：例如，用于检测支付宝在线无现金支付系统中的恶意账户。

**具体算法应用效果：**
- 数据集大小：论文中使用了多个数据集，包括PubMed（约20,000个节点）、BlogCatalog（约10,000个节点）、PPI（约57,000个节点）和Alipay（约982,000个节点和超过230万条边）。
- 指标提升：在Alipay数据集上，GeniePath达到了0.826的F1分数，在PPI数据集上达到了0.979的Micro-F1分数，显示出在大规模图数据上相比于现有方法的显著性能提升。

**论文创新的点：**
- 提出了自适应路径层（adaptive path layer），包含自适应广度和深度函数，用于分别学习不同大小的邻域的重要性以及从不同跳数的邻居中提取和过滤信号。
- GeniePath能够在归纳（inductive）和演绎（transductive）设置下工作，且相比于其他竞争方法，在归纳设置下尤其有效。

**算法输入：**
- 输入包括图的邻接矩阵、节点特征矩阵以及节点标签（对于有监督任务）。

**算法迭代过程/优化停止条件：**
- 算法通过迭代过程来优化自适应路径层中的参数，使用Adam优化器进行梯度下降。
- 优化停止条件是当模型在验证集上的性能不再提升时。

**算法在风控场景的应用：**
- 在风控场景中，GeniePath被用于构建账户-设备网络，以识别恶意账户。通过学习账户和设备之间的交互模式，GeniePath能够有效地识别出欺诈行为。
- Ziqi Liu, Chaochao Chen, Longfei Li, Jun Zhou, Xiaolong Li, Le Song, Yuan Qi
- [\[Paper\]](https://arxiv.org/abs/1802.00910)
- [\[Code\]](https://github.com/safe-graph/DGFraud)

3.  **One-Class Adversarial Nets for Fraud Detection** *2019年* ***target***

**算法类型：**
- 无监督学习：OCAN是一种无监督学习方法，因为它不需要恶意用户的标记数据。

**适用场景：**
- 在线社交平台和知识库：例如维基百科（Wikipedia）的恶意用户检测，以及电子商务网站上的欺诈性评论检测。

**具体算法应用效果：**
- 数据集大小：论文中使用了UMDWikipedia数据集，包含约770K次编辑，以及一个信用卡交易数据集，包含284,807笔交易。
- 指标提升：在Wikipedia数据集上，OCAN在F1分数和准确率方面优于现有的单类分类模型，并与需要良性和恶意用户数据的最新多源LSTM模型（M-LSTM）表现相当。

**论文创新的点：**
- 提出了一种新型的生成对抗网络（GAN），即互补GAN（complementary GAN），用于生成与良性用户表示互补的样本，即潜在的恶意用户。
- OCAN不需要恶意用户的任何信息，因此不需要手动创建混合训练数据集，更适用于不同类型的恶意用户识别任务。
- OCAN能够捕捉用户活动的序列信息，并且能够动态更新用户表示，从而实时预测用户是否为欺诈者。

**算法输入：**
- 输入包括用户的在线活动序列，例如编辑历史或交易记录。

**算法迭代过程/优化停止条件：**
- OCAN包含两个阶段：首先使用LSTM-Autoencoder学习良性用户的表示，然后训练互补GAN模型，其中生成器尝试生成与良性用户表示互补的样本，判别器则尝试区分良性用户和生成的样本。
- 优化停止条件是当模型在验证集上的性能不再提升时。

**算法在风控场景的应用：**
- OCAN可以应用于在线社交平台和知识库的欺诈检测，以及信用卡交易欺诈检测等风险控制场景。

总的来说，OCAN是一种创新的无监督学习方法，特别适用于只有良性用户数据可用的欺诈检测场景。通过结合LSTM-Autoencoder和互补GAN，OCAN能够有效地识别出潜在的恶意用户，即使在没有恶意用户标记数据的情况下也能达到令人满意的性能。
- Panpan Zheng, Shuhan Yuan, Xintao Wu, Jun Li, Aidong Lu
- [\[Paper\]](https://arxiv.org/abs/1803.01798)
- [\[Code\]](https://github.com/ILoveAI2019/OCAN)

4.  **SAFE: A Neural Survival Analysis Model for Fraud Early Detection** *2019年*
这篇论文提出了一种名为SAFE（Neural Survival Analysis Model for Fraud Early Detection）的新型算法，用于在线平台的欺诈行为早期检测。以下是根据您提出的几个方面对该论文的总结：

**算法类型：**
- 有监督学习：SAFE是一个有监督的学习模型，它使用标记的训练数据来预测用户是否会进行欺诈活动。

**适用场景：**
- 在线社交平台和知识库：例如Twitter和维基百科（Wikipedia），用于检测和预防欺诈活动，如恶意用户传播虚假信息或有害链接。

**具体算法应用效果：**
- 数据集大小：论文中使用了两个真实世界的数据集，一个是Twitter数据集，包含51,608个用户；另一个是UMDWikipedia数据集，包含1,759个用户。
- 指标提升：在Twitter数据集上，与支持向量机（SVM）、Cox比例风险模型（CPH）和多源长短期记忆网络（M-LSTM）相比，SAFE在精确度（Precision）、召回率（Recall）、F1分数（F1）和准确率（Accuracy）上都有显著提升。例如，在Twitter数据集上，SAFE的F1分数达到了0.6537，而M-LSTM的F1分数为0.4400。

**论文创新的点：**
- SAFE是首个将生存分析应用于欺诈检测的模型，它通过递归神经网络（RNN）处理用户活动序列，并直接在每个时间戳输出危险值，然后使用危险值派生的生存概率进行一致性预测。
- SAFE不假设任何特定的生存时间分布，而是通过RNN学习用户活动随时间变化的危险率，并适应性地调整损失函数以实现欺诈行为的早期检测。

**算法输入：**
- 输入包括用户活动序列，如Twitter用户的粉丝数、关注数、推文数、点赞数和公开列表成员数，以及维基百科用户的编辑行为特征。

**算法迭代过程/优化停止条件：**
- SAFE通过反向传播算法进行训练，使用Adam优化器，批量大小为16，学习率为10^-3。模型根据损失函数进行迭代优化，直到在验证集上的性能不再提升。

**算法在风控场景的应用：**
- SAFE可以部署在需要实时欺诈检测的在线平台上，如社交媒体和电子商务网站，以便在用户被平台封禁之前尽早识别出潜在的欺诈行为。

总的来说，SAFE是一种创新的欺诈早期检测模型，它结合了生存分析和递归神经网络的优势，能够在不假设特定生存时间分布的情况下，提供一致且准确的欺诈检测预测。通过适应性地调整损失函数，SAFE能够有效地实现欺诈行为的早期检测，这对于及时防止欺诈活动和保护在线平台及其合法用户具有重要意义。
- Panpan Zheng, Shuhan Yuan, Xintao Wu
- [\[Paper\]](https://arxiv.org/abs/1809.04683v2)
- [\[Code\]](https://github.com/PanpanZheng/SAFE)

5.  **Modeling the Field Value Variations and Field Interactions Simultaneously for Fraud Detection** 2021年

- Dongbo Xi, Bowen Song, Fuzhen Zhuang, Yongchun Zhu, Shuai Chen, Tianyi Zhang, Yuan Qi, Qing He
- [\[Paper\]](https://arxiv.org/abs/2008.05600)

# ICDM：IEEE International Conference on Data Mining

1.  **A Semi-Supervised Graph Attentive Network for Fraud Detection** *2019年*
这篇论文提出了一种半监督图注意力网络（SemiGNN）用于金融欺诈检测。以下是根据您提供的方面对论文的总结：

**算法类型：**
- **半监督学习**：SemiGNN结合了有标签和无标签数据进行训练，利用了社交关系来扩展标签数据，并通过图神经网络来处理多视图数据。

**适用场景：**
- 金融欺诈检测，特别是在金融服务领域，如支付宝这样的大型第三方在线支付平台。

**具体算法应用效果：**
- 数据集大小：实验在包含约400万用户的Alipay数据集上进行，扩展到超过1亿用户。
- 指标提升：SemiGNN在用户违约预测和用户属性预测两个任务上均取得了比现有方法更好的结果。例如，在用户违约预测任务上，SemiGNN的AUC和KS指标分别为0.807和0.464，相比其他方法有显著提升。

**论文创新点：**
- 提出了一种新颖的半监督图嵌入模型，具有层次化注意力机制，用于模拟多视图图网络进行欺诈检测。
- 引入了节点级注意力和视图级注意力机制，以更好地关联不同邻居和不同视图，同时提供了模型的可解释性。
- 利用社交关系和用户属性，通过半监督学习充分利用了有限的标签数据和大量的无标签数据。

**算法输入：**
- 多视图数据，包括用户社交关系图、用户-应用图、用户昵称图和用户地址图。
- 有标签的用户数据（少量）和通过社交关系扩展的无标签用户数据（大量）。

**算法迭代过程/优化停止条件：**
- 使用随机梯度下降（SGD）进行模型优化，学习率为0.002，衰减率为0.95，批量大小为128，模型训练3个周期。
- 迭代过程中，通过随机游走生成用户配对集，使用softmax函数进行分类，并结合无监督的图基损失函数进行优化。
- 优化停止条件没有明确提及，但通常是基于达到一定的性能指标或经过预设的迭代次数。

**算法在风控场景的应用：**
- 用于预测用户是否会涉及欺诈行为，如违约预测和用户属性预测，这些对于金融服务提供商来说至关重要，可以帮助他们进行风险控制和信用评估。
- 通过分析用户间的社交关系和用户属性，模型能够识别出潜在的欺诈行为，从而保护用户和服务提供商的安全。

总的来说，SemiGNN通过结合半监督学习方法和图神经网络，有效地提高了金融欺诈检测的准确性，并且由于其可解释性，使得模型的预测结果更加可靠和易于理解。
- Daixin Wang, Jianbin Lin, Peng Cui, Quanhui Jia, Zhen Wang, Yanming Fang, Quan Yu, Jun Zhou, Shuang Yang, and Qi Yuan
- [\[Paper\]](https://arxiv.org/abs/2003.01171)
- [\[Code\]](https://github.com/safe-graph/DGFraud)

2.  **Spotting Suspicious Link Behavior with fBox: An Adversarial Perspective** *2014年*  *target**
**算法类型：**
- **无监督学习**：fBox算法是一种无监督的方法，它不需要任何标签数据来识别可疑的链接行为。

**适用场景：**
- 在线社交网络和网络服务，如Twitter和Amazon，用于检测和防范链接欺诈行为，例如购买粉丝、页面点赞等。

**具体算法应用效果：**
- 数据集大小：实验在包含4170万用户和15亿边的Twitter社交图上进行。
- 指标提升：论文没有明确提供具体的性能提升指标，但指出fBox能够以高精确度识别许多至今仍未被暂停的可疑账户。

**论文创新的点：**
- 提出了一种从对抗性角度出发的理论分析，证明了现代光谱方法的检测范围的局限性。
- 引入了fBox算法，该算法能够检测到那些逃避光谱方法的小规模、隐蔽的攻击。
- 证明了fBox在真实数据上的高效性，并且算法具有可扩展性（输入大小的线性关系）。

**算法输入：**
- 输入图的邻接矩阵，代表用户和对象（如Twitter中的用户和页面）之间的关系。

**算法迭代过程/优化停止条件：**
- fBox算法通过计算输入图的奇异值分解（SVD），并根据重建度量来识别可疑节点。
- 算法的迭代过程涉及对每个节点的重建度进行评估，并将其与特定阈值进行比较以确定其可疑性。
- 优化停止条件是当所有节点都被评估并分类为可疑或非可疑后。

**算法在风控场景的应用：**
- fBox可以应用于任何需要检测和防范链接欺诈的在线平台，特别是在金融风控领域，帮助识别和阻止欺诈行为，保护用户和平台的信誉。
- Neil Shah, Alex Beutel, Brian Gallagher, Christos Faloutsos
- [\[Paper\]](https://arxiv.org/pdf/1410.3915.pdf)
- [\[Code\]](https://github.com/safe-graph/UGFraud)

3.  **GANG: Detecting Fraudulent Users in Online Social Networks via Guilt-by-Association on Directed Graphs** *2017*
这篇论文提出了一种名为GANG的算法，用于在在线社交网络（OSNs）中检测欺诈用户。以下是根据您提供的方面对论文的总结：

**算法类型：**
- **半监督学习**：GANG算法结合了有标签的欺诈用户和正常用户信息进行训练，以预测未标记用户的标签。

**适用场景：**
- 在线社交网络，如Twitter和Sina Weibo，用于检测和识别欺诈用户，包括垃圾邮件发送者、假账户和被控制的正常用户。

**具体算法应用效果：**
- 数据集大小：实验在包含4165万用户和14.68亿条边的Twitter数据集，以及包含353万用户和6.53亿条边的Sina Weibo数据集上进行。
- 指标提升：GANG在Twitter数据集上的AUC为0.72，在Sina Weibo数据集上的AUC为0.80，显著优于现有方法。

**论文创新的点：**
- 提出了一种基于有向图的新型配对马尔可夫随机场（pMRF）模型，用于捕捉欺诈用户检测问题的独特特征。
- 设计了一种基于Loopy Belief Propagation (LBP)的推断方法，并对其进行了优化，以提高算法的可扩展性和收敛性。
- 提出了一种矩阵形式的优化GANG算法，可以更高效地计算节点的后验概率分布。

**算法输入：**
- 有向社交图的邻接矩阵，包括用户之间的有向边。
- 训练数据集中有标签的欺诈用户和正常用户的集合。

**算法迭代过程/优化停止条件：**
- 基本版本的GANG使用LBP进行迭代消息传递，直到消息变化可以忽略不计或达到预设的最大迭代次数。
- 优化版本的GANG通过消除消息维护和近似为矩阵形式来简化迭代过程，提高了算法的效率和可扩展性。

**算法在风控场景的应用：**
- 在线社交网络和网络安全领域，GANG可以用于识别和打击欺诈行为，保护用户免受恶意活动的侵害，维护社交网络的真实性和安全性。

总的来说，GANG算法通过利用有向图的结构特性和半监督学习方法，有效地提高了在线社交网络中欺诈用户检测的准确性，并且通过优化，使得算法更加高效和可扩展，适用于大规模社交网络数据的分析。
- Binghui Wang, Neil Zhenqiang Gong, Hao Fu
- [\[Paper\]](https://home.engineering.iastate.edu/~neilgong/papers/GANG.pdf)
- [\[Code\]](https://github.com/safe-graph/UGFraud)

# WSDM：ACM International Conference on Web Search and Data Mining
1.  **REV2: Fraudulent User Prediction in Rating Platforms** *2018*
- Srijan Kumar, Bryan Hooi, Disha Makhija, Mohit Kumar, Christos Faloutsos, V. S. Subrahmanian
这篇论文提出了一个名为Rev2的系统，用于在评分平台上识别欺诈用户。以下是根据您提供的方面对论文的总结：

**算法类型：**
- **半监督学习**：Rev2算法结合了网络和行为属性，同时适用于无监督和监督设置，用于检测欺诈用户。

**适用场景：**
- 在线评分平台，如电子商务网站、比特币交易网络等，用于识别给出虚假评分的欺诈用户。

**具体算法应用效果：**
- 数据集大小：实验在五个评分数据集上进行，包括Flipkart（印度最大的在线市场）、Bitcoin OTC、Alpha、Amazon和Epinions。
- 指标提升：在无监督设置中，Rev2在8个中的10个情况下平均精度最高或次之；在监督设置中，Rev2在所有数据集上的平均AUC值≥0.85，表现出色。

**论文创新的点：**
- 提出了三个相互依赖的内在质量指标：用户公平性（fairness）、评分可靠性（reliability）和产品优良性（goodness）。
- 提出了一个迭代算法Rev2，结合网络属性、冷启动处理和行为特征来计算这些指标。
- 证明了Rev2算法在有限次数的迭代内保证收敛，并且具有线性时间复杂度。

**算法输入：**
- 用户-项目评分网络，包括用户、评分和产品的集合。
- 可选的行为属性数据，如用户评分时间间隔和评分文本。

**算法迭代过程/优化停止条件：**
- 算法通过迭代更新用户的公平性、产品的优良性和评分的可靠性分数。
- 优化停止条件是当所有分数的变化小于一个设定的阈值时，认为算法已经收敛。

**算法在风控场景的应用：**
- Rev2算法在Flipkart的实际部署中，成功识别出了150个最不公平用户中的127个欺诈用户（准确率为84.6%）。

总的来说，Rev2算法通过结合网络结构和用户行为特征，有效地提高了欺诈用户检测的准确性，并且在实际应用中得到了验证。算法的设计具有理论上的保证，并且在实际数据集上表现出了优越的性能。
- [\[Paper\]](https://cs.stanford.edu/~srijan/pubs/rev2-wsdm18.pdf)
- [\[Code\]](https://cs.stanford.edu/~srijan/rev2/)

2.  **A Framework for Detecting Frauds from Extremely Few Labels** *2023*
- Ya-Lin Zhang, Yi-Xuan Sun, Fangfang Fan, Meng Li, Yeyu Zhao, Wei Wang, Longfei Li, Jun Zhou, Jinghua Feng
- [Papers](https://dl.acm.org/doi/10.1145/3539597.3573022)

# VLDB：International Conference on Very Large Data Bases
1.  ZooBP: Belief Propagation for Heterogeneous Networks (VLDB 2017)
- Dhivya Eswaran, Stephan Gunnemann, Christos Faloutsos, Disha Makhija, Mohit Kumar
根据文档内容，对论文进行总结如下：
1. **算法分类**：ZooBP算法属于半监督学习算法，利用少量标注节点进行图分类任务。
2. **适用场景**：适用于任何无向加权异构图，具有多种节点和边类型，如社交网络、电商网络等。
3. **算法效果**：在Flipkart电商网络数据集上，针对3种节点类型和5种边类型，ZooBP算法能够准确识别出92.3%的前300名欺诈用户，表现优异。
4. **创新点**：
    - 提出了通用的异构图Belief Propagation算法ZooBP，适用于任意无向加权异构图。
    - 给出了ZooBP的闭式解，并提供了收敛性证明。
    - 实现了线性时间复杂度，在图规模上具有线性扩展性，比BP算法快600倍。
5. **算法输入**：异构图的节点和边类型、节点的类别数、节点间的兼容性矩阵、部分节点的初始类别标签。
6. **迭代过程**：基于残差兼容性矩阵和残差信念，推导出节点信念的线性方程组，并利用矩阵运算一次性求解所有节点的最终信念。
7. **优化停止条件**：迭代更新收敛，满足谱半径条件||P-Q|| < 1。
8. **风控应用**：可用于电子商务平台的风控场景，如识别欺诈用户、商品和卖家。
- [\[Paper\]](http://www.vldb.org/pvldb/vol10/p625-eswaran.pdf)
- [\[Code\]](https://github.com/safe-graph/UGFraud)

# IJCAI：International Joint Conference on Artificial Intelligence

1.  Online Credit Payment Fraud Detection via Structure-Aware Hierarchical Recurrent Neural Network (IJCAI 2021)

- Wangli Lin, Li Sun, Qiwei Zhong, Can Liu, Jinghua Feng, Xiang Ao, Hao Yang
- [Paper](https://www.ijcai.org/proceedings/2021/505)


# NeurIPS：Conference on Neural Information Processing Systems

暂时没有合适的论文，欢迎交流。

# ICML：International Conference on Machine Learning

暂时没有合适的论文，欢迎交流。

# 参考网站

[Awesome Fraud Detection Research Papers.](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers "全网比较权威和完整的反欺诈文章整理")  
[Fraud Detection Papers](https://github.com/IPL/fraud-detection-papers "广告反欺诈内容比较多，看着停更有几年了")  
[Best Paper Awards](https://jeffhuang.com/best_paper_awards/institutions.html "Update了各类顶会文章的Best Papers")  
[spartan2](https://github.com/BGT-M/spartan2 "spartan2 is a collection of data mining algorithms on big graphs and time series, providing three basic tasks: anomaly detection, forecast, and summarization.")