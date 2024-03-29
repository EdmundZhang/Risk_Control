> 持续整理和解读风控反作弊方面的顶会论文，可以关注、收藏。
> 评选标准：
> 1. 有代码、方便复现的论文优先考虑
> 2. 对账号、营销薅羊毛、支付风控、广告反作弊等场景问题有直接启发的论文优先考虑

| 名称         | 方向                 | 论文数 |有代码论文数
| ----------- | ----------------------- | --- |--|
| WWW       |       -                        | 16  |7|
| KDD         | Data Mining             | 8   |3|
| CIKM        | Data Mining            | 8   |5|
| AAAI        | Artificial Intelligence | 5   |4|
| ICDM       |Data Mining               | 3   |3|
| IJCAI        | Artificial Intelligence | 1   |
| WSDM     | -                                | 1   |
| VLDB        | -                                | 1   |
| WSDM      |-                                 | 1   |
| NeurIPS     | Machine Learning        | -   |
| ICML         | Machine Learning        | -   |
  

- 已解读论文  
[从虚假点赞到恶意评论：FRAUDAR算法如何一路斩妖除魔？](https://zhuanlan.zhihu.com/p/687094360 '')  
SliceNDice - 2024年3月    
FlowScope - 2024年4月   


# WWW
1. **SliceNDice: Mining Suspicious Multi-attribute Entity Groups with Multi-view Graphs**
> **算法原理**：捕捉具备大量共同属性的用户，比如在同一时间和地点创建的帐户、宣传相同言论和转发类似文章的账户。(IEEE DSAA (2019)) 
> **输入**：实体+属性1...+属性N
> **算法功能**：根据共同属性构造边，合并各属性构造的子图，形成Multi-View Graph.随后挖掘异常关联子图（"dense" subnetwork）.
> **实验数据集大小**：Snapchat advertiser platform（23W实体 × 12属性），发现2435异常实体，precision 89%
- [Paper](https://arxiv.org/abs/1908.07087 '') 
- [Code](https://github.com/hamedn/SliceNDice '')

2. Spam Review Detection with Graph Convolutional Networks
> **算法原理**：这篇论文中，作者构建了两种图结构：异构图Xianyu Graph和同构图Comment Graph。
> **输入**:异构图 Xianyu Graph：
> **节点**由用户（U）、物品（I）和评论（E）组成。
> 同构图 Comment Graph：节点由评论（E）组成。
> **边**表示相似的评论。通过构建近似KNN图算法，找到每个评论的K个最近邻居，从而形成同构图。
> **算法功能**：
使用异构GCN对Xianyu Graph进行embedding
采用归纳GCN对Comment Graph进行embedding
采用TextCNN对评论文本进行embedding
一起输入到GCN-based Anti-Spam（GAS）模型，进行有监督分类。
- [Paper](https://arxiv.org/abs/1908.10679 '阿里的论文 CIKM 2019')   
- [Code](https://github.com/safe-graph/DGFraud '')

3. Heterogeneous Graph Neural Networks for Malicious Account Detection
> 算法功能：考虑设备聚合和活动聚合，构建图结构embedding进行分类。本质是图卷积网络的一种变体
> 效果：GBDT+Graph 方法与 GBDT+Node2Vec 方法相比效果相似，而 GCN 的效果优于 GBDT+Graph 和 GBDT+Node2Vec。
> 作者提出的 GEM 方法在多种方面都优于 GCN，因为它处理了异构设备类型，而 GCN 只能处理同质图，无法区分图中不同类型的节点。
> GEM 还使用每种节点类型的聚合操作，而不是标准化操作，因此更好地模拟了底层的聚合模式。  
- [Paper]('https://dl.acm.org/doi/10.1145/3269206.3272010' '') 
- [Code](https://github.com/safe-graph/DGFraud '')


4. FlowScope: Spotting Money Laundering Based on Graphs
> FlowScope(AAAI 2020年)
> 输入：A（转账人）->M(中间人)->C(收款人)
> 算法功能：找到其中可能涉及洗钱的团伙
> 实验数据集大小：-
- [Paper](https://shenghua-liu.github.io/papers/aaai2020cr-flowscope.pdf '')   
- [Code](https://github.com/BGT-M/spartan2 '')

5. FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System (WWW 2019)
  - Rui Wen, Jianyu Wang and Yu Huang
  - [[Paper]](https://dl.acm.org/citation.cfm?id=3316586 '')
  - [[Code]](https://github.com/safe-graph/DGFraud '')
  
6. CatchCore: Catching Hierarchical Dense Subtensor
- [Paper](https://shenghua-liu.github.io/papers/pkdd2019-catchcore.pdf '') 
- [Code](https://github.com/wenchieh/catchcore/tree/master '')

7. No Place to Hide: Catching Fraudulent Entities in Tensors
- [Paper](https://arxiv.org/pdf/1810.06230.pdf '')
- [Code](https://proceedings.mlr.press/v71/ram18a/ram18a.pdf '')

8. HiDDen: Hierarchical Dense Subgraph Detection with Application to Financial Fraud Detection
- [Paper](https://www.public.asu.edu/~hdavulcu/SDM17.pdf '')

9. FairPlay: Fraud and Malware Detection in Google Play
- [Paper](https://arxiv.org/abs/1703.02002 '')

10. Fraud Transactions Detection via Behavior Tree with Local Intention Calibration
- [Paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403354 '')

11. Identifying Anomalies in Graph Streams Using Change Detection
- [Paper](https://www.mlgworkshop.org/2016/paper/MLG2016_paper_12.pdf '')

12. HitFraud: A Broad Learning Approach for Collective Fraud Detection in Heterogeneous Information Networks
- [Paper](https://arxiv.org/abs/1709.04129 '')

13. Friend or Faux: Graph-Based Early Detection of Fake Accounts on Social Networks
- [Paper](https://arxiv.org/abs/2004.04834 '') 

14. Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection (WWW 2021)
  - Yang Liu, Xiang Ao, Zidi Qin, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He
  - [[Paper]](https://dl.acm.org/doi/abs/10.1145/3442381.3449989 '')

  15. Financial Defaulter Detection on Online Credit Payment via Multi-view Attributed Heterogeneous Information Network (WWW 2020)
  - Qiwei Zhong, Yang Liu, Xiang Ao, Binbin Hu, Jinghua Feng, Jiayu Tang, Qing He
  - [[Paper]](https://dl.acm.org/doi/abs/10.1145/3366423.3380159 '')

  16. Friend or Faux: Graph-Based Early Detection of Fake Accounts on Social Networks (WWW 2020)
  - Adam Breuer, Roee Eilat, Udi Weinsberg
  - [[Paper]](https://arxiv.org/abs/2004.04834 '')

# KDD
1. [**Best Paper**]2026-FRAUDAR: Bounding Graph Fraud in the Face of Camouflage
 - 论文解读: [从虚假点赞到恶意评论：FRAUDAR算法如何一路斩妖除魔？](https://zhuanlan.zhihu.com/p/687094360 '')
  - Bryan Hooi-Carnegie Mellon University 
   - [Papers](https://www.semanticscholar.org/paper/FRAUDAR%3A-Bounding-Graph-Fraud-in-the-Face-of-Hooi-Song/2852982175beeb92e14127277cb158c4cb31f5c5 '')
   - [Code](https://github.com/safe-graph '')

2. [**Best Paper**]2018-Adversarial Attacks on Neural Networks for Graph Data
   - Daniel Zügner-Technical University of Munich;
   - [Papers](https://www.semanticscholar.org/paper/Adversarial-Attacks-on-Neural-Networks-for-Graph-Z%C3%BCgner-Akbarnejad/6c44f8e62d824bcda4f291c679a5518bbd4225f6 '')
   - [Codes](https://github.com/danielzuegner/nettack '')
3. Collective Opinion Spam Detection: Bridging Review Networks and Metadata (KDD 2015)
  - Shebuti Rayana, Leman Akoglu
  - [[Paper]](https://www.andrew.cmu.edu/user/lakoglu/pubs/15-kdd-collectiveopinionspam.pdf '')
  - [[Code]](https://github.com/safe-graph/UGFraud '')
  
4. [**Best Paper**]2012-Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping
   - Thanawin Rakthanmanon, University of California, Riverside;
   - [Papers](https://www.semanticscholar.org/paper/Searching-and-Mining-Trillions-of-Time-Series-under-Rakthanmanon-Campana/c5a8d4145dc3530d437c93112b790bddb1927b59 '')
5. [**Best Paper**]2004-A probabilistic framework for semi-supervised clustering
   - Sugato Basu University of Texas at Austin;
   - [Papers](https://www.semanticscholar.org/paper/A-probabilistic-framework-for-semi-supervised-Basu-Bilenko/01d5bf24c0b35d9a234d534bf69924fa16201dee '')
6. [**Best Paper**]2002-Pattern discovery in sequences under a Markov assumption
   - Darya Chudova & Padhraic Smyth, University of California, Irvine
   - [Papers](https://www.semanticscholar.org/paper/Pattern-discovery-in-sequences-under-a-Markov-Chudova-Smyth/cf640511fb17f544b99a80470b018b7ea0d7b7b4 '')

7. Intention-aware Heterogeneous Graph Attention Networks for Fraud Transactions Detection (KDD 2021)
  - Can Liu, Li Sun, Xiang Ao, Jinghua Feng, Qing He, Hao Yang
  - [[Paper]](https://dl.acm.org/doi/10.1145/3447548.3467142 '')

8. Live-Streaming Fraud Detection: A Heterogeneous Graph Neural Network Approach (KDD 2021)
  - Haishuai Wang, Zhao Li, Peng Zhang, Jiaming Huang, Pengrui Hui, Jian Liao, Ji Zhang, Jiajun Bu
  - [[Paper]](https://dl.acm.org/doi/abs/10.1145/3447548.3467065 '')


# CIKM
1. Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters (CIKM 2020)
  - Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, Philip S. Yu
  - [[Paper]](https://arxiv.org/abs/2008.08692)
  - [[Code]](https://github.com/YingtongDou/CARE-GNN '') 

2. BotSpot: A Hybrid Learning Framework to Uncover Bot Install Fraud in Mobile Advertising (CIKM 2020)
  - Tianjun Yao, Qing Li, Shangsong Liang, Yadong Zhu
  - [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3340531.3412690 '')
  - [[Code]](https://github.com/akakeigo2020/CIKM-Applied_Research-2150 '')

3. Spam Review Detection with Graph Convolutional Networks (CIKM 2019)
  - Ao Li, Zhou Qin, Runshi Liu, Yiqun Yang, Dong Li
  - [[Paper]](https://arxiv.org/abs/1908.10679 '')
  - [[Code]](https://github.com/safe-graph/DGFraud '')

4. Key Player Identification in Underground Forums Over Attributed Heterogeneous Information Network Embedding Framework (CIKM 2019)
   - Yiming Zhang, Yujie Fan, Yanfang Ye, Liang Zhao, Chuan Shi
   - [[Paper]](http://mason.gmu.edu/~lzhao9/materials/papers/lp0110-zhangA.pdf '')
   - [[Code]](https://github.com/safe-graph/DGFraud '')

5. Heterogeneous Graph Neural Networks for Malicious Account Detection (CIKM 2018)
  - Ziqi Liu, Chaochao Chen, Xinxing Yang, Jun Zhou, Xiaolong Li, and Le Song
  - [[Paper]](https://dl.acm.org/doi/10.1145/3269206.3272010 '')
  - [[Code]](https://github.com/safe-graph/DGFraud '')

  6. Explainable Graph-based Fraud Detection via Neural Meta-graph Search (CIKM 2022)
  - Zidi Qin, Yang Liu, Qing He, Xiang Ao
  - [[Paper]](https://dl.acm.org/doi/abs/10.1145/3511808.3557598 '')

7. Dual-Augment Graph Neural Network for Fraud Detection (CIKM 2022)
  - Qiutong Li, Yanshen He, Cong Xu, Feng Wu, Jianliang Gao, Zhao Li
  - [[Paper]](https://dl.acm.org/doi/10.1145/3511808.3557586 '')

  8. 2022-BRIGHT - Graph Neural Networks in Real-time Fraud Detection-Mingxuan Lu, Zhichao Han, Susie Xi Rao, Zitao Zhang, Yang Zhao, Yinan Shan, Ramesh Raghunathan, Ce Zhang, Jiawei Jiang
- [Papers](https://arxiv.org/abs/2205.13084 '')


# AAAI
1. Cash-Out User Detection Based on Attributed Heterogeneous Information Network with a Hierarchical Attention Mechanism (AAAI 2019)
  - Binbin Hu, Zhiqiang Zhang, Chuan Shi, Jun Zhou, Xiaolong Li, Yuan Qi
  - [[Paper]](https://aaai.org/ojs/index.php/AAAI/article/view/3884 '')
  - [[Code]](https://github.com/safe-graph/DGFraud '')
  
2. GeniePath: Graph Neural Networks with Adaptive Receptive Paths (AAAI 2019)
  - Ziqi Liu, Chaochao Chen, Longfei Li, Jun Zhou, Xiaolong Li, Le Song, Yuan Qi
  - [[Paper]](https://arxiv.org/abs/1802.00910 '')
  - [[Code]](https://github.com/safe-graph/DGFraud '')

3. One-Class Adversarial Nets for Fraud Detection (AAAI 2019)
  - Panpan Zheng, Shuhan Yuan, Xintao Wu, Jun Li, Aidong Lu
  - [[Paper]](https://arxiv.org/abs/1803.01798 '')
  - [[Code]](https://github.com/ILoveAI2019/OCAN '')

4. SAFE: A Neural Survival Analysis Model for Fraud Early Detection (AAAI 2019)
  - Panpan Zheng, Shuhan Yuan, Xintao Wu
  - [[Paper]](https://arxiv.org/abs/1809.04683v2 '')
  - [[Code]](https://github.com/PanpanZheng/SAFE '')

  5. Modeling the Field Value Variations and Field Interactions Simultaneously for Fraud Detection (AAAI 2021)
  - Dongbo Xi, Bowen Song, Fuzhen Zhuang, Yongchun Zhu, Shuai Chen, Tianyi Zhang, Yuan Qi, Qing He
  - [[Paper]](https://arxiv.org/abs/2008.05600 '')

# ICDM
1. A Semi-Supervised Graph Attentive Network for Fraud Detection (ICDM 2019)
  - Daixin Wang, Jianbin Lin, Peng Cui, Quanhui Jia, Zhen Wang, Yanming Fang, Quan Yu, Jun Zhou, Shuang Yang, and Qi Yuan 
  - [[Paper]](https://arxiv.org/abs/2003.01171 '')
  - [[Code]](https://github.com/safe-graph/DGFraud '')


2. Spotting Suspicious Link Behavior with fBox: An Adversarial Perspective (ICDM 2014)
  - Neil Shah, Alex Beutel, Brian Gallagher, Christos Faloutsos
  - [[Paper]](https://arxiv.org/pdf/1410.3915.pdf '')
  - [[Code]](https://github.com/safe-graph/UGFraud '')


3. GANG: Detecting Fraudulent Users in Online Social Networks via Guilt-by-Association on Directed Graphs (ICDM 2017)
  - Binghui Wang, Neil Zhenqiang Gong, Hao Fu
  - [[Paper]](https://ieeexplore.ieee.org/document/8215519 '')
  - [[Code]](https://github.com/safe-graph/UGFraud '')

# IJCAI  
1. Online Credit Payment Fraud Detection via Structure-Aware Hierarchical Recurrent Neural Network (IJCAI 2021)
  - Wangli Lin, Li Sun, Qiwei Zhong, Can Liu, Jinghua Feng, Xiang Ao, Hao Yang
  - [[Paper]](https://www.ijcai.org/proceedings/2021/505 '')

# WSDM 
1. 2023-A Framework for Detecting Frauds from Extremely Few Labels-Ya-Lin Zhang, Yi-Xuan Sun, Fangfang Fan, Meng Li, Yeyu Zhao, Wei Wang, Longfei Li, Jun Zhou, Jinghua Feng
- [Papers](https://dl.acm.org/doi/10.1145/3539597.3573022 '')



# VLDB
1. ZooBP: Belief Propagation for Heterogeneous Networks (VLDB 2017)
  - Dhivya Eswaran, Stephan Gunnemann, Christos Faloutsos, Disha Makhija, Mohit Kumar
  - [[Paper]](http://www.vldb.org/pvldb/vol10/p625-eswaran.pdf '')
  - [[Code]](https://github.com/safe-graph/UGFraud '')

# WSDM
 1. REV2: Fraudulent User Prediction in Rating Platforms (WSDM 2018)
  - Srijan Kumar, Bryan Hooi, Disha Makhija, Mohit Kumar, Christos Faloutsos, V. S. Subrahmanian
  - [[Paper]](https://cs.stanford.edu/~srijan/pubs/rev2-wsdm18.pdf '')
  - [[Code]](https://cs.stanford.edu/~srijan/rev2/ '')

# NeurIPS 
暂时没有合适的论文，欢迎交流。
# ICML  
暂时没有合适的论文，欢迎交流。

# 参考网站
[Awesome Fraud Detection Research Papers.](https://github.com/benedekrozemberczki/awesome-fraud-detection-papers '全网比较权威和完整的反欺诈文章整理')
[Fraud Detection Papers](https://github.com/IPL/fraud-detection-papers '广告反欺诈内容比较多，看着停更有几年了')  
[Best Paper Awards](https://jeffhuang.com/best_paper_awards/institutions.html 'Update了各类顶会文章的Best Papers')  
[spartan2](https://github.com/BGT-M/spartan2 'spartan2 is a collection of data mining algorithms on big graphs and time series, providing three basic tasks: anomaly detection, forecast, and summarization.')  
