> 人人看得懂的顶会论文系列之：SHAP
> (NeurIPS:Conference and Workshop on Neural Information Processing Systems) 2017，引用量5940
> 论文标题：A Unified Approach to Interpreting Model Predictions
> Author：华盛顿大学计算机科学学院的Scott M. Lundberg和Su-In Lee。

# 1.博弈论(Game Theory)
博弈论主要研究，参与者在竞争性环境中怎样决策。基本思想是：每个参与者都在寻求最大化自己的利益，他们的选择会受到其他参与者选择的影响。最有名的案例应该就是**囚徒困境（Prisoner’s Dilemma）**、**联盟博弈（Coalitional Game）**。

联盟博弈（Coalitional Game）是一种合作博弈，其中多个参与者通过形成联盟来共同追求目标。在这个游戏中，参与者需要决定是否加入联盟，以及如果加入，如何**分配获得的收益**。

## Shapley值
Shapley值（Shapley Value）是一种在合作博弈中分配收益的方法，它由数学家刘易斯·沙普利（Lloyd S. Shapley）提出，用于解决如何**公平地分配合作博弈中获得的收益问题**。

Shapley值的基本思想是，每个参与者对博弈的贡献应该得到公平的回报。它通过计算每个参与者加入不同联盟的概率，以及他们在该联盟中的贡献，来计算每个参与者的Shapley值。Shapley值确保了每个参与者对博弈的贡献都能得到公平的回报，这是一种公平的合作博弈收益分配方法。

**一个计算Shapley值的案例如下**：
假设有员工A、B、C，参加一个团队项目，完成后公司将提供一笔奖金。
员工A单独参加可以赢得1万元，员工B单独参加可以赢得1.5万元，员工C单独参加可以赢得2万元。
各种组合的情况下，员工预期的收益如下：
![82504ec09818d419d126f9a1e96e954d.png](../_resources/82504ec09818d419d126f9a1e96e954d.png)

**问题**：如果员工ABC一起参加，赢得了10万元，怎样公平分配奖金呢？
**Shapley值计算**：
Shapley 考虑了所有可能的联盟组合，并计算每个参与者对联盟的贡献。
计算公式：
$$Shapley(i) = Σ_{S ⊆ N \setminus \{i\}} \frac{|S|!(N-|S|)!}{N!} × Contribution(i, S)$$
$i$ 是当前的参与者。
$N$是所有参与者的集合
$∣S∣$ 是联盟 $S$ 中参与者的数量。
$Contribution(i,S)$ 是参与者 $i$ 在联盟$S$中的贡献，即员工加入联盟后联盟的总收益增加了多少。
$Shapley$公式其实就是计算各种可能的组合中，用户加入以后的贡献期望。

**结果解析**：
![e1d36c23d1b91c12dadbc62c93f2928d.png](../_resources/e1d36c23d1b91c12dadbc62c93f2928d.png)
针对单打独斗能力较弱的辅助型员工C，可以看到Shapley值公平的计算了其在团队中的贡献。



# 2、SHAP适用场景
SHAP 值（SHapley Additive exPlanations）是 Shapley 值在机器学习和模型解释领域的特定应用。
在许多应用场景中，模型**可解释性**和**准确性**同等重要。逻辑回归、决策树等模型的流行和广泛应用，很大原因就在于其良好的可解释性。
但是，在工业界实际应用中，我们发现最高的准确率往往是通过复杂模型实现的，比如集成模型(CatBoost、RandomForest)或深度学习模型，这些模型即便是专家也难以解释。

论文提出的SHAP框架，可以针对**任意黑盒模型**的**每一次预测**，解析特征的贡献度。
![a50af125ee5daf1442ab80f95eb22754.png](../_resources/a50af125ee5daf1442ab80f95eb22754.png)

# 3、创新点与SHAP公式
- 引入了博弈论领域的Shapley 值理论。
- 统一了**Additive feature attribution methods(可加特征归因方法)** 这一领域的六种现有的方法：
	- **LIME**(Local Interpretable Model-agnostic Explanations)
	- **DeepLIFT**(Deep Learning Important FeaTures)
	- **Layer-Wise Relevance Propagation**（分层相关传播）
	- **Shapley regression values**（Shapley回归值）
	- **Shapley sampling values**（Shapley采样值）
	- **Quantitative Input Influence**（定量输入影响）


## 3.1 Additive feature attribution methods(可加特征归因方法)的独特属性
**定义**：**Additive feature attribution methods(可加特征归因方法)**:
 $$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i$$
 $g$是解释模型   simplified inputs
 $z' ∈ \{0, 1\}^M$ ,$M$ 简化以后的输入特征,$\phi_i ∈ R$.
### 三个关键属性
1. **局部准确性 (Local accuracy)**
对于局部特定输入$x$，$x'$是简化以后的特征，映射关系为$x = h_x(x')$，g是解释模型。则：
  $f(x) = g(x') = \phi_0 + \sum_{i=1}^{M} \phi_i x'_i$
 **解读**: 解释模型各特征归因的总和等于原始模型的输出。实际在用的过程中，不一定相等，但肯定是正相关。

2. **缺失性 (Missingness)**
$𝑥_𝑖$表示第$𝑖$个特征在简化输入中的值。如果这个值为0，表示这个特征是"缺失"的，或者说它没有被考虑在内。根据Missingness属性，这种情况下，该特征的SHAP值应当为零，反映出这个特征对预测结果没有贡献。
**公式**： $x'_i = 0 \Rightarrow \phi_i = 0$ 
**解读**：这里"缺失"是指观察不到，针对日常分析的结构化数据应该是不存在这个问题。

3. **一致性 (Consistency)**
    **定义**：如果模型变更，导致特征对模型的贡献增加(或保持不变），则解释模型中变化趋势应该一致。
    **公式**： 
$f'_x(z') - f'_x(z' \setminus i) \geq f_x(z') - f_x(z' \setminus i)$则
  $$\phi_i(f', x) \geq \phi_i(f, x)$$
  **解读**：原始模型 $f$ 和解释模型 $g$ **正相关**。

### 定理: **唯一解释模型**
只有一个可能的解释模型 $g$ 满足属性1, 2和3,如下：
$$\phi_i(f, x) = \sum_{z' \subseteq x'} \frac{|z'|!(M - |z'| - 1)!}{M!} [f_x(z') - f_x(z' \setminus i)]$$
$z' ⊆ x'$ 是其中所有非零子集的集合，|z'| 是所有非零子集的集合个数。**其中的值 $\phi_i$ 被认为是SHAP值**。


# 4、SHAP公式的计算
虽然精确计算SHAP值可能很困难，但可以通过一些近似方法来估算。
## Model-Agnostic Approximations（模型无关近似）
**Kernel SHAP** (Linear LIME + Shapley values)：模型无关，适用于任何模型。

## Model-Specific Approximations（模型相关近似）
- **Linear SHAP**：适用于特征独立不相关的线性模型
- **Tree SHAP**：适用于树模型和基于树模型的集成算法，比如XGBoost、LightGBM、CatBoost等。
- **Deep SHAP** (DeepLIFT + Shapley values)：用于计算深度学习模型，基于DeepLIFT算法，支持TensorFlow 和 PyTorch 库等主流库。~~
- Low-Order SHAP
- Max SHAP


# 附件
- [\[Paper\] A Unified Approach to Interpreting Model Predictions](https://arxiv.org/pdf/1705.07874 '')
- [\[Docs\]SHAP documentation](https://shap.readthedocs.io/en/latest/ '')