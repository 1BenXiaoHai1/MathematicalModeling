# 特征方法1

### 一、时域特征

对于一个长度为 \( N \) 的离散振动信号序列 $$ x(n) = \{x_1, x_2, \ldots, x_N\} $$，计算以下特征：

**均值**：
$$ \mu = \frac{1}{N} \sum_{i=1}^{N} x_i $$

**均方根值**：
$$ X_{\text{rms}} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} x_i^2 } $$

**峰值**：
$$ X_{\text{peak}} = \max( |x_i| ) $$

**峰峰值**：
$$ X_{\text{pp}} = \max(x_i) - \min(x_i) $$

**偏度**：
$$ \gamma = \frac{ \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^3 }{ \left( \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 } \right)^3 } $$

**峭度**：
$$ \kappa = \frac{ \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^4 }{ \left( \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 \right)^2 } $$

**波形因子**：
$$ S_f = \frac{ X_{\text{rms}} }{ \frac{1}{N} \sum_{i=1}^{N} |x_i| } $$

**脉冲因子**：
$$ I_f = \frac{ X_{\text{peak}} }{ \frac{1}{N} \sum_{i=1}^{N} |x_i| } $$

**裕度因子**：
$$ C_f = \frac{ X_{\text{peak}} }{ X_{\text{rms}} } $$

**方差**：
$$ \sigma^2 = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2 $$

### 二、频域特征

对 $$x(n) $$进行离散傅里叶变换得到频谱 $$ X(k) = \{X_1, X_2, \ldots, X_M\} $$，其中 $$ M = \lfloor N/2 \rfloor $$，对应频率 $$ f_k = \frac{k \cdot f_s}{N} $$，$$ k = 0, 1, \ldots, M-1 $$，$$ f_s $$ 为采样频率。

**频谱重心**：
$$ FC = \frac{ \sum_{k=1}^{M} f_k \cdot |X_k| }{ \sum_{k=1}^{M} |X_k| } $$

**均方频率**：
$$ MSF = \frac{ \sum_{k=1}^{M} f_k^2 \cdot |X_k| }{ \sum_{k=1}^{M} |X_k| } $$

**频率方差**：
$$ VF = \frac{ \sum_{k=1}^{M} (f_k - FC)^2 \cdot |X_k| }{ \sum_{k=1}^{M} |X_k| } $$

**频谱峭度**：
$$ SK = \frac{ \frac{1}{M} \sum_{k=1}^{M} (|X_k| - \bar{X})^4 }{ \left( \frac{1}{M} \sum_{k=1}^{M} (|X_k| - \bar{X})^2 \right)^2 } $$
其中 \( \bar{X} = \frac{1}{M} \sum_{k=1}^{M} |X_k| \)

### 三、轴承故障特征频率

给定轴承转速 \( n \) (rpm)、滚动体直径 \( d \)、节径 \( D \)、滚动体个数 \( N_d \)

**转频**：
$$ f_r = \frac{n}{60} $$

**外圈故障特征频率**：
$$ \text{BPFO} = f_r \cdot \frac{N_d}{2} \cdot \left( 1 - \frac{d}{D} \right) $$

**内圈故障特征频率**：
$$ \text{BPFI} = f_r \cdot \frac{N_d}{2} \cdot \left( 1 + \frac{d}{D} \right) $$

**滚动体故障特征频率**：
$$ \text{BSF} = f_r \cdot \frac{D}{d} \cdot \left[ 1 - \left( \frac{d}{D} \right)^2 \right] $$

### 四、小波包能量特征

对信号进行 \( L \) 层小波包分解，得到 $$2^L $$个子频带。第 $$ j $$ 个子频带的小波包系数为 $$ w_{j, m} $$，$$ m = 1, 2, \ldots, M_j $$

**子带能量**：
$$ E_j = \sum_{m=1}^{M_j} |w_{j, m}|^2 $$

**总能量**：
$$ E_{\text{total}} = \sum_{j=1}^{2^L} E_j $$

**归一化能量百分比**：
$$ p_j = \frac{E_j}{E_{\text{total}}} $$

**小波包能量熵**：
$$ H_{\text{wp}} = - \sum_{j=1}^{2^L} p_j \cdot \log(p_j) $$

### 五、特征向量构建

最终特征向量 $$\mathbf{f} $$由以上所有特征拼接而成：

$$ \mathbf{f} = [\mu, X_{\text{rms}}, X_{\text{peak}}, \ldots, FC, MSF, \ldots, p_1, p_2, \ldots, p_{2^L}]^T \in \mathbb{R}^D $$

其中 \( D \) 为特征总维数。

# 特征方法2

基于时间序列的驱动端（DE）、风扇端（FE）和基座（BA）加速度数据的特征提取数学表达式如下：

### 一、多通道时间序列表示

设三个通道的加速度时间序列为：

驱动端：$$ \mathbf{x}_{\text{DE}} = \{x_{\text{DE},1}, x_{\text{DE},2}, \ldots, x_{\text{DE},N}\} $$

风扇端：$$ \mathbf{x}_{\text{FE}} = \{x_{\text{FE},1}, x_{\text{FE},2}, \ldots, x_{\text{FE},N}\} $$

基座：$$\mathbf{x}_{\text{BA}} = \{x_{\text{BA},1}, x_{\text{BA},2}, \ldots, x_{\text{BA},N}\} $$

其中 \( N \) 为每个通道的数据点数。

### 二、单通道时域特征（以DE为例）

**均值**：
$$ \mu_{\text{DE}} = \frac{1}{N} \sum_{i=1}^{N} x_{\text{DE},i} $$

**均方根值**：
$$ X_{\text{rms,DE}} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} x_{\text{DE},i}^2 } $$

**峰值**：
$$ X_{\text{peak,DE}} = \max( |x_{\text{DE},i}| ) $$

**峭度**：
$$ \kappa_{\text{DE}} = \frac{ \frac{1}{N} \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})^4 }{ \left( \frac{1}{N} \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})^2 \right)^2 } $$

**偏度**：
$$ \gamma_{\text{DE}} = \frac{ \frac{1}{N} \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})^3 }{ \left( \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})^2 } \right)^3 } $$

### 三、通道间时域关系特征

**DE-FE互相关系数**：
$$ \rho_{\text{DE-FE}} = \frac{ \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})(x_{\text{FE},i} - \mu_{\text{FE}}) }{ \sqrt{ \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})^2 } \sqrt{ \sum_{i=1}^{N} (x_{\text{FE},i} - \mu_{\text{FE}})^2 } } $$

**DE-BA互相关系数**：
$$ \rho_{\text{DE-BA}} = \frac{ \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})(x_{\text{BA},i} - \mu_{\text{BA}}) }{ \sqrt{ \sum_{i=1}^{N} (x_{\text{DE},i} - \mu_{\text{DE}})^2 } \sqrt{ \sum_{i=1}^{N} (x_{\text{BA},i} - \mu_{\text{BA}})^2 } } $$

**FE-BA互相关系数**：
$$ \rho_{\text{FE-BA}} = \frac{ \sum_{i=1}^{N} (x_{\text{FE},i} - \mu_{\text{FE}})(x_{\text{BA},i} - \mu_{\text{BA}}) }{ \sqrt{ \sum_{i=1}^{N} (x_{\text{FE},i} - \mu_{\text{FE}})^2 } \sqrt{ \sum_{i=1}^{N} (x_{\text{BA},i} - \mu_{\text{BA}})^2 } } $$

### 四、多通道频域特征

对每个通道进行离散傅里叶变换：

$$ X_{\text{DE}}(k) = \sum_{n=0}^{N-1} x_{\text{DE},n} e^{-j2\pi kn/N} $$

$$ X_{\text{FE}}(k) = \sum_{n=0}^{N-1} x_{\text{FE},n} e^{-j2\pi kn/N} $$

$$ X_{\text{BA}}(k) = \sum_{n=0}^{N-1} x_{\text{BA},n} e^{-j2\pi kn/N} $$

**DE通道频谱重心**：
$$ FC_{\text{DE}} = \frac{ \sum_{k=0}^{N-1} f_k |X_{\text{DE}}(k)| }{ \sum_{k=0}^{N-1} |X_{\text{DE}}(k)| } $$

**通道间频谱相干性**（DE-FE）：
$$ C_{\text{DE-FE}}(f) = \frac{ |S_{\text{DE-FE}}(f)|^2 }{ S_{\text{DE}}(f) S_{\text{FE}}(f) } $$
其中 $$ S_{\text{DE}}(f) $$, $$ S_{\text{FE}}(f) $$为功率谱密度，$$ S_{\text{DE-FE}}(f) $$ 为互功率谱密度。

### 五、时频域特征（小波包变换）

对每个通道进行L层小波包分解：

**DE通道第j子带能量**：
$$ E_{\text{DE},j} = \sum_{m=1}^{M_j} |w_{\text{DE},j,m}|^2 $$

**多通道能量比特征**：
$$ R_{\text{DE/FE},j} = \frac{ E_{\text{DE},j} }{ E_{\text{FE},j} } $$

$$ R_{\text{DE/BA},j} = \frac{ E_{\text{DE},j} }{ E_{\text{BA},j} } $$

$$ R_{\text{FE/BA},j} = \frac{ E_{\text{FE},j} }{ E_{\text{BA},j} } $$

### 六、最终特征向量构建

综合所有通道的特征，构建最终特征向量：

$$ \mathbf{f} = [\mu_{\text{DE}}, \kappa_{\text{DE}}, \mu_{\text{FE}}, \kappa_{\text{FE}}, \mu_{\text{BA}}, \kappa_{\text{BA}}, \rho_{\text{DE-FE}}, \rho_{\text{DE-BA}}, \rho_{\text{FE-BA}}, FC_{\text{DE}}, FC_{\text{FE}}, FC_{\text{BA}}, E_{\text{DE},1}, \ldots, E_{\text{DE},2^L}, R_{\text{DE/FE},1}, \ldots, R_{\text{DE/FE},2^L}]^T $$

该特征向量充分利用了三个通道的时域、频域、时频域信息以及通道间的相互关系，为轴承故障诊断提供了丰富的特征表示。