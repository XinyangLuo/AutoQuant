# 1 常用因子算子
* 优先级：P0
* backtest/factor
参考https://platform.worldquantbrain.com/learn/operators，实现一线常用的算子方便快速因子构造，避免每次因子代码重复构造

# 2 建立Barra风险模型
* 优先级：P0
* 涉及模块：backtest/factor
删除旧的因子表，建立一张正式的因子表，以barra风格因子为底。
## 2.1 风格因子
每个因子具体含义如下
| 一级因子          | 二级因子              | 保留三级因子    | 计算公式                                                                                                                                        | 一级合成                         | 砍掉的部分                                                     |            |                                          |
| :------------ | :---------------- | :-------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------- | :-------------------------------------------------------- | ---------- | ---------------------------------------- |
| **Size**      | Size              | **LNCAP** | $\ln(\text{流通市值})$                                                                                                                          | **= LNCAP**                  | MIDCAP（需正交化，中市值效应不稳定）                                     |            |                                          |
| **Beta**      | Beta              | **BETA**  | 个股日收益 $r_t$ 对沪深300日收益 $R_t$ 做 WLS 回归（窗口 **252日**，半衰期 **63日**）：<<br>$r_t = \alpha + \beta R_t + \varepsilon_t$<<br>权重 $w_t = 0.5^{(T-t)/63}$ | **= BETA**                   | Residual Volatility（HSIGMA/DASTD/CMRA，计算复杂，属风控层）          |            |                                          |
| **Momentum**  | Momentum          | **RSTR**  | **简化版**：过去 **252日** 累计对数收益率，**滞后 11日**<<br>**严谨版**：每日 $\ln(1+r_t)$ 指数加权求和（半衰期126日），再滞后11日做11日等权平均                                           | **= RSTR**                   | STREV（短期反转与动量对冲）、Seasonality、IndMom、HALPHA                |            |                                          |
| **Value**     | BTOP              | **BTOP**  | $\text{最近报告期净资产} / \text{流通市值}$                                                                                                             | **(BTOP + ETOP + DTOP) / 3** | CETOP/EM（数据口径不统一）、Long-Term Reversal（需4年窗口）               |            |                                          |
|               | EarningsYield     | **ETOP**  | $\text{过去12个月净利润} / \text{流通市值}$                                                                                                            |                              |                                                           |            |                                          |
|               | DividendYield     | **DTOP**  | $\text{最近12个月每股股息} / \text{上月末股价}$                                                                                                          |                              | 从原 Dividend Yield 一级并入                                    |            |                                          |
| **Quality**   | Profitability     | **ROA**   | $\text{过去12个月净利润} / \text{最近报告期总资产}$                                                                                                        | **(ROA + GP + AGRO) / 3**    | EarningsVar（盈利波动）、EarningsQuality（应计项目）、Leverage（与行业绑定过强） |            |                                          |
|               | Profitability     | **GP**    | $(\text{过去12个月营收} - \text{过去12个月营业成本}) / \text{总资产}$                                                                                        |                              |                                                           |            |                                          |
|               | InvestmentQuality | **AGRO**  | 最近5年总资产对时间回归斜率 $k$：<br>$\text{AGRO} = -k / \text{mean}(\text{TA})$<<br>（负号：扩张越慢，质量越高）                                                       |                              |                                                           |            |                                          |
| **Liquidity** | Liquidity         | **STOM**  | $\ln\left(\sum_{t=1}^{21} \frac{\text{日成交额}_t}{\text{流通市值}_t}\right)$                                                                       | **= STOM**                   | STOQ/STOA（STOM 平滑版，相关性>0.9）、ATVR（252日维护成本高）               |            |                                          |
| **Growth**    | HistoricalGrowth  | **EGRO**  | 最近5年 EPS 对时间回归斜率 $k$：<br>\$\text{EGRO} = k / \text{mean}(                                                                                   | \text{EPS}                   | )\$                                                       | **= EGRO** | SGRO（与 EGRO 相关性高）、EGRLF（A股分析师覆盖率低，无数据可省） |

计算规则：
1. 计算三级因子暴露
2. MAD去极值，median ± 3×1.4826 × MAD，超出的clip到边界
3. 截面行业（申万一级行业）中位数填充缺失值
4. 在整个截面上z-score，使均值0方差1
5. 等权相加合成二级因子和一级因子
注意点：
1. 财务因子不可穿越，缺失值处理

## 2.2 因子中性化
Barra风格因子中性化，用于取代原有的分层中性化。
即使用因子原始值对行业（）dummy variable和z-score(log（市值）)做回归，取残差作为中性化后的纯净因子值。行业使用申万一级行业
详细流程：MAD 去极值 → 行业填充 → Z-Score → 截面 OLS（因子 ~ 行业哑变量 + Size_z）→ 取残差 → 残差再标准化

## 2.3 入库检查
入库前和剩余6个一级因子做Ridge Regression，看R^2来判断因子在剩余风格上的暴露，决定这是一个pure alpha还是smart beta

# 3 交易日历
* 优先级：P0
* 涉及模块：backtest/data、backtest/strategy、backtest/simulation
表记下周/月第一个交易日，便于周频和月频因子策略的调仓。

# 4 因子挖掘流程优化
* 优先级：P1（依赖于前面P0的建设）
* 与rd agent的开发相结合
* 设计因子回测pipeline的改造，需要分离每一步的评测

制定一套完整的因子挖掘pipeline，串行运行，制定每一步的淘汰标准。

* **step1: 因子原始值计算**
  * 截面缺失率：量价因子 < 10%，财务因子 < 30%
* **step2: Barra风格中性化后计算与已有因子库的相关性**
  * 与 size、industry 的相关性 < 0.05（验中性化是否成功）
* **step3: 离线 ICIR 计算**
  * 年化 ICIR 计算方式：$\frac{\mu(IC)}{\sigma(IC)} \cdot \sqrt{\frac{252}{h}}$，$h$ 为 ret 周期
  * 日频阈值：|IC| > 0.01；年化 ICIR > **1.0**；t > 2.0；正 IC 占比 > 55%
  * 月频阈值：|IC| > 0.03；年化 ICIR > 0.8；t > 2.5；正 IC 占比 > 65%
  * 任一指标不达标即淘汰
  * 备注：日频结合看 1 日和 5 日两套指标，任一通过即可（对应高频和中频因子）；月频看 1M
* **step4: 分 10 组检验单调性、稳定性**
  * 单调性测度：**Spearman corr(组号, 平均收益)**，阈值 > 0.7
* **step5: 制定策略**
  * 默认 top 10%，decay=5（日频因子），universe 默认全A（可选不同指数）
* **step6: 简单回测（向量化快速）**
  * 必检阈值（全部通过）：
    * 日频 Sharpe > 0.8，月频 Sharpe > 1.0
    * 年化收益 > 10%
    * 最大回撤 < 30%
    * Calmar > 0.5
    * 年化双边换手率 < 20 倍
  * 不通过返回 step5，agent 根据本次数据调整 decay/universe，**最多重试 3 次**
* **step7: 详细回测（含分红、摩擦成本等）**
  * 必检阈值（全部通过）：
    * 日频 Sharpe > 0.4，月频 Sharpe > 0.6
    * 年化收益 > 8%
    * 最大回撤 < 30%
    * Calmar > 0.5
    * 年化双边换手率 < 20 倍
  * 不通过返回 step5，agent 根据本次数据调整 decay/universe，**最多重试 3 次**
* **step8: 对剩余 Barra 一级因子（除市值和行业）做 Ridge Regression，按 R² 分流**
  * 首先按日频/月频分库
  * 与已有因子的最大 corr < 0.5
  * R² 分层：
    * R² < 0.10：**pure alpha**
    * 0.10 ≤ R² < 0.50：**smart beta**
    * 0.50 ≤ R² < 0.80：**边缘 smart beta**，需在残差上额外通过 ICIR 检验才保留
    * R² ≥ 0.80：丢弃（视为现有风格因子的复制品）
  * 严格版（记为TODO），用因子对Barra分格因子和现有因子的回归残差作为纯净值，看有没有ICIR增益（统一化为月度ICIR，阈值0.2），通过入库，不然淘汰，用于代替corr检验
* **step9: 操作入库**
  * 在因子 meta 中记录回测细节（decay、universe）和详细指标（每一种测试方式的结果）
  * 生成完整的 markdown 报告，包含表达式、因子设计思路、回测结果
  * 纳入到因子库中，在Barra风格因子的基础上新增列

备注：
1. 实践中step1和2可以合并
2. step3 中日频 1 日和 5 日两套指标分别评估，任一通过即可。
3. 对于不通过的因子执行reject操作，统一删除回测时产生的文件，保持目录和因子库整洁

待定：
* 基于市值和行业的 smart beta 标准（这些不能走统一中性化 pipeline）
* OOS / IS 切分检验（二期）
* 多 universe 稳健性检验（二期）

# 5 因子库可视化
* 优先级：P2
将所有因子的报告可视化，整合成一个web方便榴莲

# 6 分钟级因子
* 优先级：P2
* 将分钟级K线数据整合成天级因子