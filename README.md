# CUMCM 2026 Team Workspace

全国大学生数学建模竞赛 2026 团队协作仓库，仅团队内部使用，用于统一管理题目材料、数据处理、模型代码、实验结果和论文。

## 目录结构

```text
data/raw/          原始数据，只读保存
data/processed/    清洗、转换后的数据
notebooks/         探索性分析与实验记录
src/               可复用的数据处理、建模与绘图代码
results/figures/   论文图
results/tables/    论文表格与关键数值
paper/             论文源文件及附录
```

## 三人协作流程

1. 每位队员克隆仓库，并在 Codex 中将本地克隆目录添加为项目。
2. 开始任务前同步 `main`，再创建个人功能分支，例如：

   ```bash
   git switch main
   git pull
   git switch -c model/problem-1
   ```

3. 小步提交，提交信息写清楚修改对象和结论，例如：

   ```bash
   git add src results
   git commit -m "model: implement baseline for problem 1"
   git push -u origin model/problem-1
   ```

4. 在 GitHub 发起 Pull Request，由另一位队员检查公式、代码、结果和论文是否一致后合并。
5. 不直接覆盖他人的结果文件；有冲突时优先保留可追溯的原始版本。

## 推荐分工

- 建模：问题抽象、变量、目标函数、约束和算法设计；
- 编程：数据清洗、模型求解、验证、绘图和复现；
- 写作：论文结构、公式、结果解释、参考文献和最终对账。

分工不是隔离。每个核心结论至少由另一位队员独立复核一次。

## 提交前检查

- 代码能够从空环境重复运行；
- 原始数据未被修改；
- 随机过程固定种子；
- 重要结果包含基线、误差或稳定性验证；
- 优化结果逐条满足约束；
- 摘要、正文、图表和程序输出中的数字一致；
- 未提交账号、密钥、个人隐私或无授权材料。

详细协作规则见 [CONTRIBUTING.md](CONTRIBUTING.md)，Codex 项目规则见 [AGENTS.md](AGENTS.md)。

## 当前建模成果

- [问题一：交会定位区域直径与覆盖圆判定](paper/问题一_交会定位区域直径与覆盖判定.md)
- [问题二：第二检测点的鲁棒选址策略](paper/问题二_第二检测点鲁棒选址策略.md)
- [问题三：全向多源自适应搜索定位与清除](paper/问题三_全向多源自适应搜索定位与清除.md)
- [问题四：混合方向源保证发现与保信号追踪](paper/问题四_混合方向源保证发现与保信号追踪.md)

复现问题一、二结果：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/question1_demo.py
python src/question2_demo.py
python src/question3_demo.py 30
python src/question4_demo.py 30
```

问题四当前默认使用 `balanced` 23 点巡检和 8° 双侧追踪；本地 30 局固定种子演练全部清除，平均总定位清除时间为 7458.7 s。若需要更强几何证明，可在正式客户端中切换 `--survey-profile certified`。
