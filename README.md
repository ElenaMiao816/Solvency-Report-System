# Solvency Report System

偿付能力报告平台是一个基于 Streamlit 的寿险公司偿付能力季度报告处理工具。系统支持报告更新检查、PDF 目标表页码定位、人工复核、跨页表格重构、标准化、校验和后续分析。

当前版本：`v0.2.2`

## 报告 Profile

系统以报告 Profile 隔离不同报告类型的处理和比较范围。当前首个正式
Profile 为 `LIFE_SOLVENCY`（寿险偿付能力季度报告）：

- 支持同一 Profile 内跨公司、跨期间比较；
- 不在不同 Profile 之间合并或比较指标；
- 公司来源、指标字典、校验规则和标准模板由 Profile 指定；
- 目标表、页码定位关键词及续表/停止/排除规则可导出为配置工作簿，
  人工维护后重新上传；
- 上传配置只在当前浏览器会话内生效，不直接覆盖正式配置文件。

Profile 配置工作簿包含：

- `报告类型`：报告身份、公司范围、资源文件和比较范围；
- `目标表`：目标表 ID、名称、最大页数、完整性与续表参数；
- `定位关键词`：标题、内容、表头、首尾、续表、停止和排除关键词。

正式 Profile 描述位于 `config/report_profiles/`。复杂版式仍由代码中的
策略模块处理；配置负责声明“识别什么”，策略负责处理“如何重构”。

### 表格提取策略 Registry

`services/table_strategy_registry.py` 统一登记可用的版本化提取策略。
`目标表`配置通过 `strategy_id` 引用策略，配置中不允许填写 Python
函数或文件路径。当前五张寿险偿付能力目标表分别登记独立的 `v1`
策略。每个策略分别登记提示词、边界裁剪、提取后清洗和完整性检查
handler；共享的页面解析、跨页拼接和校验算法作为通用内核复用。

`services/table_strategy_handlers.py` 保存四类 handler 接口及现有寿险
偿付能力策略。近三年投资收益率的句式归一化、最低资本尾段恢复、
经营指标网格边界和偿付能力主表完整性门槛均由对应策略选择，不再由
提取主流程中的 `table_id` 分支决定。

结构简单的新表可先使用 `generic.grid_table.v1`。复杂新表应新增并注册
版本化策略；若配置引用未注册策略，或把专用策略配置到错误的目标表，
Profile 会在运行前校验失败。

## v0.2.2 更新

- 将寿险偿付能力季度报告改造为 Profile 驱动，支持配置工作簿导入导出。
- 建立版本化策略 Registry，独立登记提示词、边界、清洗和完整性检查处理器。
- 完善不同寿险公司在页码、跨页表格、字段别名、句式披露及预测列方面的兼容能力。
- 增加 gold standard、报告身份检查、性能优化和提取质量回归机制。
- 全量自动化测试扩展至 147 项。

## v0.2.1 更新

- 支持无文字层及混合型 PDF 的两阶段图片页码定位和逐页视觉表格提取。
- 加强跨页表格拼接、项目边界校验和句式披露归一化。
- 将原始单位作为独立元数据提取，支持单位缺失补识别与金额单位换算。
- 增加 Moonshot/Kimi 固定采样参数适配和模型接口重试机制。
- 补充扫描件、经营指标边界、单位换算及模型配置回归测试。

## 主要功能

- 逐家或批量检查偿付能力报告更新情况。
- 定位并可视化复核五类目标表的 PDF 页码。
- 自动识别扫描页和混合型 PDF，并通过低分辨率候选扫描、高分辨率候选确认定位物理页码。
- 提取偿付能力充足率、主要经营指标、实际资本、近三年（综合）投资收益率和最低资本表。
- 支持将句式披露的近三年投资收益率归一化为标准两行数据。
- 支持跨页表格拼接、网格化重构和提取质量校验。
- 生成标准化 Excel，并为后续公司报告与行业分析提供模板。

## 项目结构

```text
app.py                         Streamlit 入口
services/                      PDF 定位、表格提取、标准化和校验逻辑
config/                        公司范围、页码特征、分类与校验规则
templates/                     标准目标表和分析模板
other_doc/                     参考配置、笔记本和操作材料
step7_solvency.py              公司报告分析步骤
step8_solvency.py              行业分析步骤
requirements.txt               Python 依赖
```

`Download/` 用于存放运行过程中产生的文件，不纳入 Git 版本管理。

## 本地运行

建议使用 Python 3.12 或兼容版本的 Python：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

启动后访问 `http://localhost:8501`。

## 大模型配置

STEP1 和 STEP2 共用页面中的模型服务地址、模型名称和 API Key。处理扫描版或纯图片 PDF 时，所选模型及兼容接口必须支持 `image_url` 图片输入；仅支持文本的模型会收到明确提示，并可改为人工填写物理页码。API Key 不应写入代码、配置表或 Git 历史；请为正式环境使用独立的服务凭据和访问控制。

## Streamlit Community Cloud 部署

1. 确保需要发布的代码已推送到 `main` 分支。
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)，并选择本 GitHub 仓库。
3. 分支选择 `main`，入口文件填写 `app.py`。
4. 完成部署后，Streamlit 会生成可分享的 `*.streamlit.app` 地址。

仓库中不应上传保险公司非公开报告、用户提取结果、API Key 或 `.streamlit/secrets.toml`。

## 版本管理建议

- 每次修改使用独立分支，通过 Pull Request 合并到 `main`。
- 提交信息说明修改的业务步骤和原因。
- 发布稳定版本时创建 Git tag，例如 `v0.1.0`。
- 配置表调整与代码调整一起提交，便于追踪规则变化。
