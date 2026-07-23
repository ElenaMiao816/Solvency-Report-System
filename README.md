# Solvency Report System

偿付能力报告平台是一个基于 Streamlit 的寿险公司偿付能力季度报告处理工具。系统支持报告更新检查、PDF 目标表页码定位、人工复核、跨页表格重构、标准化、校验和后续分析。

当前版本：`v0.2.1`

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
