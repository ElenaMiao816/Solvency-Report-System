# Solvency Report System

偿付能力报告平台是一个基于 Streamlit 的寿险公司偿付能力季度报告处理工具。系统支持报告更新检查、PDF 目标表页码定位、人工复核、跨页表格重构、标准化、校验和后续分析。

## 主要功能

- 逐家或批量检查偿付能力报告更新情况。
- 定位并可视化复核五类目标表的 PDF 页码。
- 提取偿付能力充足率、主要经营指标、实际资本、近三年（综合）投资收益率和最低资本表。
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

STEP2 在页面中接收模型服务地址、模型名称和 API Key。API Key 不应写入代码、配置表或 Git 历史；请为正式环境使用独立的服务凭据和访问控制。

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
