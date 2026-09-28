"""CROSS II 主程序（Spyder 版本）。

使用方法：
1. 将本文件、CROSS_II_function_spyder.py 和 CROSS_II_config.xlsx 放在同一目录。
2. 在 Spyder 顶部“运行配置”中选择“在当前控制台执行”。
3. 修改下方 SETTINGS，然后按 F5 运行整个文件。

配置表 A 列 control=Y 的公司才会进入下载流程。数据提取默认处理报告目录
中的全部 PDF，不再写死为“财信人寿”。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import camelot
import pandas as pd

from CROSS_II_function_spyder import (
    company_name_from_pdf,
    download_pdf,
    extract_tables_from_pdf,
    get_company_params,
    get_download_url,
    offset,
    write_to_excel,
)


# ============================== SETTINGS ==============================
# 脚本、函数文件和配置表默认放在同一目录，因此不再写死用户目录。
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "CROSS_II_config.xlsx"

# 输出目录默认为：当前目录\output\季度\报告。
OUTPUT_BASE_DIR = SCRIPT_DIR / "output"

# 为空时由 Selenium Manager 自动匹配 EdgeDriver；只有离线环境才填写 exe 路径。
EDGE_DRIVER_PATH: Optional[str] = None

# 可选值："all"、"download"、"extract"、"major"。
RUN_MODE = "all"

# None 表示提取报告目录中的所有公司；也可写成 {"财信人寿", "平安寿险"}。
COMPANIES_TO_EXTRACT: Optional[set[str]] = None


FIELDS = [
    "综合偿付能力充足率",
    "核心偿付能力充足率",
    "综合偿付能力溢额",
    "核心偿付能力溢额",
    "认可资产",
    "认可负债",
    "实际资本",
    "核心一级资本",
    "核心二级资本",
    "附属一级资本",
    "附属二级资本",
    "最低资本",
    "保险业务收入",
    "净利润",
    "总资产",
    "净资产",
    "保险合同负债",
    "计入核心一级资本的保单未来盈余",
    "计入核心二级资本的保单未来盈余",
    "计入附属一级资本的保单未来盈余",
    "计入附属二级资本的保单未来盈余",
    "独立账户负债",
    "量化风险最低资本",
    "量化风险最低资本（未考虑特征系数前）",
    "寿险业务保险风险最低资本合计",
    "寿险业务保险风险-损失发生风险最低资本",
    "寿险业务保险风险-退保风险最低资本",
    "寿险业务保险风险-费用风险最低资本",
    "寿险业务保险风险-风险分散效应",
    "非寿险业务保险风险最低资本合计",
    "非寿险业务保险风险-保费及准备金风险最低资本",
    "非寿险业务保险风险-巨灾风险最低资本",
    "非寿险业务保险风险-风险分散效应",
    "市场风险-最低资本合计",
    "市场风险-利率风险最低资本",
    "市场风险-权益价格风险最低资本",
    "市场风险-房地产价格风险最低资本",
    "市场风险-境外固定收益类资产价格风险最低资本",
    "市场风险-境外权益类资产价格风险最低资本",
    "市场风险-汇率风险最低资本",
    "市场风险-风险分散效应",
    "信用风险-最低资本合计",
    "信用风险-利差风险最低资本",
    "信用风险-交易对手违约风险最低资本",
    "信用风险-风险分散效应",
    "量化风险分散效应",
    "特定类别保险合同损失吸收效应",
    "损失吸收调整-不考虑上限",
    "损失吸收效应调整上限",
    "控制风险最低资本",
    "附加资本",
    "逆周期附加资本",
    "D-SII附加资本",
    "G-SII附加资本",
    "其他附加资本",
]


def load_config(config_file: Path) -> tuple[pd.DataFrame, str]:
    if not config_file.exists():
        raise FileNotFoundError(f"找不到配置表：{config_file}")
    dataframe = pd.read_excel(config_file, sheet_name="config")
    required = {
        "control",
        "公司",
        "网址",
        "text_to_find",
        "xpath",
        "季度",
        "保存文件名",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"配置表缺少字段：{sorted(missing)}")

    quarters = dataframe["季度"].dropna().astype(str)
    if quarters.empty:
        raise ValueError("配置表“季度”列为空。请先在 Excel 中修改 I2、重新计算并保存。")
    quarter = quarters.iloc[0].strip()
    return dataframe, quarter


def selected_for_download(value: object) -> bool:
    return str(value).strip().upper() == "Y"


def download_reports(
    dataframe: pd.DataFrame,
    report_dir: Path,
    log_file: Path,
) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    selected_count = int(dataframe["control"].map(selected_for_download).sum())
    print(f"下载目录：{report_dir}")
    print(f"配置表中 control=Y 的公司数：{selected_count}")
    if selected_count == 0:
        print("未选择下载公司；跳过下载。")
        return dataframe

    for index, row in dataframe.iterrows():
        if not selected_for_download(row["control"]):
            continue
        company = str(row["公司"]).strip()
        quarter = str(row["季度"]).strip()
        url = str(row["网址"]).strip()
        file_name = str(row["保存文件名"]).strip()
        text_to_find = str(row["text_to_find"]).strip()
        xpath = row.get("xpath")
        print(f"\n[{company} {quarter}] 开始下载")

        try:
            pdf_url = get_download_url(
                url=url,
                text_to_find=text_to_find,
                xpath=xpath,
                company=company,
                driver_path=EDGE_DRIVER_PATH,
            )
            dataframe.at[index, "pdf_url"] = pdf_url
            dataframe.at[index, "operation_time"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except Exception as exc:
            dataframe.at[index, "download_status"] = f"无法获取下载链接: {exc}"
            print(dataframe.at[index, "download_status"])
            continue

        try:
            success = download_pdf(
                pdf_url=pdf_url,
                file_path=report_dir,
                file_name=file_name,
                company=company,
                driver_path=EDGE_DRIVER_PATH,
            )
            dataframe.at[index, "download_status"] = "下载成功" if success else "下载失败"
        except Exception as exc:
            dataframe.at[index, "download_status"] = f"下载失败: {exc}"
        dataframe.at[index, "download_time"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    log_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_excel(log_file, sheet_name="log", index=False)
    print(f"下载日志：{log_file}")
    return dataframe


def _table_dataframes(pdf_file: Path, company_name: str) -> Iterable[pd.DataFrame]:
    if company_name in {"中邮人寿", "海保人寿", "人保健康", "华汇人寿"}:
        yield from extract_tables_from_pdf(pdf_file)
    else:
        tables = camelot.read_pdf(
            filepath=str(pdf_file), **get_company_params(company_name, pdf_file)
        )
        for table in tables:
            yield table.df


def _table_is_relevant(dataframe: pd.DataFrame) -> bool:
    text = str(dataframe.values.flatten())
    keywords = {
        "偿付能力",
        "寿险业务保险风险",
        "控制风险最低资本",
        "保险合同负债",
        "保单未来盈余",
        "认可资产",
        "认可负债",
        "实际资本",
    }
    return any(keyword in text for keyword in keywords)


def process_pdf(
    pdf_file: Path,
    company_name: str,
    fields: list[str],
    result: pd.DataFrame,
) -> None:
    print(f"提取：{company_name} -> {pdf_file.name}")
    for field in fields:
        if field not in result.columns:
            result[field] = None
        if company_name not in result.index:
            result.loc[company_name, :] = None

    try:
        tables = list(_table_dataframes(pdf_file, company_name))
        relevant_tables = [table for table in tables if _table_is_relevant(table)]
        if not relevant_tables:
            print(f"{company_name}：未识别到相关表格")
            return

        column_offsets = [2, 1] if company_name == "泰康养老" else [1, 2]
        for field in fields:
            found_value = None
            for col_offset in column_offsets:
                for table in relevant_tables:
                    value = offset(
                        table,
                        keyphrase=field,
                        company_name=company_name,
                        cols=col_offset,
                    )
                    if value is None or pd.isna(value) or str(value).strip() == "":
                        continue
                    value_text = str(value).split("\n", 1)[0].strip()
                    if value_text:
                        found_value = value_text
                        break
                if found_value is not None:
                    break
            result.at[company_name, field] = found_value if found_value is not None else "无"
    except Exception as exc:
        print(f"{company_name}：PDF处理失败：{exc}")


def extract_metrics(report_dir: Path, output_file: Path) -> pd.DataFrame:
    pdf_files = sorted(report_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"报告目录没有 PDF：{report_dir}")
        return pd.DataFrame(columns=FIELDS)

    result = pd.DataFrame(columns=FIELDS)
    for pdf_file in pdf_files:
        company_name = company_name_from_pdf(pdf_file)
        if COMPANIES_TO_EXTRACT and company_name not in COMPANIES_TO_EXTRACT:
            continue
        process_pdf(pdf_file, company_name, FIELDS, result)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_file, index=True, index_label="公司")
    print(f"偿付能力数据：{output_file}")
    return result


def extract_major_information(report_dir: Path, output_file: Path) -> None:
    pdf_files = sorted(report_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"报告目录没有 PDF：{report_dir}")
        return
    write_to_excel(pdf_files, ["重大投资", "重大融资"], output_file)
    print(f"重大投资及融资信息：{output_file}")


def main() -> None:
    start_time = time.time()
    dataframe, quarter = load_config(CONFIG_FILE)
    quarter_dir = OUTPUT_BASE_DIR / quarter
    report_dir = quarter_dir / "报告"
    log_file = quarter_dir / f"CROSS_II_log_{quarter}.xlsx"
    result_file = quarter_dir / f"df_result_{quarter}.xlsx"
    major_file = quarter_dir / f"重大融资_{quarter}.xlsx"

    mode = RUN_MODE.strip().lower()
    allowed_modes = {"all", "download", "extract", "major"}
    if mode not in allowed_modes:
        raise ValueError(f"RUN_MODE 必须是 {sorted(allowed_modes)} 之一")

    if mode in {"all", "download"}:
        download_reports(dataframe, report_dir, log_file)
    if mode in {"all", "extract"}:
        extract_metrics(report_dir, result_file)
    if mode in {"all", "major"}:
        extract_major_information(report_dir, major_file)

    print(f"总运行时间：{time.time() - start_time:.1f} 秒")


if __name__ == "__main__":
    main()
