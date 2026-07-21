from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["数值_numeric"] = pd.to_numeric(result["数值"], errors="coerce")
    return result


def _latest_value(frame: pd.DataFrame, code: str, period_terms: list[str]):
    rows = frame[frame["指标编码"] == code]
    for term in period_terms:
        match = rows[rows["期间口径"].astype(str).str.contains(term, na=False)]
        if not match.empty:
            value = pd.to_numeric(match["数值"], errors="coerce").dropna()
            if not value.empty:
                return float(value.iloc[0])
    return None


def show_step_7_solvency(data: pd.DataFrame) -> None:
    st.subheader("公司级偿付能力分析")
    if data is None or data.empty:
        st.info("请先完成标准化或上传标准数据。")
        return

    companies = sorted(data["公司"].dropna().astype(str).unique())
    company = st.selectbox("选择公司", companies, key="s7_solvency_company")
    company_data = _numeric(data[data["公司"] == company])
    periods = sorted(company_data["报告期"].dropna().astype(str).unique())
    selected_period = st.selectbox("选择报告期", periods, index=max(len(periods) - 1, 0), key="s7_solvency_period")
    current = company_data[company_data["报告期"].astype(str) == str(selected_period)]

    core_ratio = _latest_value(current, "CORE_SOLVENCY_RATIO", ["本季度末", "期末"])
    combined_ratio = _latest_value(current, "COMBINED_SOLVENCY_RATIO", ["本季度末", "期末"])
    actual_capital = _latest_value(current, "ACTUAL_CAPITAL", ["本季度末", "期末"])
    minimum_capital = _latest_value(current, "MINIMUM_CAPITAL", ["本季度末", "期末"])
    cols = st.columns(4)
    cols[0].metric("核心偿付能力充足率", "-" if core_ratio is None else f"{core_ratio:.2f}%")
    cols[1].metric("综合偿付能力充足率", "-" if combined_ratio is None else f"{combined_ratio:.2f}%")
    cols[2].metric("实际资本", "-" if actual_capital is None else f"{actual_capital / 10000:,.2f} 亿元")
    cols[3].metric("最低资本", "-" if minimum_capital is None else f"{minimum_capital / 10000:,.2f} 亿元")

    trend = company_data[company_data["指标编码"].isin(["CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"])].dropna(subset=["数值_numeric"])
    if not trend.empty:
        trend = trend[trend["期间口径"].astype(str).str.contains("本季度末|期末", regex=True, na=False)]
        fig = px.line(trend, x="报告期", y="数值_numeric", color="指标名称", markers=True, labels={"数值_numeric": "百分比(%)"})
        fig.update_layout(title="偿付能力充足率趋势", yaxis_ticksuffix="%", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    capital_codes = ["CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"]
    capital = current[current["指标编码"].isin(capital_codes)].dropna(subset=["数值_numeric"])
    if not capital.empty:
        fig = px.bar(capital, x="指标名称", y="数值_numeric", color="指标名称", labels={"数值_numeric": "万元"})
        fig.update_layout(title="实际资本构成", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看标准化数据"):
        st.dataframe(current, use_container_width=True, hide_index=True)

