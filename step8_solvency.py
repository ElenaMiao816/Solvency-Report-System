from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def show_step_8_solvency(data: pd.DataFrame) -> None:
    st.subheader("行业偿付能力分析")
    if data is None or data.empty:
        st.info("请先完成多公司数据集成。")
        return
    frame = data.copy()
    frame["数值_numeric"] = pd.to_numeric(frame["数值"], errors="coerce")
    periods = sorted(frame["报告期"].dropna().astype(str).unique())
    selected_period = st.selectbox("行业分析报告期", periods, index=max(len(periods) - 1, 0), key="s8_solvency_period")
    company_types = sorted(frame["公司类型"].dropna().astype(str).unique())
    selected_types = st.multiselect("公司类型", company_types, default=company_types, key="s8_solvency_types")
    filtered = frame[(frame["报告期"].astype(str) == str(selected_period)) & (frame["公司类型"].isin(selected_types))]
    filtered = filtered[filtered["期间口径"].astype(str).str.contains("本季度末|期末", regex=True, na=False)]

    metric_options = {
        "核心偿付能力充足率": "CORE_SOLVENCY_RATIO",
        "综合偿付能力充足率": "COMBINED_SOLVENCY_RATIO",
        "实际资本": "ACTUAL_CAPITAL",
        "最低资本": "MINIMUM_CAPITAL",
        "净现金流": "NET_CASH_FLOW",
    }
    metric_name = st.selectbox("排名指标", list(metric_options), key="s8_solvency_metric")
    ranking = filtered[filtered["指标编码"] == metric_options[metric_name]].dropna(subset=["数值_numeric"]).sort_values("数值_numeric", ascending=False)
    if ranking.empty:
        st.warning("当前报告期没有该指标。")
        return
    fig = px.bar(ranking, x="公司", y="数值_numeric", color="公司类型", text_auto=".2f")
    fig.update_layout(title=f"{metric_name}行业排名", xaxis_tickangle=-35, legend_title_text="")
    if "充足率" in metric_name:
        fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

    ratios = filtered[filtered["指标编码"].isin(["CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"])].dropna(subset=["数值_numeric"])
    if not ratios.empty:
        box = px.box(ratios, x="指标名称", y="数值_numeric", color="公司类型", points="all")
        box.update_layout(title="偿付能力充足率行业分布", yaxis_ticksuffix="%", legend_title_text="")
        st.plotly_chart(box, use_container_width=True)

