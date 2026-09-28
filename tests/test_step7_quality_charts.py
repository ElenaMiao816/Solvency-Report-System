import unittest

import pandas as pd
import plotly.graph_objects as go

from services.solvency_step7_charts import (
    build_solvency_ratio_combo_chart,
    report_period_combo_bar_color_map,
)
from services.solvency_step7_quality import combo_figure
from services.solvency_step7_quality_charts import company_quality_chart


def marks(spec):
    if "mark" in spec:
        yield spec
    for layer in spec.get("layer", []):
        yield from marks(layer)


def records(spec):
    return [row for dataset in spec["datasets"].values() for row in dataset]


class QualityChartStyleTests(unittest.TestCase):
    def combo(self):
        periods = ["2025Q1", "2025Q3", "2026Q1"]
        frame = pd.DataFrame([
            {"公司": "甲公司", "报告期": period, "指标编码": code, "指标名称": code,
             "数值": value, "期间口径": "本季度末数", "单位": "亿元"}
            for period in periods
            for code, value in [("SIGNED_PREMIUM", 100), ("NEW_BUSINESS_MARGIN", 5),
                                ("CORE_SOLVENCY_RATIO", 150), ("COMBINED_SOLVENCY_RATIO", 200)]
        ])
        fig, _, _ = combo_figure(frame, periods, "SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN")
        fig.update_yaxes(range=[0, 120], secondary_y=False)
        fig.update_yaxes(range=[0, 6], secondary_y=True)
        return frame, periods, fig

    def test_new_panels_use_reference_bar_geometry_at_every_density(self):
        frame, periods, fig = self.combo()
        for count in [3, 7, 12]:
            with self.subTest(count=count):
                old = build_solvency_ratio_combo_chart(frame, periods, panel_count=count).to_dict()
                new = company_quality_chart(fig, "甲公司", count, unit="亿元").to_dict()
                old_bar = next(m for m in marks(old) if m["mark"]["type"] == "bar")
                new_bar = next(m for m in marks(new) if m["mark"]["type"] == "bar")
                self.assertEqual(new["height"], old["height"])
                self.assertEqual(new["config"], old["config"])
                self.assertEqual(new_bar["mark"], old_bar["mark"] | {"color": fig.data[0].marker.color})
                self.assertEqual(new_bar["encoding"]["x"], old_bar["encoding"]["x"])

    def test_combo_preserves_independent_units_missing_gaps_and_readable_labels(self):
        _, _, fig = self.combo()
        fig.data[1].y = [5, None, 5]
        spec = company_quality_chart(fig, "甲公司", 7, unit="亿元").to_dict()
        rows = [r for r in records(spec) if "线段" in r]
        self.assertNotEqual(rows[0]["线段"], rows[2]["线段"])
        self.assertIsNone(rows[1]["线值"])
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")
        self.assertEqual(rows[0]["柱标签位置"], 50)
        self.assertEqual(rows[0]["柱标签"], "100")
        self.assertEqual(rows[0]["线标签"], "5.0%")
        self.assertEqual([r["报告期"] for r in rows], ["2025Q1", "2025Q3", "2026Q1"])

    def test_combo_bar_period_colors_never_match_the_pink_line(self):
        periods = ["2024Q4", "2025Q2", "2025Q4", "2026Q2"]
        colors = report_period_combo_bar_color_map(periods)
        self.assertEqual(colors["2026Q2"], "#098E7E")
        self.assertNotEqual(colors["2026Q2"], "#FD349C")

        frame = pd.DataFrame([
            {
                "公司": "甲公司",
                "报告期": period,
                "指标编码": code,
                "指标名称": code,
                "数值": value,
                "期间口径": "累计数",
                "单位": "%",
            }
            for period in periods
            for code, value in (
                ("CORE_SOLVENCY_RATIO", 150),
                ("COMBINED_SOLVENCY_RATIO", 200),
            )
        ])
        solvency = build_solvency_ratio_combo_chart(frame, periods).to_dict()
        bar = next(mark for mark in marks(solvency) if mark["mark"]["type"] == "bar")
        self.assertEqual(bar["encoding"]["color"]["scale"]["range"][-1], "#098E7E")

        fig, _, _ = combo_figure(
            frame.rename(columns={"指标编码": "原指标编码"})
            .assign(
                指标编码=lambda data: data["原指标编码"].map({
                    "CORE_SOLVENCY_RATIO": "INVESTMENT_RETURN",
                    "COMBINED_SOLVENCY_RATIO": "COMPREHENSIVE_INVESTMENT_RETURN",
                }),
                指标名称=lambda data: data["指标编码"],
            ),
            periods,
            "INVESTMENT_RETURN",
            "COMPREHENSIVE_INVESTMENT_RETURN",
            scope="cumulative",
        )
        fig.update_yaxes(range=[0, 220], secondary_y=False)
        fig.update_yaxes(range=[0, 220], secondary_y=True)
        quality = company_quality_chart(
            fig,
            "甲公司",
            7,
            unit="%",
            percentage_bar=True,
            color_by_period=True,
        ).to_dict()
        quality_bar = next(mark for mark in marks(quality) if mark["mark"]["type"] == "bar")
        self.assertEqual(quality_bar["encoding"]["color"]["scale"]["range"][-1], "#098E7E")

    def test_stack_preserves_signed_bounds_and_does_not_invent_missing_values(self):
        fig = go.Figure([
            go.Bar(x=["2025Q3", "2026Q1"], y=[20, None], name="甲项", marker_color="#00B8F5"),
            go.Bar(x=["2025Q3", "2026Q1"], y=[-5, 30], name="乙项", marker_color="#FD349C"),
        ])
        fig.update_yaxes(range=[-6, 33])
        spec = company_quality_chart(fig, "甲公司", 7, unit="亿元").to_dict()
        rows = [r for r in records(spec) if "指标名称" in r]
        self.assertEqual(len(rows), 3)
        negative = next(r for r in rows if r["数值"] == -5)
        self.assertEqual((negative["起点"], negative["终点"]), (0, -5))
        bar = next(m for m in marks(spec) if m["mark"]["type"] == "bar")
        self.assertIsNone(bar["encoding"]["y"]["stack"])

    def test_pie_radius_and_labels_follow_the_same_container_size(self):
        fig = go.Figure(go.Pie(labels=["甲项", "乙项"], values=[25, 75], marker_colors=["#00B8F5", "#FD349C"]))
        spec = company_quality_chart(fig, "甲公司", 7, unit="亿元").to_dict()
        arc = next(m for m in marks(spec) if m["mark"]["type"] == "arc")
        label = next(m for m in marks(spec) if m["mark"]["type"] == "text")
        radius = arc["mark"]["outerRadius"]["expr"]
        self.assertIn("min(width, height)", radius)
        self.assertIn(radius, label["mark"]["radius"]["expr"])
        self.assertEqual([r["占比"] for r in records(spec)], [.25, .75])

    def test_waterfall_retains_disclosed_endpoint_and_missing_adjustment(self):
        fig = go.Figure(go.Waterfall(
            x=["净资产", "扣减", "未披露项", "增加", "核心一级资本"],
            customdata=["净资产", "扣减", "未披露项", "增加", "核心一级资本"],
            y=[100, -15, None, 20, 110],
            measure=["absolute", "relative", "relative", "relative", "absolute"],
        ))
        fig.update_yaxes(range=[0, 121])
        spec = company_quality_chart(fig, "甲公司", 7, unit="亿元").to_dict()
        rows = [r for r in records(spec) if "指标名称" in r]
        self.assertEqual((rows[1]["起点"], rows[1]["终点"]), (100, 85))
        self.assertIsNone(rows[2]["数值"])
        self.assertIsNone(rows[2]["终点"])
        self.assertEqual((rows[3]["起点"], rows[3]["终点"]), (85, 105))
        self.assertEqual((rows[4]["起点"], rows[4]["终点"]), (0, 110))

    def test_radar_missing_value_does_not_become_zero_or_close_polygon(self):
        fig = go.Figure(go.Scatterpolar(r=[1, 2, None, -1, 3, 4], theta=list("abcdef")))
        fig.update_layout(polar=dict(radialaxis=dict(range=[-2, 5])))
        spec = company_quality_chart(fig, "甲公司", 7, unit="%").to_dict()
        rows = records(spec)
        edges = {r["边"] for r in rows if "边" in r}
        self.assertEqual(edges, {0, 3, 4, 5})
        self.assertFalse(any(m["mark"].get("interpolate") == "linear-closed" for m in marks(spec)))
        self.assertTrue(any(r.get("数值") == -1 for r in rows))


if __name__ == "__main__":
    unittest.main()
