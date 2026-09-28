import unittest

import pandas as pd

from services.solvency_normalizer import STANDARD_COLUMNS
from services.solvency_validator import validate_standard_data


CORE_METRICS = [
    ("RECOGNIZED_ASSETS", "认可资产", "万元", "金额"),
    ("RECOGNIZED_LIABILITIES", "认可负债", "万元", "金额"),
    ("ACTUAL_CAPITAL", "实际资本", "万元", "金额"),
    ("MINIMUM_CAPITAL", "最低资本", "万元", "金额"),
    ("CORE_SOLVENCY_RATIO", "核心偿付能力充足率", "%", "百分比"),
    ("COMBINED_SOLVENCY_RATIO", "综合偿付能力充足率", "%", "百分比"),
]


def taxonomy_frame() -> pd.DataFrame:
    rows = []
    for code, name, unit, data_type in CORE_METRICS:
        rows.append({
            "指标编码": code,
            "指标名称": name,
            "标准单位": unit,
            "数据类型": data_type,
            "允许期间口径": "本季度末数|期末数|上季度末数|期初数|下季度末预测数",
        })
    for code, name in [
        ("CORE_T1_CAPITAL", "核心一级资本"),
        ("CORE_T2_CAPITAL", "核心二级资本"),
        ("ANC_T1_CAPITAL", "附属一级资本"),
        ("ANC_T2_CAPITAL", "附属二级资本"),
        ("QUANT_RISK_CAPITAL", "量化风险最低资本"),
        ("CONTROL_RISK_CAPITAL", "控制风险最低资本"),
        ("ADDITIONAL_CAPITAL", "附加资本"),
        ("CORE_SOLVENCY_SURPLUS", "核心偿付能力溢额"),
        ("COMBINED_SOLVENCY_SURPLUS", "综合偿付能力溢额"),
    ]:
        rows.append({
            "指标编码": code,
            "指标名称": name,
            "标准单位": "万元",
            "数据类型": "金额",
            "允许期间口径": "本季度末数|期末数|上季度末数|期初数|下季度末预测数",
        })
    return pd.DataFrame(rows)


def standard_row(code: str, value, *, period: str = "本季度末数", **overrides) -> dict:
    taxonomy = taxonomy_frame().set_index("指标编码")
    metric = taxonomy.loc[code]
    row = {column: "" for column in STANDARD_COLUMNS}
    row.update({
        "公司": "测试人寿",
        "原始公司名称": "测试人寿",
        "标准公司名称": "测试人寿",
        "公司统一编码": "TEST_LIFE",
        "公司类型": "寿险",
        "同业分类": "中型公司",
        "报告类型": "LIFE_SOLVENCY",
        "报告年度": 2026,
        "报告季度": "Q1",
        "报告期": "2026Q1",
        "指标编码": code,
        "指标名称": metric["指标名称"],
        "期间口径": period,
        "数值": value,
        "单位": metric["标准单位"],
        "数据类型": metric["数据类型"],
        "来源页码": "10",
        "来源工作表": "主要指标表",
    })
    row.update(overrides)
    return row


def rules_frame(*rule_ids: str) -> pd.DataFrame:
    definitions = {
        "ACTUAL_CAPITAL_COMPONENTS": ("实际资本等于四类资本之和", 1),
        "DUPLICATE_ACTUAL_CAPITAL": ("实际资本跨表一致性", 1),
        "DUPLICATE_RECOGNIZED_ASSETS": ("认可资产跨表一致性", 1),
    }
    return pd.DataFrame([
        {
            "规则ID": rule_id,
            "规则名称": definitions[rule_id][0],
            "适用期间": "本季度末数|期末数|上季度末数|期初数",
            "容差": definitions[rule_id][1],
            "启用": "是",
        }
        for rule_id in rule_ids
    ])


class Step4ValidatorTests(unittest.TestCase):
    def test_cross_table_rule_compares_equivalent_period_labels(self):
        first = standard_row("ACTUAL_CAPITAL", 100, period="本季度末数")
        second = standard_row(
            "ACTUAL_CAPITAL",
            103,
            period="期末数",
            来源工作表="实际资本表",
        )
        result = validate_standard_data(
            pd.DataFrame([first, second]),
            rules_frame("DUPLICATE_ACTUAL_CAPITAL"),
        )
        row = result[result["rule_id"] == "DUPLICATE_ACTUAL_CAPITAL"].iloc[0]
        self.assertEqual(row["period"], "当前期末")
        self.assertEqual(row["status"], "未通过")
        self.assertEqual(row["difference"], 3)

    def test_duplicate_recognized_assets_compares_main_and_asset_table(self):
        main = standard_row("RECOGNIZED_ASSETS", 16526.99169, period="本季度末数")
        asset = standard_row(
            "RECOGNIZED_ASSETS",
            16526.99169,
            period="期末数",
            来源工作表="S03-认可资产表",
        )
        result = validate_standard_data(
            pd.DataFrame([main, asset]),
            rules_frame("DUPLICATE_RECOGNIZED_ASSETS"),
        )
        row = result[result["rule_id"] == "DUPLICATE_RECOGNIZED_ASSETS"].iloc[0]
        self.assertEqual(row["period"], "当前期末")
        self.assertEqual(row["status"], "通过")
        self.assertEqual(row["difference"], 0)

    def test_business_rules_are_isolated_by_company(self):
        rows = [
            standard_row("ACTUAL_CAPITAL", 100),
            standard_row("ACTUAL_CAPITAL", 200, 公司="另一人寿", 标准公司名称="另一人寿", 公司统一编码="OTHER"),
        ]
        result = validate_standard_data(
            pd.DataFrame(rows),
            rules_frame("DUPLICATE_ACTUAL_CAPITAL"),
        )
        duplicate_rows = result[result["rule_id"] == "DUPLICATE_ACTUAL_CAPITAL"]
        self.assertEqual(set(duplicate_rows["status"]), {"不适用"})
        self.assertEqual(len(duplicate_rows), 2)

    def test_formula_rule_uses_equivalent_period_group(self):
        rows = [
            standard_row("ACTUAL_CAPITAL", 100, period="本季度末数"),
            standard_row("CORE_T1_CAPITAL", 70, period="期末数", 来源工作表="实际资本表"),
            standard_row("CORE_T2_CAPITAL", 10, period="期末数", 来源工作表="实际资本表"),
            standard_row("ANC_T1_CAPITAL", 15, period="期末数", 来源工作表="实际资本表"),
            standard_row("ANC_T2_CAPITAL", 5, period="期末数", 来源工作表="实际资本表"),
        ]
        result = validate_standard_data(
            pd.DataFrame(rows),
            rules_frame("ACTUAL_CAPITAL_COMPONENTS"),
        )
        row = result[result["rule_id"] == "ACTUAL_CAPITAL_COMPONENTS"].iloc[0]
        self.assertEqual(row["period"], "当前期末")
        self.assertEqual(row["status"], "通过")

    def test_business_rules_ignore_non_current_period_groups(self):
        rows = [
            standard_row("ACTUAL_CAPITAL", 100, period="本季度末数"),
            standard_row("CORE_T1_CAPITAL", 70, period="期末数"),
            standard_row("CORE_T2_CAPITAL", 10, period="期末数"),
            standard_row("ANC_T1_CAPITAL", 15, period="期末数"),
            standard_row("ANC_T2_CAPITAL", 5, period="期末数"),
            standard_row("ACTUAL_CAPITAL", 999, period="上季度末数"),
            standard_row("CORE_T1_CAPITAL", 1, period="期初数"),
            standard_row("CORE_T2_CAPITAL", 1, period="期初数"),
            standard_row("ANC_T1_CAPITAL", 1, period="期初数"),
            standard_row("ANC_T2_CAPITAL", 1, period="期初数"),
            standard_row("ACTUAL_CAPITAL", 888, period="本年累计数"),
            standard_row("CORE_T1_CAPITAL", 1, period="本年累计数"),
            standard_row("CORE_T2_CAPITAL", 1, period="本年累计数"),
            standard_row("ANC_T1_CAPITAL", 1, period="本年累计数"),
            standard_row("ANC_T2_CAPITAL", 1, period="本年累计数"),
        ]
        result = validate_standard_data(
            pd.DataFrame(rows),
            rules_frame("ACTUAL_CAPITAL_COMPONENTS"),
        )
        business = result[result["rule_id"] == "ACTUAL_CAPITAL_COMPONENTS"]

        self.assertEqual(len(business), 1)
        self.assertEqual(business.iloc[0]["period"], "当前期末")
        self.assertEqual(business.iloc[0]["status"], "通过")

    def test_schema_checks_detect_unit_period_and_unknown_code(self):
        rows = [
            standard_row("ACTUAL_CAPITAL", 100, 单位="元"),
            standard_row("MINIMUM_CAPITAL", 50, period="本季度数"),
            {
                **standard_row("ACTUAL_CAPITAL", 1),
                "指标编码": "UNKNOWN_METRIC",
                "指标名称": "未知指标",
            },
        ]
        result = validate_standard_data(
            pd.DataFrame(rows),
            pd.DataFrame(),
            taxonomy_frame(),
        )
        failed_ids = set(result.loc[result["status"] == "未通过", "rule_id"])
        self.assertIn("SCHEMA_UNIT_MATCH", failed_ids)
        self.assertIn("SCHEMA_ALLOWED_PERIOD", failed_ids)
        self.assertIn("SCHEMA_KNOWN_METRIC", failed_ids)

    def test_system_derived_metrics_are_registered_for_schema_validation(self):
        derived = standard_row("ACTUAL_CAPITAL", 1)
        derived.update({
            "指标编码": "ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS",
            "指标名称": "实际资本/认可资产",
            "数值": 0.1,
            "单位": "倍",
            "数据类型": "比率",
            "来源类型": "系统计算",
            "指标属性": "计算",
        })

        result = validate_standard_data(
            pd.DataFrame([derived]),
            pd.DataFrame(),
            taxonomy_frame(),
        )

        known = result[result["rule_id"] == "SCHEMA_KNOWN_METRIC"].iloc[0]
        self.assertEqual(known["status"], "通过")
        self.assertEqual(known["actual"], 0)

    def test_blank_unit_is_allowed_only_when_metric_dictionary_has_no_unit(self):
        text_metric = standard_row("ACTUAL_CAPITAL", "(20%,35%]")
        text_metric.update({
            "指标编码": "POLICY_SURPLUS_CORE_BAND",
            "指标名称": "保单未来盈余/核心资本分布",
            "单位": "",
            "数据类型": "文本",
            "来源类型": "系统计算",
            "指标属性": "计算",
        })
        amount_metric = standard_row("ACTUAL_CAPITAL", 100, 单位="")

        text_result = validate_standard_data(
            pd.DataFrame([text_metric]),
            pd.DataFrame(),
            taxonomy_frame(),
        )
        amount_result = validate_standard_data(
            pd.DataFrame([amount_metric]),
            pd.DataFrame(),
            taxonomy_frame(),
        )

        text_required = text_result[text_result["rule_id"].eq("SCHEMA_REQUIRED_FIELDS")].iloc[0]
        amount_required = amount_result[amount_result["rule_id"].eq("SCHEMA_REQUIRED_FIELDS")].iloc[0]
        self.assertEqual(text_required["status"], "通过")
        self.assertEqual(amount_required["status"], "未通过")
        self.assertIn("单位缺失1条", amount_required["notes"])

    def test_completeness_reports_each_missing_core_metric(self):
        rows = [
            standard_row(code, 100 if data_type == "金额" else 200)
            for code, _name, _unit, data_type in CORE_METRICS
            if code != "CORE_SOLVENCY_RATIO"
        ]
        result = validate_standard_data(
            pd.DataFrame(rows),
            pd.DataFrame(),
            taxonomy_frame(),
        )
        missing = result[
            result["rule_id"].astype(str).str.startswith("REQUIRED_CORE_METRIC:")
            & result["status"].eq("缺失")
        ]
        self.assertEqual(missing["rule_id"].tolist(), [
            "REQUIRED_CORE_METRIC:CORE_SOLVENCY_RATIO"
        ])
        self.assertEqual(missing.iloc[0]["status"], "缺失")

    def test_completeness_checks_only_current_period_group(self):
        rows = [
            standard_row(code, 100 if data_type == "金额" else 200)
            for code, _name, _unit, data_type in CORE_METRICS
        ]
        rows.extend([
            standard_row("ACTUAL_CAPITAL", 90, period="上季度末数"),
            standard_row("ACTUAL_CAPITAL", 110, period="下季度末预测数"),
        ])
        result = validate_standard_data(
            pd.DataFrame(rows),
            pd.DataFrame(),
            taxonomy_frame(),
        )
        completeness = result[
            result["rule_id"].astype(str).str.startswith("REQUIRED_CORE_METRIC:")
        ]

        self.assertEqual(len(completeness), len(CORE_METRICS))
        self.assertEqual(set(completeness["period"]), {"当前期末"})
        self.assertEqual(set(completeness["status"]), {"通过"})

    def test_conflicting_values_in_same_source_are_rejected(self):
        rows = [
            standard_row("ACTUAL_CAPITAL", 100),
            standard_row("ACTUAL_CAPITAL", 105),
        ]
        result = validate_standard_data(
            pd.DataFrame(rows),
            pd.DataFrame(),
            taxonomy_frame(),
        )
        row = result[result["rule_id"] == "SCHEMA_CONFLICTING_DUPLICATE"].iloc[0]
        self.assertEqual(row["status"], "未通过")


if __name__ == "__main__":
    unittest.main()
