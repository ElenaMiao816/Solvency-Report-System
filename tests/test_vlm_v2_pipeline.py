from __future__ import annotations

import io
import json
import re
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import pandas as pd
import requests

import services.solvency_vlm_v2_pipeline as vlm_pipeline
from services.solvency_dataset_adapter import read_standard_workbook, standard_workbook_bytes
from services.solvency_normalizer import normalize_tables
from services.solvency_pdf_locator import PageMatch
from services.solvency_step3_standardizer import _standardize_step3_period_scopes
from services.solvency_step7_quality import combo_figure
from services.solvency_vlm_v2_pipeline import (
    EXTRACTION_MODE,
    RETRY_MODE,
    build_vlm_v2_target_cards,
    extract_metrics_vlm_v2,
    locate_tables_vlm_v2,
    render_vlm_v2_contact_sheets,
    evaluate_vlm_v2_step3_gate,
    validate_vlm_v2_metrics,
    vlm_v2_to_extracted_tables,
    vlm_v2_workbook_bytes,
)


def image_only_pdf(page_count: int = 2) -> bytes:
    source = fitz.open()
    for page_number in range(1, page_count + 1):
        page = source.new_page(width=420, height=300)
        page.insert_text((40, 80), f"SCANNED PAGE {page_number}", fontsize=24)
        page.insert_text((40, 130), "ACTUAL CAPITAL 100", fontsize=18)

    scanned = fitz.open()
    for source_page in source:
        pixmap = source_page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
        target = scanned.new_page(width=source_page.rect.width, height=source_page.rect.height)
        target.insert_image(target.rect, stream=pixmap.tobytes("png"))
    output = scanned.tobytes()
    source.close()
    scanned.close()
    return output


def minimal_taxonomy() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "指标编码": "ACTUAL_CAPITAL",
            "指标名称": "实际资本",
            "别名": "实际资本合计|实际资本",
            "一级模块": "主要指标",
            "二级模块": "资本",
            "标准单位": "万元",
            "数据类型": "金额",
            "允许期间口径": "期末|本报告期",
        }
    ])


def table_config() -> list[dict]:
    return [{
        "table_id": "SOLVENCY_MAIN",
        "table_name": "偿付能力主要指标",
        "required": True,
        "max_pages": 2,
    }]


class FakeResponse:
    def __init__(self, value: dict, status_code: int = 200):
        self.ok = 200 <= status_code < 300
        self.status_code = status_code
        self.headers = {}
        self.text = json.dumps(value, ensure_ascii=False)
        self._value = value

    def json(self):
        return {
            "choices": [{"message": {"content": json.dumps(self._value, ensure_ascii=False)}}]
        }


class VLMV2PipelineTests(unittest.TestCase):
    def test_policy_surplus_requires_a_visible_source_row_before_zero_is_accepted(self):
        labels = vlm_pipeline.POLICY_SURPLUS_SOURCE_LABELS
        card = vlm_pipeline.VLMV2TargetCard(
            "ACTUAL_CAPITAL", "实际资本", "实际资本明细",
            tuple({"metric_id": code, "name": label, "expected_unit": "万元"}
                  for code, label in labels.items()),
        )
        match = PageMatch("ACTUAL_CAPITAL", "实际资本", [1], 99, "实际资本表")
        metrics = [
            {"metric_id": code, "status": "found" if code == "POLICY_SURPLUS_CORE_T1" else "disclosed_zero",
             "value_raw": "100" if code == "POLICY_SURPLUS_CORE_T1" else "-",
             "unit": "万元", "page": 1, "period_label": "期末数",
             "source_label": label, "evidence_text": f"{label} 期末数 -"}
            for code, label in labels.items()
        ]
        with patch.object(vlm_pipeline, "visible_policy_surplus_codes", return_value=frozenset({"POLICY_SURPLUS_CORE_T1"})), patch.object(
            vlm_pipeline, "_call_vlm_json", return_value={"metrics": metrics},
        ):
            rows, _ = vlm_pipeline._extract_card_records(
                b"pdf", match, card, api_key="key", base_url="https://example.test",
                model="model", timeout=1, request_max_attempts=1, post_func=None,
                page_image_cache={1: "data:image/jpeg;base64,x"}, split_metrics=False,
            )
        states = {row["指标编码"]: row["状态"] for row in rows}
        self.assertEqual(states["POLICY_SURPLUS_CORE_T1"], "found")
        for code in labels.keys() - {"POLICY_SURPLUS_CORE_T1"}:
            self.assertEqual(states[code], "not_disclosed")

        with patch.object(vlm_pipeline, "visible_policy_surplus_codes", return_value=frozenset({
            "POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2",
        })), patch.object(vlm_pipeline, "_call_vlm_json", return_value={"metrics": metrics}):
            rows, _ = vlm_pipeline._extract_card_records(
                b"pdf", match, card, api_key="key", base_url="https://example.test",
                model="model", timeout=1, request_max_attempts=1, post_func=None,
                page_image_cache={1: "data:image/jpeg;base64,x"}, split_metrics=False,
            )
        states = {row["指标编码"]: row["状态"] for row in rows}
        self.assertEqual(states["POLICY_SURPLUS_CORE_T2"], "disclosed_zero")

    def test_known_alias_and_three_year_semantics_are_added(self):
        taxonomy = pd.DataFrame([
            {"指标编码": "FINANCIAL_STATEMENT_NET_ASSETS", "指标名称": "财务报表资产负债差额", "别名": "", "一级模块": "实际资本", "二级模块": "", "标准单位": "万元", "数据类型": "金额", "允许期间口径": "期末"},
            {"指标编码": "INVESTMENT_RETURN", "指标名称": "投资收益率", "别名": "", "一级模块": "经营指标", "二级模块": "", "标准单位": "%", "数据类型": "比例", "允许期间口径": "本季度"},
        ])
        configs = [
            {"table_id": "ACTUAL_CAPITAL", "table_name": "实际资本"},
            {"table_id": "THREE_YEAR_INVESTMENT_RETURN", "table_name": "近三年收益率"},
        ]
        cards = build_vlm_v2_target_cards(configs, taxonomy)
        actual_metric = cards[0].metrics[0]
        three_year_metric = cards[1].metrics[0]
        self.assertIn("净资产", actual_metric["aliases"])
        self.assertEqual(three_year_metric["semantic_key"], "INVESTMENT_RETURN:THREE_YEAR_AVERAGE")

    def test_contact_sheets_support_pdf_without_text_layer(self):
        pdf_bytes = image_only_pdf(2)
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.assertEqual([page.get_text().strip() for page in document], ["", ""])
        document.close()

        sheets = render_vlm_v2_contact_sheets(pdf_bytes, pages_per_sheet=6)
        self.assertEqual(len(sheets), 1)
        self.assertEqual(sheets[0][0], (1, 2))
        self.assertTrue(sheets[0][1].startswith("data:image/jpeg;base64,"))

    def test_target_card_uses_canonical_taxonomy(self):
        cards = build_vlm_v2_target_cards(table_config(), minimal_taxonomy())
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].table_id, "SOLVENCY_MAIN")
        self.assertEqual(cards[0].metrics[0]["metric_id"], "ACTUAL_CAPITAL")
        self.assertTrue(cards[0].metrics[0]["required"])

    def test_registered_capital_card_uses_registry_definition_when_taxonomy_omits_it(self):
        cards = build_vlm_v2_target_cards(
            [{"table_id": "REGISTERED_CAPITAL", "table_name": "注册资本"}],
            minimal_taxonomy(),
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(
            [metric["metric_id"] for metric in cards[0].metrics],
            ["REGISTERED_CAPITAL"],
        )
        self.assertEqual(cards[0].metrics[0]["expected_unit"], "万元")
        self.assertTrue(cards[0].metrics[0]["required"])
        self.assertIn("注册资本金", cards[0].metrics[0]["aliases"])

    def test_registered_capital_extraction_normalizes_value_to_ten_thousand_yuan(self):
        config = {"table_id": "REGISTERED_CAPITAL", "table_name": "注册资本"}
        match = PageMatch(
            table_id="REGISTERED_CAPITAL",
            table_name="注册资本",
            pages=[1],
            score=99,
            evidence="公司基本信息载明注册资本",
            table_config=config,
        )

        def fake_post(*_args, **kwargs):
            prompt = kwargs["json"]["messages"][0]["content"][0]["text"]
            self.assertIn("不得使用实收资本、股本、实际资本、核心资本", prompt)
            return FakeResponse({
                "metrics": [{
                    "metric_id": "REGISTERED_CAPITAL",
                    "status": "found",
                    "value_raw": "12.5",
                    # Regression: the model copied the target standard unit even
                    # though the source evidence explicitly says 亿元.
                    "unit": "万元",
                    "period_label": "",
                    "source_label": "注册资本",
                    "row_header_path": ["公司基本信息", "注册资本"],
                    "column_header_path": [],
                    "page": 1,
                    "evidence_text": "注册资本：人民币12.5亿元",
                    "confidence": 0.99,
                }],
            })

        run = extract_metrics_vlm_v2(
            image_only_pdf(1),
            [match],
            [config],
            minimal_taxonomy(),
            api_key="test",
            base_url="https://example.test/v1",
            model="vision-test",
            post_func=fake_post,
            auto_retry=False,
            max_workers=1,
        )

        row = run.records.iloc[0]
        self.assertEqual(row["指标编码"], "REGISTERED_CAPITAL")
        self.assertEqual(row["目标表ID"], "REGISTERED_CAPITAL")
        self.assertEqual(row["单位"], "亿元")
        self.assertEqual(row["标准单位"], "万元")
        self.assertEqual(row["标准数值"], 125000.0)
        self.assertIn("模型原单位“万元”校正为“亿元”", row["证据原文"])
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_workbook_embeds_report_metadata_and_source_pdf(self):
        run = vlm_pipeline.VLMV2ExtractionRun(
            pd.DataFrame(columns=vlm_pipeline.METRIC_COLUMNS),
            pd.DataFrame(columns=vlm_pipeline.VALIDATION_COLUMNS),
            (),
            0,
            0,
        )
        workbook = vlm_v2_workbook_bytes(
            None,
            run,
            report_metadata={
                "公司": "长生人寿",
                "报告年度": 2026,
                "报告季度": "Q1",
                "披露日期": "2026-04-29",
            },
            source_filename="长生人寿2026Q1偿付能力季度报告摘要.pdf",
        )

        excel = pd.ExcelFile(io.BytesIO(workbook))
        self.assertEqual(excel.sheet_names[0], "报告元信息")
        metadata = pd.read_excel(excel, sheet_name="报告元信息").set_index("字段")["值"]
        self.assertEqual(metadata["公司"], "长生人寿")
        self.assertEqual(metadata["报告期"], "2026Q1")
        self.assertEqual(metadata["来源文件"], "长生人寿2026Q1偿付能力季度报告摘要.pdf")

    def test_actual_capital_prompt_uses_sparse_disclosures(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": code,
                "指标名称": name,
                "别名": name,
                "一级模块": "实际资本",
                "二级模块": "资本",
                "标准单位": "万元",
                "数据类型": "金额",
                "允许期间口径": "期末|期初",
            }
            for code, name in (
                ("CORE_T1_CAPITAL", "核心一级资本"),
                ("ANC_T1_CAPITAL", "附属一级资本"),
            )
        ])
        card = build_vlm_v2_target_cards(
            [{"table_id": "ACTUAL_CAPITAL", "table_name": "S02-实际资本明细表"}],
            taxonomy,
        )[0]
        prompt = vlm_pipeline._metric_prompt(card, [31, 32])

        self.assertIn('"response_mode": "sparse_disclosures"', prompt)
        self.assertIn("未披露指标不要逐条返回", prompt)
        self.assertIn("系统会确定性补为not_disclosed", prompt)
        self.assertNotIn('"allowed_period_labels"', prompt)
        self.assertNotIn("每个target_metrics指标都返回一条", prompt)

    def test_operating_prompt_requires_current_and_cumulative_columns(self):
        taxonomy = pd.DataFrame([{
            "指标编码": "NET_PROFIT", "指标名称": "净利润", "别名": "净利润",
            "一级模块": "经营指标", "二级模块": "经营成果", "标准单位": "万元",
            "数据类型": "金额", "允许期间口径": "本季度（末）数|本年累计数",
        }])
        card = build_vlm_v2_target_cards(
            [{"table_id": "OPERATING_METRICS", "table_name": "主要经营指标"}],
            taxonomy,
        )[0]

        prompt = vlm_pipeline._metric_prompt(card, [11])

        self.assertIn("经营指标必须同时提取", prompt)
        self.assertIn("即使第一季度两列数值完全相同", prompt)
        self.assertIn("同一指标两列均有披露时返回两条记录", prompt)
        self.assertNotIn("严禁选择任何累计列", prompt)

    def test_operating_keeps_current_quarter_and_cumulative_candidates(self):
        taxonomy = pd.DataFrame([{
            "指标编码": "NET_PROFIT", "指标名称": "净利润", "别名": "净利润",
            "一级模块": "经营指标", "二级模块": "经营成果", "标准单位": "万元",
            "数据类型": "金额", "允许期间口径": "本季度（末）数|本年累计数",
        }])
        config = {"table_id": "OPERATING_METRICS", "table_name": "主要经营指标"}
        match = PageMatch(
            table_id="OPERATING_METRICS", table_name="主要经营指标", pages=[1],
            score=99, evidence="主要经营指标", table_config=config,
        )

        def fake_post(_url, *, headers, json, timeout):
            return FakeResponse({"metrics": [
                {
                    "metric_id": "NET_PROFIT", "status": "found", "value_raw": "999",
                    "unit": "万元", "period_label": "本年度累计数", "source_label": "净利润",
                    "column_header_path": ["本年度累计数"], "page": 1,
                    "evidence_text": "净利润 本年度累计数 999", "confidence": .99,
                },
                {
                    "metric_id": "NET_PROFIT", "status": "found", "value_raw": "88",
                    "unit": "万元", "period_label": "本季度数", "source_label": "净利润",
                    "column_header_path": ["本季度数"], "page": 1,
                    "evidence_text": "净利润 本季度数 88", "confidence": .80,
                },
            ]})

        run = extract_metrics_vlm_v2(
            image_only_pdf(1), [match], [config], taxonomy,
            api_key="x", base_url="https://example.test/v1", model="vision-test",
            post_func=fake_post, auto_retry=False, max_workers=1,
        )

        rows = run.records.sort_values("期间口径")
        self.assertEqual(set(rows["原始值"]), {"88", "999"})
        self.assertEqual(set(rows["期间口径"]), {"本季度数", "本年度累计数"})
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_operating_periods_survive_step3_step5_and_feed_step7_cumulative_chart(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": code, "指标名称": name, "别名": name,
                "一级模块": "经营指标", "二级模块": "投资收益",
                "标准单位": "%", "数据类型": "比例",
                "允许期间口径": "本季度（末）数|本年累计数",
            }
            for code, name in (
                ("INVESTMENT_RETURN", "投资收益率"),
                ("COMPREHENSIVE_INVESTMENT_RETURN", "综合投资收益率"),
            )
        ])
        config = {"table_id": "OPERATING_METRICS", "table_name": "主要经营指标"}
        match = PageMatch(
            table_id="OPERATING_METRICS", table_name="主要经营指标", pages=[1],
            score=99, evidence="主要经营指标", table_config=config,
        )

        def fake_post(_url, *, headers, json, timeout):
            metrics = []
            for code, current, cumulative in (
                ("INVESTMENT_RETURN", "1.2%", "4.5%"),
                ("COMPREHENSIVE_INVESTMENT_RETURN", "1.4%", "5.1%"),
            ):
                for period, value in (("本季度数", current), ("本年度累计数", cumulative)):
                    metrics.append({
                        "metric_id": code, "status": "found", "value_raw": value,
                        "unit": "%", "period_label": period, "source_label": code,
                        "column_header_path": [period], "page": 1,
                        "evidence_text": f"{code} {period} {value}", "confidence": .99,
                    })
            return FakeResponse({"metrics": metrics})

        run = extract_metrics_vlm_v2(
            image_only_pdf(1), [match], [config], taxonomy,
            api_key="x", base_url="https://example.test/v1", model="vision-test",
            post_func=fake_post, auto_retry=False, max_workers=1,
        )
        tables = vlm_v2_to_extracted_tables(run)
        self.assertEqual(tables[0].rows[0], ["指标名称", "本季度数", "本年度累计数"])

        metadata = {
            "公司": "测试人寿", "报告年度": 2026, "报告季度": "Q2",
            "报告期": "2026Q2", "来源文件": "fixture.pdf",
        }
        normalized = normalize_tables(
            tables, taxonomy, metadata, company_type="寿险",
            report_profile_id="LIFE_SOLVENCY",
        )
        normalized = _standardize_step3_period_scopes(normalized, metadata)
        self.assertEqual(set(normalized["期间口径"]), {"本季度数", "本年累计数"})

        restored = read_standard_workbook(
            standard_workbook_bytes(normalized), "step3_fixture.xlsx",
        )
        self.assertEqual(set(restored["期间口径"]), {"本季度数", "本年累计数"})
        figure, missing, invalid = combo_figure(
            restored, ["2026Q2"], "INVESTMENT_RETURN",
            "COMPREHENSIVE_INVESTMENT_RETURN", scope="cumulative",
        )
        self.assertEqual(list(figure.data[0].y), [4.5])
        self.assertEqual(list(figure.data[1].y), [5.1])
        self.assertEqual(missing, [])
        self.assertEqual(invalid, [])

    @unittest.skipUnless(
        Path(r"F:\CROSS\V2\人保养老2026Q1偿付能力季度报告摘要.pdf").exists(),
        "人保养老回归样本在当前环境不可用",
    )
    def test_renbao_pension_operating_current_and_cumulative_periods_are_accepted(self):
        pdf_path = Path(r"F:\CROSS\V2\人保养老2026Q1偿付能力季度报告摘要.pdf")
        taxonomy = pd.DataFrame([
            {
                "指标编码": code, "指标名称": name, "别名": name,
                "一级模块": "经营指标", "二级模块": "经营成果",
                "标准单位": unit, "数据类型": data_type,
                "允许期间口径": "本季度（末）数|本年累计数",
            }
            for code, name, unit, data_type in (
                ("NET_PROFIT", "净利润", "万元", "金额"),
                ("INVESTMENT_RETURN", "投资收益率", "%", "比例"),
            )
        ])
        config = {"table_id": "OPERATING_METRICS", "table_name": "主要经营指标"}
        match = PageMatch(
            table_id="OPERATING_METRICS", table_name="主要经营指标", pages=[11],
            score=99, evidence="主要经营指标", table_config=config,
        )
        prompts = []

        def fake_post(_url, *, headers, json, timeout):
            prompts.append(json["messages"][0]["content"][0]["text"])
            metrics = []
            for period in ("本季度数", "本年度累计数"):
                metrics.extend([
                    {
                        "metric_id": "NET_PROFIT", "status": "found", "value_raw": "9,750.41",
                        "unit": "万元", "period_label": period, "source_label": "（二）净利润",
                        "column_header_path": [period], "page": 11,
                        "evidence_text": f"（二）净利润 {period} 9,750.41", "confidence": .99,
                    },
                    {
                        "metric_id": "INVESTMENT_RETURN", "status": "found", "value_raw": "0.92%",
                        "unit": "%", "period_label": period, "source_label": "（九）投资收益率",
                        "column_header_path": [period], "page": 11,
                        "evidence_text": f"（九）投资收益率 {period} 0.92%", "confidence": .99,
                    },
                ])
            return FakeResponse({"metrics": metrics})

        run = extract_metrics_vlm_v2(
            pdf_path.read_bytes(), [match], [config], taxonomy,
            api_key="x", base_url="https://example.test/v1", model="vision-test",
            post_func=fake_post, auto_retry=True, max_workers=1,
            retry_max_workers=1, request_max_attempts=1,
        )

        self.assertEqual(run.model_calls, 1)
        self.assertEqual(run.retry_calls, 0)
        self.assertEqual(set(run.records["期间口径"]), {"本季度数", "本年度累计数"})
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_locator_reads_scanned_pages_as_images(self):
        observed = {}

        def fake_post(_url, *, headers, json, timeout):
            observed["content"] = json["messages"][0]["content"]
            return FakeResponse({
                "targets": [{
                    "table_id": "SOLVENCY_MAIN",
                    "found": True,
                    "pages": [2],
                    "source_title": "Solvency summary",
                    "evidence": "actual capital and solvency ratios",
                    "confidence": 0.96,
                }]
            })

        run = locate_tables_vlm_v2(
            image_only_pdf(2),
            table_config(),
            minimal_taxonomy(),
            api_key="test",
            base_url="https://example.test/v1",
            model="vision-test",
            post_func=fake_post,
        )
        self.assertEqual(run.matches[0].pages, [2])
        self.assertFalse(run.matches[0].review_required)
        self.assertEqual(run.model_calls, 1)
        self.assertTrue(any(item.get("type") == "image_url" for item in observed["content"]))

    def test_locator_sends_one_sheet_per_call_with_two_workers(self):
        lock = threading.Lock()
        active_calls = 0
        max_active_calls = 0
        image_counts = []

        def fake_post(_url, *, headers, json, timeout):
            nonlocal active_calls, max_active_calls
            with lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            image_counts.append(sum(
                item.get("type") == "image_url"
                for item in json["messages"][0]["content"]
            ))
            time.sleep(0.05)
            with lock:
                active_calls -= 1
            return FakeResponse({"targets": []})

        run = locate_tables_vlm_v2(
            image_only_pdf(13),
            table_config(),
            minimal_taxonomy(),
            api_key="test",
            base_url="https://example.test/v1",
            model="vision-test",
            post_func=fake_post,
            request_max_attempts=1,
            sheets_per_call=4,
            max_workers=5,
        )

        self.assertEqual(run.model_calls, 3)
        self.assertEqual(sorted(image_counts), [1, 1, 1])
        self.assertEqual(max_active_calls, 2)

    def test_locator_retries_only_timeout_batches_and_preserves_successes(self):
        attempts = {1: 0, 7: 0, 13: 0}
        captured_payloads = []
        lock = threading.Lock()

        def fake_post(_url, *, headers, json, timeout):
            prompt = json["messages"][0]["content"][0]["text"]
            match = re.search(r"当前批次包含物理页：\[([^\]]+)\]", prompt)
            first_page = int(match.group(1).split(",")[0].strip())
            with lock:
                attempts[first_page] += 1
                current_attempt = attempts[first_page]
                captured_payloads.append(json)
            if first_page == 1 and current_attempt == 1:
                raise requests.ReadTimeout("batch 1 timed out")
            if first_page == 13:
                raise RuntimeError("non-timeout provider error")
            hit_page = 2 if first_page == 1 else 7
            return FakeResponse({"targets": [{
                "table_id": "SOLVENCY_MAIN",
                "found": True,
                "pages": [hit_page],
                "source_title": "偿付能力主要指标",
                "evidence": "实际资本",
                "confidence": .95,
            }]})

        run = locate_tables_vlm_v2(
            image_only_pdf(13),
            table_config(),
            minimal_taxonomy(),
            api_key="test",
            base_url="https://api.moonshot.cn/v1",
            model="kimi-k2.6",
            post_func=fake_post,
            request_max_attempts=2,
            max_workers=2,
        )

        self.assertEqual(attempts, {1: 2, 7: 1, 13: 1})
        self.assertEqual(run.model_calls, 4)
        self.assertEqual(run.matches[0].pages, [2, 7])
        self.assertTrue(all(
            payload["thinking"] == {"type": "disabled"}
            for payload in captured_payloads
        ))
        latest = (
            run.diagnostics.sort_values("调用序号")
            .groupby("批次序号", sort=False)
            .tail(1)
            .set_index("批次序号")
        )
        self.assertEqual(latest.loc[1, "状态"], "成功")
        self.assertEqual(latest.loc[2, "状态"], "成功")
        self.assertEqual(latest.loc[3, "状态"], "失败")

    def test_only_failed_metric_is_retried_and_uses_shared_page_cache(self):
        responses = [
            {
                "metrics": [{
                    "metric_id": "ACTUAL_CAPITAL",
                    "status": "found",
                    "value_raw": "1",
                    "unit": "亿元",
                    "period_label": "期末",
                    "source_label": "实际资本",
                    "row_header_path": ["实际资本"],
                    "column_header_path": ["期末数"],
                    "page": 1,
                    "evidence_text": "",
                    "confidence": 0.82,
                }]
            },
            {
                "metrics": [{
                    "metric_id": "ACTUAL_CAPITAL",
                    "status": "found",
                    "value_raw": "1",
                    "unit": "亿元",
                    "period_label": "期末",
                    "source_label": "实际资本",
                    "row_header_path": ["实际资本"],
                    "column_header_path": ["期末数"],
                    "page": 1,
                    "evidence_text": "实际资本 100 万元",
                    "confidence": 0.97,
                }]
            },
        ]
        prompts = []

        def fake_post(_url, *, headers, json, timeout):
            prompts.append(json["messages"][0]["content"][0]["text"])
            return FakeResponse(responses.pop(0))

        match = PageMatch(
            table_id="SOLVENCY_MAIN",
            table_name="偿付能力主要指标",
            pages=[1],
            score=96.0,
            evidence="visual evidence",
            table_config=table_config()[0],
        )
        with patch.object(
            vlm_pipeline,
            "_render_page_images",
            wraps=vlm_pipeline._render_page_images,
        ) as render_pages:
            run = extract_metrics_vlm_v2(
                image_only_pdf(1),
                [match],
                table_config(),
                minimal_taxonomy(),
                api_key="test",
                base_url="https://example.test/v1",
                model="vision-test",
                post_func=fake_post,
                auto_retry=True,
                max_workers=1,
            )

        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.retry_calls, 1)
        self.assertEqual(run.records.iloc[0]["提取轮次"], RETRY_MODE)
        self.assertNotEqual(run.records.iloc[0]["提取轮次"], EXTRACTION_MODE)
        self.assertEqual(run.records.iloc[0]["标准单位"], "万元")
        self.assertEqual(run.records.iloc[0]["标准数值"], 10000.0)
        self.assertFalse((run.validations["状态"] == "失败").any())
        self.assertIn("只重新核对这些失败指标", prompts[1])
        self.assertEqual(render_pages.call_count, 1)

    def test_large_page_set_is_batched_and_reports_table_progress(self):
        image_counts = []
        events = []

        def fake_post(_url, *, headers, json, timeout):
            content = json["messages"][0]["content"]
            image_counts.append(sum(item.get("type") == "image_url" for item in content))
            return FakeResponse({"metrics": [{
                "metric_id": "ACTUAL_CAPITAL",
                "status": "found",
                "value_raw": "100",
                "unit": "万元",
                "period_label": "期末",
                "source_label": "实际资本",
                "page": 1,
                "evidence_text": "实际资本100万元",
                "confidence": .95,
            }]})

        match = PageMatch(
            table_id="SOLVENCY_MAIN",
            table_name="偿付能力主要指标",
            pages=[1, 2, 3, 4, 5],
            score=96.0,
            evidence="visual evidence",
            table_config=table_config()[0],
        )
        run = extract_metrics_vlm_v2(
            image_only_pdf(5),
            [match],
            table_config(),
            minimal_taxonomy(),
            api_key="test",
            base_url="https://example.test/v1",
            model="vision-test",
            post_func=fake_post,
            auto_retry=False,
            max_workers=1,
            max_pages_per_request=2,
            progress_callback=events.append,
        )

        self.assertEqual(run.model_calls, 3)
        self.assertEqual(image_counts, [2, 2, 1])
        completed = [event for event in events if event["phase"] == "initial_completed"]
        self.assertEqual(completed[0]["completed_tables"], 1)
        self.assertEqual(completed[0]["total_tables"], 1)
        self.assertEqual(events[-1]["phase"], "completed")

    def test_timeout_table_is_retried_after_other_tables_and_successes_are_kept(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": "ACTUAL_CAPITAL", "指标名称": "实际资本", "别名": "实际资本",
                "一级模块": "实际资本", "二级模块": "", "标准单位": "万元",
                "数据类型": "金额", "允许期间口径": "期末",
            },
            {
                "指标编码": "INSURANCE_REVENUE", "指标名称": "保险业务收入", "别名": "保险业务收入",
                "一级模块": "经营指标", "二级模块": "", "标准单位": "万元",
                "数据类型": "金额", "允许期间口径": "本季度",
            },
        ])
        configs = [
            {"table_id": "ACTUAL_CAPITAL", "table_name": "S02-实际资本明细表"},
            {"table_id": "OPERATING_METRICS", "table_name": "主要经营指标"},
        ]
        matches = [
            PageMatch(table_id="ACTUAL_CAPITAL", table_name=configs[0]["table_name"], pages=[1], score=90, evidence="", table_config=configs[0]),
            PageMatch(table_id="OPERATING_METRICS", table_name=configs[1]["table_name"], pages=[2], score=90, evidence="", table_config=configs[1]),
        ]
        attempts = {"ACTUAL_CAPITAL": 0, "OPERATING_METRICS": 0}
        events = []
        lock = threading.Lock()

        def fake_post(_url, *, headers, json, timeout):
            prompt = json["messages"][0]["content"][0]["text"]
            table_id = (
                "ACTUAL_CAPITAL"
                if '"table_id": "ACTUAL_CAPITAL"' in prompt
                else "OPERATING_METRICS"
            )
            with lock:
                attempts[table_id] += 1
                attempt = attempts[table_id]
            if table_id == "ACTUAL_CAPITAL" and attempt == 1:
                raise requests.ReadTimeout("S02 timed out")
            metric_id = "ACTUAL_CAPITAL" if table_id == "ACTUAL_CAPITAL" else "INSURANCE_REVENUE"
            return FakeResponse({"metrics": [{
                "metric_id": metric_id,
                "status": "found",
                "value_raw": "100",
                "unit": "万元",
                "period_label": "期末" if table_id == "ACTUAL_CAPITAL" else "本季度",
                "source_label": metric_id,
                "page": 1 if table_id == "ACTUAL_CAPITAL" else 2,
                "evidence_text": f"{metric_id} 100万元",
                "confidence": .95,
            }]})

        run = extract_metrics_vlm_v2(
            image_only_pdf(2), matches, configs, taxonomy,
            api_key="x", base_url="https://example.test/v1", model="vision-test",
            post_func=fake_post, auto_retry=False, max_workers=2,
            retry_max_workers=1, request_max_attempts=2,
            progress_callback=events.append,
        )

        self.assertEqual(attempts, {"ACTUAL_CAPITAL": 2, "OPERATING_METRICS": 1})
        self.assertEqual(set(run.records["指标编码"]), {"ACTUAL_CAPITAL", "INSURANCE_REVENUE"})
        self.assertEqual(run.retry_calls, 1)
        self.assertIn("initial_failed", [event["phase"] for event in events])
        self.assertIn("timeout_retry_completed", [event["phase"] for event in events])

    def test_final_table_timeout_returns_partial_results_instead_of_raising(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": "ACTUAL_CAPITAL", "指标名称": "实际资本", "别名": "实际资本",
                "一级模块": "实际资本", "二级模块": "", "标准单位": "万元",
                "数据类型": "金额", "允许期间口径": "期末",
            },
            {
                "指标编码": "INSURANCE_REVENUE", "指标名称": "保险业务收入", "别名": "保险业务收入",
                "一级模块": "经营指标", "二级模块": "", "标准单位": "万元",
                "数据类型": "金额", "允许期间口径": "本季度",
            },
        ])
        configs = [
            {"table_id": "ACTUAL_CAPITAL", "table_name": "S02-实际资本明细表"},
            {"table_id": "OPERATING_METRICS", "table_name": "主要经营指标"},
        ]
        matches = [
            PageMatch(table_id="ACTUAL_CAPITAL", table_name=configs[0]["table_name"], pages=[1], score=90, evidence="", table_config=configs[0]),
            PageMatch(table_id="OPERATING_METRICS", table_name=configs[1]["table_name"], pages=[2], score=90, evidence="", table_config=configs[1]),
        ]

        def fake_post(_url, *, headers, json, timeout):
            prompt = json["messages"][0]["content"][0]["text"]
            if '"table_id": "ACTUAL_CAPITAL"' in prompt:
                raise requests.ReadTimeout("S02 still timed out")
            return FakeResponse({"metrics": [{
                "metric_id": "INSURANCE_REVENUE", "status": "found",
                "value_raw": "100", "unit": "万元", "period_label": "本季度",
                "source_label": "保险业务收入", "page": 2,
                "evidence_text": "保险业务收入100万元", "confidence": .95,
            }]})

        run = extract_metrics_vlm_v2(
            image_only_pdf(2), matches, configs, taxonomy,
            api_key="x", base_url="https://example.test/v1", model="vision-test",
            post_func=fake_post, auto_retry=True, max_workers=2,
            retry_max_workers=1, request_max_attempts=2,
        )

        self.assertEqual(run.records["指标编码"].tolist(), ["INSURANCE_REVENUE"])
        actual_failure = run.validations[
            run.validations["校验ID"] == "ACTUAL_CAPITAL:ACTUAL_CAPITAL:REQUIRED"
        ]
        self.assertEqual(actual_failure.iloc[0]["状态"], "失败")
        self.assertTrue(any("已保留其他成功表" in log for log in run.logs))

    def test_failed_tables_retry_with_bounded_parallelism(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": "ACTUAL_CAPITAL", "指标名称": "实际资本", "别名": "实际资本",
                "一级模块": "主要指标", "二级模块": "", "标准单位": "万元",
                "数据类型": "金额", "允许期间口径": "期末",
            },
            {
                "指标编码": "INSURANCE_REVENUE", "指标名称": "保险业务收入", "别名": "保险业务收入",
                "一级模块": "经营指标", "二级模块": "", "标准单位": "万元",
                "数据类型": "金额", "允许期间口径": "本季度",
            },
        ])
        configs = [
            {"table_id": "SOLVENCY_MAIN", "table_name": "主要指标"},
            {"table_id": "OPERATING_METRICS", "table_name": "经营指标"},
        ]
        matches = [
            PageMatch(table_id="SOLVENCY_MAIN", table_name="主要指标", pages=[1], score=90, evidence="", table_config=configs[0]),
            PageMatch(table_id="OPERATING_METRICS", table_name="经营指标", pages=[2], score=90, evidence="", table_config=configs[1]),
        ]
        lock = threading.Lock()
        active_retries = 0
        max_active_retries = 0
        events = []

        def fake_post(_url, *, headers, json, timeout):
            nonlocal active_retries, max_active_retries
            prompt = json["messages"][0]["content"][0]["text"]
            metric_id = "ACTUAL_CAPITAL" if '"table_id": "SOLVENCY_MAIN"' in prompt else "INSURANCE_REVENUE"
            if "上一轮未通过校验" not in prompt:
                return FakeResponse({"metrics": [{"metric_id": metric_id, "status": "not_disclosed"}]})
            with lock:
                active_retries += 1
                max_active_retries = max(max_active_retries, active_retries)
            time.sleep(0.08)
            with lock:
                active_retries -= 1
            return FakeResponse({"metrics": [{
                "metric_id": metric_id,
                "status": "found",
                "value_raw": "100",
                "unit": "万元",
                "period_label": "期末" if metric_id == "ACTUAL_CAPITAL" else "本季度",
                "source_label": metric_id,
                "page": 1 if metric_id == "ACTUAL_CAPITAL" else 2,
                "evidence_text": f"{metric_id} 100万元",
                "confidence": .95,
            }]})

        run = extract_metrics_vlm_v2(
            image_only_pdf(2),
            matches,
            configs,
            taxonomy,
            api_key="test",
            base_url="https://example.test/v1",
            model="vision-test",
            post_func=fake_post,
            max_workers=1,
            retry_max_workers=2,
            progress_callback=events.append,
        )

        self.assertEqual(run.model_calls, 4)
        self.assertEqual(run.retry_calls, 2)
        self.assertEqual(max_active_retries, 2)
        self.assertEqual(
            len([event for event in events if event["phase"] == "retry_completed"]),
            2,
        )

    def test_count_unit_and_dash_are_normalized_and_gate_passes(self):
        taxonomy = pd.DataFrame([
            {"指标编码": "AGENT_COUNT", "指标名称": "代理人数量", "别名": "代理人", "一级模块": "经营指标", "二级模块": "", "标准单位": "人", "数据类型": "数量", "允许期间口径": "本季度末"},
            {"指标编码": "ADDITIONAL_CAPITAL", "指标名称": "附加资本", "别名": "附加资本", "一级模块": "最低资本", "二级模块": "", "标准单位": "万元", "数据类型": "金额", "允许期间口径": "期末"},
        ])
        configs = [
            {"table_id": "OPERATING_METRICS", "table_name": "经营指标"},
            {"table_id": "MINIMUM_CAPITAL", "table_name": "最低资本"},
        ]
        responses = iter([
            {"metrics": [{"metric_id": "AGENT_COUNT", "status": "found", "value_raw": "19", "unit": "万人", "period_label": "本季度末", "source_label": "代理人", "page": 1, "evidence_text": "代理人19万人", "confidence": .9}]},
            {"metrics": [{"metric_id": "ADDITIONAL_CAPITAL", "status": "found", "value_raw": "-", "unit": "万元", "period_label": "期末", "source_label": "附加资本", "page": 2, "evidence_text": "附加资本 -", "confidence": .9}]},
        ])
        thinking_values = []
        def fake_post(_url, *, headers, json, timeout):
            thinking_values.append(json.get("thinking"))
            return FakeResponse(next(responses))
        matches = [
            PageMatch(table_id="OPERATING_METRICS", table_name="经营指标", pages=[1], score=90, evidence="", table_config=configs[0]),
            PageMatch(table_id="MINIMUM_CAPITAL", table_name="最低资本", pages=[2], score=90, evidence="", table_config=configs[1]),
        ]
        run = extract_metrics_vlm_v2(image_only_pdf(2), matches, configs, taxonomy, api_key="x", base_url="https://api.moonshot.cn/v1", model="kimi-k2.6", post_func=fake_post, auto_retry=False, max_workers=1)
        agent = run.records[run.records["指标编码"] == "AGENT_COUNT"].iloc[0]
        additional = run.records[run.records["指标编码"] == "ADDITIONAL_CAPITAL"].iloc[0]
        self.assertEqual(agent["标准数值"], 190000.0)
        self.assertEqual(additional["状态"], "disclosed_zero")
        self.assertEqual(additional["标准数值"], 0.0)
        self.assertEqual(thinking_values, [
            {"type": "disabled"},
            {"type": "disabled"},
        ])

    def test_other_dash_and_missing_metric_have_distinct_statuses(self):
        taxonomy = pd.DataFrame([
            {"指标编码": "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT", "指标名称": "投资性房地产公允价值调整", "别名": "", "一级模块": "实际资本", "二级模块": "", "标准单位": "万元", "数据类型": "金额", "允许期间口径": "期末"},
            {"指标编码": "FINANCIAL_STATEMENT_NET_ASSETS", "指标名称": "财务报表净资产", "别名": "", "一级模块": "实际资本", "二级模块": "", "标准单位": "万元", "数据类型": "金额", "允许期间口径": "期末"},
        ])
        configs = [{"table_id": "ACTUAL_CAPITAL", "table_name": "实际资本"}]
        def fake_post(_url, *, headers, json, timeout):
            return FakeResponse({"metrics": [{
                "metric_id": "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT",
                "status": "found", "value_raw": "—", "unit": "万元", "period_label": "期末",
                "source_label": "投资性房地产公允价值调整", "page": 1,
                "evidence_text": "投资性房地产公允价值调整 —", "confidence": .9,
            }]})
        match = PageMatch(table_id="ACTUAL_CAPITAL", table_name="实际资本", pages=[1], score=90, evidence="", table_config=configs[0])
        run = extract_metrics_vlm_v2(image_only_pdf(1), [match], configs, taxonomy, api_key="x", base_url="https://x/v1", model="v", post_func=fake_post, auto_retry=False, max_workers=1)
        statuses = run.records.set_index("指标编码")["状态"].to_dict()
        self.assertEqual(statuses["CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT"], "disclosed_zero")
        dash = run.records.set_index("指标编码").loc["CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT"]
        self.assertEqual(dash["数值"], 0)
        self.assertEqual(dash["标准数值"], 0)
        self.assertEqual(dash["原始值"], "—")
        self.assertEqual(statuses["FINANCIAL_STATEMENT_NET_ASSETS"], "not_disclosed")

    def test_step3_gate_blocks_failed_validation_and_adapter_requires_pass(self):
        records = pd.DataFrame([{
            "目标表ID": "SOLVENCY_MAIN", "目标表名称": "主要指标", "指标编码": "ACTUAL_CAPITAL",
            "指标名称": "实际资本", "指标语义键": "ACTUAL_CAPITAL", "状态": "found", "原始值": "100",
            "数值": 100.0, "单位": "万元", "标准数值": 100.0, "标准单位": "万元", "期间口径": "上季度末",
            "原始标签": "实际资本", "行表头路径": "实际资本", "列表头路径": "上季度末", "物理页码": 1,
            "证据原文": "实际资本100", "置信度": .9, "提取轮次": EXTRACTION_MODE,
        }])
        from services.solvency_vlm_v2_pipeline import VLMV2ExtractionRun
        run = VLMV2ExtractionRun(records, pd.DataFrame(columns=["状态"]), (), 1, 0)
        gate = evaluate_vlm_v2_step3_gate(run)
        self.assertFalse(gate.passed)
        with self.assertRaises(ValueError):
            vlm_v2_to_extracted_tables(run)

    def test_financial_net_assets_is_conditionally_required(self):
        taxonomy = pd.DataFrame([
            {"指标编码": code, "指标名称": name, "别名": name, "一级模块": "实际资本", "二级模块": "", "标准单位": "万元", "数据类型": "金额", "允许期间口径": "期末"}
            for code, name in (
                ("FINANCIAL_STATEMENT_ASSETS", "财务报表资产"),
                ("FINANCIAL_STATEMENT_LIABILITIES", "财务报表负债"),
                ("FINANCIAL_STATEMENT_NET_ASSETS", "财务报表净资产"),
            )
        ])
        cards = build_vlm_v2_target_cards([{"table_id": "ACTUAL_CAPITAL", "table_name": "实际资本"}], taxonomy)
        rows = []
        for code, value in (("FINANCIAL_STATEMENT_ASSETS", 120.0), ("FINANCIAL_STATEMENT_LIABILITIES", 20.0)):
            rows.append({
                "目标表ID": "ACTUAL_CAPITAL", "目标表名称": "实际资本", "指标编码": code,
                "指标名称": code, "指标语义键": code, "状态": "found", "原始值": str(value),
                "数值": value, "单位": "万元", "标准数值": value, "标准单位": "万元", "期间口径": "期末",
                "原始标签": code, "行表头路径": code, "列表头路径": "期末", "物理页码": 1,
                "证据原文": f"{code} {value}", "置信度": .9, "提取轮次": EXTRACTION_MODE,
            })
        checks = validate_vlm_v2_metrics(pd.DataFrame(rows), cards)
        failure = checks[checks["校验ID"] == "ACTUAL_CAPITAL:FINANCIAL_STATEMENT_NET_ASSETS:CONDITIONAL"]
        self.assertEqual(failure.iloc[0]["状态"], "失败")

    @unittest.skipUnless(
        Path(r"F:\CROSS\V2\平安养老2026Q1偿付能力季度报告摘要.pdf").exists(),
        "平安养老回归样本在当前环境不可用",
    )
    def test_pingan_pension_s02_pages_31_32_regression(self):
        pdf_path = Path(r"F:\CROSS\V2\平安养老2026Q1偿付能力季度报告摘要.pdf")
        taxonomy = pd.DataFrame([
            {
                "指标编码": code, "指标名称": name, "别名": name,
                "一级模块": "实际资本", "二级模块": "资本", "标准单位": "万元",
                "数据类型": "金额", "允许期间口径": "期末|期初",
            }
            for code, name in (
                ("ACTUAL_CAPITAL", "实际资本"),
                ("CORE_T1_CAPITAL", "核心一级资本"),
                ("CORE_T2_CAPITAL", "核心二级资本"),
                ("ANC_T1_CAPITAL", "附属一级资本"),
                ("ANC_T2_CAPITAL", "附属二级资本"),
            )
        ])
        config = {"table_id": "ACTUAL_CAPITAL", "table_name": "S02-实际资本明细表"}
        match = PageMatch(
            table_id="ACTUAL_CAPITAL", table_name=config["table_name"],
            pages=[31, 32], score=99, evidence="S02-实际资本表", table_config=config,
        )
        observed = {}

        def fake_post(_url, *, headers, json, timeout):
            content = json["messages"][0]["content"]
            observed["image_count"] = sum(item.get("type") == "image_url" for item in content)
            observed["prompt"] = content[0]["text"]
            return FakeResponse({"metrics": [
                {"metric_id": "CORE_T1_CAPITAL", "status": "found", "value_raw": "16,544,189,514.57", "unit": "元", "period_label": "期末数", "source_label": "核心一级资本", "page": 31, "evidence_text": "核心一级资本 16,544,189,514.57", "confidence": .99},
                {"metric_id": "CORE_T2_CAPITAL", "status": "disclosed_na", "value_raw": "", "unit": "元", "period_label": "期末数", "source_label": "核心二级资本", "page": 31, "evidence_text": "核心二级资本行期末数为空", "confidence": .95},
                {"metric_id": "ANC_T1_CAPITAL", "status": "found", "value_raw": "6,012,539,327.35", "unit": "元", "period_label": "期末数", "source_label": "附属一级资本", "page": 31, "evidence_text": "附属一级资本 6,012,539,327.35", "confidence": .99},
                {"metric_id": "ANC_T2_CAPITAL", "status": "disclosed_na", "value_raw": "", "unit": "元", "period_label": "期末数", "source_label": "附属二级资本", "page": 32, "evidence_text": "附属二级资本行期末数为空", "confidence": .95},
                {"metric_id": "ACTUAL_CAPITAL", "status": "found", "value_raw": "22,556,728,841.92", "unit": "元", "period_label": "期末数", "source_label": "实际资本合计", "page": 32, "evidence_text": "实际资本合计 22,556,728,841.92", "confidence": .99},
            ]})

        run = extract_metrics_vlm_v2(
            pdf_path.read_bytes(), [match], [config], taxonomy,
            api_key="x", base_url="https://api.moonshot.cn/v1", model="kimi-k2.6",
            post_func=fake_post, auto_retry=False, max_workers=1,
        )

        rows = run.records.set_index("指标编码")
        self.assertEqual(observed["image_count"], 2)
        self.assertIn('"response_mode": "sparse_disclosures"', observed["prompt"])
        self.assertAlmostEqual(rows.loc["ACTUAL_CAPITAL", "标准数值"], 2255672.884192)
        self.assertEqual(rows.loc["CORE_T2_CAPITAL", "状态"], "disclosed_na")
        self.assertEqual(rows.loc["ANC_T2_CAPITAL", "状态"], "disclosed_na")
        required_failures = run.validations[
            run.validations["校验ID"].str.endswith(":REQUIRED")
            & run.validations["状态"].eq("失败")
        ]
        self.assertTrue(required_failures.empty)


if __name__ == "__main__":
    unittest.main()
