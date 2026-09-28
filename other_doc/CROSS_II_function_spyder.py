"""CROSS II 公共函数（Spyder 版本）。

本文件由 ``CROSS_II_process_spyder.py`` 导入，不需要单独运行。
与原 Notebook 相比，本版本不再依赖全局变量 ``company``，也不再写死
某位用户的 EdgeDriver 路径。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urljoin

import camelot
import pymupdf as fitz
import pandas as pd
import pdfplumber
import requests
from openpyxl import Workbook
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def _new_edge_driver(
    driver_path: Optional[str] = None,
    download_dir: Optional[Path] = None,
) -> webdriver.Edge:
    """创建 Edge 浏览器；driver_path 为空时交给 Selenium Manager 管理驱动。"""
    options = Options()
    if download_dir is not None:
        download_dir.mkdir(parents=True, exist_ok=True)
        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": str(download_dir.resolve()),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
            },
        )

    if driver_path:
        resolved = Path(driver_path)
        if not resolved.exists():
            raise FileNotFoundError(f"找不到 EdgeDriver：{resolved}")
        return webdriver.Edge(service=Service(str(resolved)), options=options)
    return webdriver.Edge(options=options)


def _valid_xpath(xpath: object) -> bool:
    return isinstance(xpath, str) and xpath.strip() and xpath.strip().lower() != "nan"


def extract_elements_with_fallback(
    url: str,
    text_to_find: str,
    xpath: object,
    company: str,
    driver_path: Optional[str] = None,
) -> Optional[str]:
    """从披露页面寻找报告链接，并兼容若干公司特有的页面结构。"""
    driver = _new_edge_driver(driver_path=driver_path)
    try:
        driver.get(str(url))
        time.sleep(3)
        link: Optional[str] = None

        selectors = [
            (By.XPATH, f"//*[contains(text(), '{text_to_find}')]"),
            (By.XPATH, f"//a[contains(text(), '{text_to_find}')]"),
        ]
        for by, selector in selectors:
            try:
                element = driver.find_element(by, selector)
                if company == "海保人寿":
                    element = element.find_element(By.XPATH, "./following-sibling::a")
                candidate = element.get_attribute("href")
                if candidate == "javascript:void(0)":
                    candidate = element.get_attribute("boff")
                if candidate:
                    link = candidate
                    break
            except Exception as exc:
                print(f"{company}：按文字查找链接未成功：{exc}")

        if not link:
            try:
                current = driver.find_element(
                    By.XPATH, f"//*[contains(text(), '{text_to_find}')]"
                )
                while current:
                    parent = current.find_element(By.XPATH, "..")
                    if parent.tag_name.lower() == "a":
                        link = parent.get_attribute("href")
                        break
                    current = parent
            except Exception as exc:
                print(f"{company}：向上查找父级链接未成功：{exc}")

        if not link:
            try:
                element = driver.find_element(
                    By.XPATH, f"//a[contains(@href, '{text_to_find}')]"
                )
                link = element.get_attribute("href")
            except Exception as exc:
                print(f"{company}：按 href 查找未成功：{exc}")

        if not link and _valid_xpath(xpath):
            try:
                element = driver.find_element(By.XPATH, str(xpath).strip())
                link = element.get_attribute("href")
            except Exception as exc:
                print(f"{company}：按配置的 XPath 查找未成功：{exc}")

        if company == "泰康养老":
            try:
                element = driver.find_element(
                    By.XPATH, f"//a[contains(text(), '{text_to_find}')]"
                )
                onclick = element.find_element(By.XPATH, "../..").get_attribute("onclick")
                match = re.search(r"window\.open\('([^']+)'\)", onclick or "")
                if match:
                    link = urljoin(driver.current_url, match.group(1))
            except Exception as exc:
                print(f"{company}：特殊页面处理失败：{exc}")

        if company in {"瑞华健康", "招商仁和", "横琴人寿", "陆家嘴国泰"}:
            try:
                key_text = text_to_find if company != "招商仁和" else "偿付能力"
                menu = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, f"//*[contains(text(), '{key_text}')]")
                    )
                )
                menu.click()
                report = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, f"//a[contains(text(), '{text_to_find}')]")
                    )
                )
                link = report.get_attribute("href")
            except Exception as exc:
                print(f"{company}：展开栏目后查找失败：{exc}")

        if company == "弘康人寿":
            try:
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//*[contains(text(), '专项信息')]")
                    )
                ).click()
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//*[contains(text(), '偿付能力')]")
                    )
                ).click()
                report = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, f"//*[contains(text(), '{text_to_find}')]")
                    )
                )
                link = report.get_attribute("href")
            except Exception as exc:
                print(f"{company}：展开偿付能力栏目失败：{exc}")

        if company == "德华安顾":
            try:
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//*[contains(text(), '偿付能力')]")
                    )
                ).click()
                report = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, f"//a[contains(@download, '{text_to_find}')]")
                    )
                )
                link = report.get_attribute("href")
            except Exception as exc:
                print(f"{company}：download 属性查找失败：{exc}")

        if company in {"阳光人寿", "国民养老", "平安健康"}:
            try:
                original_windows = set(driver.window_handles)
                element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, f"//*[contains(text(), '{text_to_find}')]")
                    )
                )
                element.click()
                WebDriverWait(driver, 10).until(
                    lambda d: len(d.window_handles) > len(original_windows)
                )
                new_window = next(iter(set(driver.window_handles) - original_windows))
                driver.switch_to.window(new_window)
                link = driver.current_url
            except Exception as exc:
                print(f"{company}：新窗口链接获取失败：{exc}")

        if company == "中银三星":
            try:
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, f"//*[contains(text(), '{text_to_find}')]")
                    )
                ).click()
                link = driver.current_url
            except Exception as exc:
                print(f"{company}：点击报告链接失败：{exc}")

        return urljoin(driver.current_url, link) if link else None
    finally:
        driver.quit()


def construct_download_url(suffix: str, company: str) -> str:
    if company == "中国人寿":
        return urljoin("https://www.e-chinalife.com/", suffix)
    if company == "中意人寿":
        return "https://www.generalichina.com/expandPDF/publish.html?PDF=" + suffix
    return suffix


def get_download_url(
    url: str,
    text_to_find: str,
    xpath: object,
    company: str,
    driver_path: Optional[str] = None,
) -> str:
    download_url = extract_elements_with_fallback(
        url, text_to_find, xpath, company, driver_path
    )
    if not download_url:
        return "not found"

    nested_page_companies = {
        "国华人寿",
        "英大人寿",
        "财信人寿",
        "北京人寿",
        "复星保德信",
        "渤海人寿",
        "信泰人寿",
    }
    if company in nested_page_companies and not download_url.lower().endswith(".pdf"):
        suffix = extract_elements_with_fallback(
            download_url, text_to_find, xpath, company, driver_path
        )
        if suffix:
            download_url = construct_download_url(suffix, company)

    if company == "中意人寿":
        download_url = construct_download_url(download_url, company)
    elif company == "华汇人寿":
        suffix = extract_elements_with_fallback(
            download_url, "见附件", xpath, company, driver_path
        )
        if suffix:
            download_url = suffix
    elif company == "中荷人寿":
        suffix = extract_elements_with_fallback(
            download_url, ".pdf", xpath, company, driver_path
        )
        if suffix:
            download_url = suffix

    if company == "阳光人寿" and "file=" in download_url:
        download_url = unquote(download_url.split("file=", 1)[1])
    if company == "东吴人寿":
        download_url = download_url.replace("/cs/pdfjs/web/viewer.html?file=", "")
    if company == "国寿养老":
        download_url = download_url.replace(
            "uiFramework/js/pdfjs/web/viewer.html?file=", ""
        )
    return download_url or "not found"


def try_download_with_requests(
    pdf_url: str,
    file_path: Path,
    file_name: str,
    company: str,
) -> bool:
    if not pdf_url or pdf_url == "not found":
        return False
    file_path.mkdir(parents=True, exist_ok=True)
    target = file_path / file_name
    temp_target = target.with_suffix(target.suffix + ".part")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        with requests.get(
            pdf_url, stream=True, timeout=(15, 60), headers=headers
        ) as response:
            if response.status_code != 200:
                print(f"{company}：HTTP {response.status_code}")
                return False
            content_type = response.headers.get("Content-Type", "").lower()
            allow_unknown_type = company in {
                "东吴人寿",
                "国寿养老",
                "新华养老",
                "人保养老",
                "大家养老",
                "阳光人寿",
            }
            if "pdf" not in content_type and not allow_unknown_type:
                return False
            with temp_target.open("wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)
        temp_target.replace(target)
        print(f"{company}：requests 下载成功 -> {target}")
        return True
    except Exception as exc:
        print(f"{company}：requests 下载失败：{exc}")
        if temp_target.exists():
            temp_target.unlink()
        return False


def _wait_for_download(download_dir: Path, before: set[Path], timeout: int = 60) -> Path:
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = set(download_dir.iterdir())
        candidates = [
            path
            for path in current - before
            if path.is_file() and not path.name.endswith((".crdownload", ".tmp"))
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
        time.sleep(1)
    raise TimeoutError("等待浏览器下载超时")


def download_with_selenium(
    pdf_url: str,
    file_path: Path,
    file_name: str,
    company: str,
    driver_path: Optional[str] = None,
) -> bool:
    file_path.mkdir(parents=True, exist_ok=True)
    before = set(file_path.iterdir())
    driver = _new_edge_driver(driver_path=driver_path, download_dir=file_path)
    try:
        driver.get(pdf_url)
        time.sleep(4)

        if company == "中意人寿":
            driver.switch_to.frame("PDF")

        download_button = None
        for by, selector in [
            (By.ID, "download"),
            (By.XPATH, "//*[contains(text(), '点击下载')]"),
            (By.ID, "save"),
        ]:
            try:
                download_button = driver.find_element(by, selector)
                break
            except Exception:
                continue
        if download_button is not None:
            download_button.click()

        downloaded = _wait_for_download(file_path, before)
        target = file_path / file_name
        if downloaded.resolve() != target.resolve():
            os.replace(downloaded, target)
        print(f"{company}：浏览器下载成功 -> {target}")
        return True
    except Exception as exc:
        print(f"{company}：浏览器下载失败：{exc}")
        return False
    finally:
        driver.quit()


def download_pdf(
    pdf_url: str,
    file_path: Path,
    file_name: str,
    company: str,
    driver_path: Optional[str] = None,
) -> bool:
    if company != "恒安标准养老" and try_download_with_requests(
        pdf_url, file_path, file_name, company
    ):
        return True
    return download_with_selenium(
        pdf_url, file_path, file_name, company, driver_path
    )


def extract_tables_to_excel(pdf_path: Path, specific_chars: Iterable[str]) -> Path:
    tables = camelot.read_pdf(
        filepath=str(pdf_path),
        flavor="stream",
        pages="1-end",
        edge_tol=280,
        strip_text=" ",
        row_tol=15,
    )
    output_path = pdf_path.with_name(pdf_path.stem + "_output.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        for index, table in enumerate(tables):
            flattened = str(table.df.values.flatten())
            if any(char in flattened for char in specific_chars) and (
                "数" in flattened or "末" in flattened
            ):
                table.df.to_excel(writer, sheet_name=f"table_{index}", index=False)
    return output_path


def read_pdf_pages_with_keywords(pdf_path: Path, keywords: Iterable[str]) -> str:
    doc = fitz.open(pdf_path)
    try:
        content = []
        for page in doc:
            text = page.get_text()
            if any(keyword in text for keyword in keywords):
                content.append(text)
        return "\n".join(content)
    finally:
        doc.close()


def company_name_from_pdf(pdf_path: Path) -> str:
    return re.sub(r"20\d{2}Q[1-4].*$", "", pdf_path.stem)


def write_to_excel(
    pdf_files: Iterable[Path], keywords: Iterable[str], excel_path: Path
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "重大投资&融资"
    sheet.append(["Company", "Content"])
    for pdf_file in pdf_files:
        content = read_pdf_pages_with_keywords(pdf_file, keywords)
        content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", content)
        sheet.append([company_name_from_pdf(pdf_file), content[:32767]])
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(excel_path)


def extract_tables_from_pdf(pdf_path: Path) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if table and len(table) > 1:
                    tables.append(pd.DataFrame(table[1:], columns=table[0]))
    return tables


def get_non_blank_page(pdf_file: Path) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(page.page_number)
    return ",".join(map(str, pages)) or "1-end"


def get_company_params(company_name: str, pdf_file: Path) -> dict:
    params = {
        "flavor": "stream",
        "pages": "1-end",
        "edge_tol": 200,
        "strip_text": " ",
        "row_tol": 12,
        "column_tol": 0.1,
    }
    special = {
        "复星保德信": {"edge_tol": 40, "column_tol": 1},
        "复星联合": {"edge_tol": 100, "column_tol": 0},
        "国华人寿": {"edge_tol": 100, "column_tol": 10},
        "瑞泰人寿": {"pages": get_non_blank_page(pdf_file)},
        "东吴人寿": {"edge_tol": 50},
        "弘康人寿": {"edge_tol": 50},
        "恒安标准养老": {"column_tol": 0.1},
        "国民养老": {"edge_tol": 100, "column_tol": 3},
        "中韩人寿": {"flavor": "lattice"},
    }
    params.update(special.get(company_name, {}))
    if params["flavor"] == "lattice":
        for key in ("edge_tol", "row_tol", "column_tol"):
            params.pop(key, None)
    return params


def search_cell(
    dataframe: pd.DataFrame,
    include1: list[str],
    include2: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> tuple[Optional[int], Optional[int]]:
    include2 = include2 or []
    for row_index in range(len(dataframe)):
        for col_index in range(len(dataframe.columns)):
            value = str(dataframe.iloc[row_index, col_index])
            first_match = all(item in value for item in include1)
            second_match = bool(include2) and all(item in value for item in include2)
            excluded = bool(exclude) and any(item in value for item in exclude)
            if (first_match or second_match) and not excluded:
                return row_index, col_index
    return None, None


DEFAULT_CONDITIONS = {
    "认可资产": {"include1": ["认可资产"], "exclude": ["非认可资产", "其他认可资产", "认可资产合计"]},
    "认可负债": {"include1": ["认可负债"], "exclude": ["其他认可负债", "认可负债合计"]},
    "寿险业务保险风险最低资本合计": {"include1": ["寿险业务保险", "最低资本合计"]},
    "非寿险业务保险风险最低资本合计": {"include1": ["非寿险业务保险", "最低资本合计"]},
    "市场风险-最低资本合计": {"include1": ["市场风险", "最低资本合计"]},
    "信用风险-最低资本合计": {"include1": ["信用风险", "最低资本合计"]},
    "计入核心一级资本的保单未来盈余": {"include1": ["核心一级", "保单未来盈余"]},
    "计入核心二级资本的保单未来盈余": {"include1": ["核心二级", "保单未来盈余"]},
    "计入附属一级资本的保单未来盈余": {"include1": ["附属一级", "保单未来盈余"]},
    "计入附属二级资本的保单未来盈余": {"include1": ["附属二级", "保单未来盈余"]},
    "量化风险最低资本": {"include1": ["量化风险最低资本"], "include2": ["可资本化风险最低资本"], "exclude": ["未考虑特征系数前"]},
    "量化风险最低资本（未考虑特征系数前）": {"include1": ["量化风险最低资本", "未考虑特征系数前"], "include2": ["可资本化风险最低资本", "未考虑特征系数前"]},
    "量化风险分散效应": {"include1": ["量化风险分散效应"], "include2": ["可资本化风险分散效应"]},
    "最低资本": {"include1": ["最低资本"], "exclude": ["风险最低资本"]},
    "特定类别保险合同损失吸收效应": {"include1": ["特定类别保险合同损失吸收效应", "损失吸收"]},
    "寿险业务保险风险-损失发生风险最低资本": {"include1": ["寿险业务保险", "损失发生"]},
    "寿险业务保险风险-退保风险最低资本": {"include1": ["寿险业务保险", "退保风险"]},
    "寿险业务保险风险-费用风险最低资本": {"include1": ["寿险业务保险", "费用风险"]},
    "寿险业务保险风险-风险分散效应": {"include1": ["寿险业务保险", "风险分散"]},
    "非寿险业务保险风险-保费及准备金风险最低资本": {"include1": ["非寿险业务保险", "保费及准备金风险"]},
    "非寿险业务保险风险-巨灾风险最低资本": {"include1": ["非寿险业务保险", "巨灾风险"]},
    "非寿险业务保险风险-风险分散效应": {"include1": ["非寿险业务保险", "风险分散"]},
    "市场风险-利率风险最低资本": {"include1": ["市场风险", "利率风险"]},
    "市场风险-权益价格风险最低资本": {"include1": ["市场风险", "权益价格风险"]},
    "市场风险-房地产价格风险最低资本": {"include1": ["市场风险", "房地产价格风险"]},
    "市场风险-境外固定收益类资产价格风险最低资本": {"include1": ["市场风险", "境外固定收益类资产"]},
    "市场风险-境外权益类资产价格风险最低资本": {"include1": ["市场风险", "境外权益类资产"]},
    "市场风险-汇率风险最低资本": {"include1": ["市场风险", "汇率风险"]},
    "市场风险-风险分散效应": {"include1": ["市场风险", "风险分散"]},
    "信用风险-利差风险最低资本": {"include1": ["信用风险", "利差风险"]},
    "信用风险-交易对手违约风险最低资本": {"include1": ["信用风险", "交易对手违约风险"]},
    "信用风险-风险分散效应": {"include1": ["信用风险", "风险分散"]},
    "损失吸收调整-不考虑上限": {"include1": ["损失吸收调整", "不考虑上限"]},
    "附加资本": {"include1": ["附加资本"], "exclude": ["逆周期附加资本", "D-SII附加资本", "G-SII附加资本", "其他附加资本"]},
    "总资产": {"include1": ["总资产"], "exclude": ["收益率"]},
    "净资产": {"include1": ["净资产"], "exclude": ["收益率"]},
    "独立账户负债": {"include1": ["独立账户负债"]},
    "保险业务收入": {"include1": ["保险业务收入"]},
    "签单保费": {"include1": ["签单保费"], "exclude": ["续期签单保费", "期交签单保费"]},
}


ZHONGHE_CONDITIONS = {
    "量化风险分散效应": {"include1": ["风险分散效应"]},
    "特定类别保险合同损失吸收效应": {"include1": ["损失吸收"]},
    "量化风险最低资本（未考虑特征系数前）": {"include1": ["量化风险最低资本", "2.1+2.2+2.3-2.4-2.5"]},
    "寿险业务保险风险最低资本合计": {"include1": ["寿险业务保险", "最低资本"]},
    "非寿险业务保险风险最低资本合计": {"include1": ["非寿险业务保险", "最低资本"]},
    "市场风险-最低资本合计": {"include1": ["市场风险", "最低资本"]},
    "信用风险-最低资本合计": {"include1": ["信用风险", "最低资本"]},
    "寿险业务保险风险-损失发生风险最低资本": {"include1": ["损失发生风险"]},
    "寿险业务保险风险-退保风险最低资本": {"include1": ["退保风险最低资本"]},
    "寿险业务保险风险-费用风险最低资本": {"include1": ["费用风险最低资本"]},
    "寿险业务保险风险-风险分散效应": {"include1": ["寿险业务保险风险间的相关性效应"], "exclude": ["非寿险"]},
    "非寿险业务保险风险-保费及准备金风险最低资本": {"include1": ["保费及准备金风险"]},
    "非寿险业务保险风险-风险分散效应": {"include1": ["非寿险业务保险风险间的相关性效应"]},
    "市场风险-利率风险最低资本": {"include1": ["利率风险最低资本"]},
    "市场风险-权益价格风险最低资本": {"include1": ["权益价格风险最低资本"]},
    "市场风险-房地产价格风险最低资本": {"include1": ["房地产价格风险最低资本"]},
    "市场风险-境外固定收益类资产价格风险最低资本": {"include1": ["境外固收类资产价格风险最低资本"]},
    "市场风险-境外权益类资产价格风险最低资本": {"include1": ["境外权益类资产价格风险最低资本"]},
    "市场风险-汇率风险最低资本": {"include1": ["汇率风险最低资本"]},
    "市场风险-风险分散效应": {"include1": ["市场风险间的相关性效应"]},
    "信用风险-利差风险最低资本": {"include1": ["利差风险最低资本"]},
    "信用风险-交易对手违约风险最低资本": {"include1": ["交易对手违约风险最低资本"]},
    "信用风险-风险分散效应": {"include1": ["信用风险间的相关性效应"]},
}


def offset(
    dataframe: pd.DataFrame,
    keyphrase: str,
    company_name: str,
    rows: int = 0,
    cols: int = 0,
):
    if dataframe.empty:
        return None
    conditions = dict(DEFAULT_CONDITIONS.get(keyphrase, {"include1": [keyphrase]}))
    if company_name == "中荷人寿" and keyphrase in ZHONGHE_CONDITIONS:
        conditions = ZHONGHE_CONDITIONS[keyphrase]
    row_index, col_index = search_cell(
        dataframe,
        conditions.get("include1", []),
        conditions.get("include2", []),
        conditions.get("exclude"),
    )
    if row_index is None or col_index is None:
        return None
    target_row = row_index + rows
    target_col = col_index + cols
    if target_row >= len(dataframe) or target_col >= len(dataframe.columns):
        return None
    return dataframe.iloc[target_row, target_col]
