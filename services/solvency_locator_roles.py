"""Resolve visual page candidates by source role, never by company or page offset."""
import math
import re


LOCATOR_ROLE_LABELS = {
    'vlm_primary': 'VLM 主表页', 'vlm_continuation': 'VLM 续表 / 分项页',
    'vlm_summary': '汇总页（仅辅助核对）', 'vlm_reference': '变动分析 / 引用页（非主表）',
    'vlm_toc': '目录页（非数据页）', 'vlm_uncertain': '来源待核验',
    'vlm_recovery_candidate': '标题 / 续页补查候选（非确认结果）',
}

LOCATOR_SCOPE_RULES = {
    'SOLVENCY_MAIN': '优先主要指标章节的偿付能力充足率指标主表；管理层分析、偿付能力变化及原因分析的重复数据标reference，不与主表合并。',
    'ACTUAL_CAPITAL': '只定位实际资本明细主体及续表（S02或实际资本表），应有净资产调整、保单未来盈余等明细；偿付能力充足率指标中核心/附属资本几行合计仅标summary，不能当S02。标题不要求含S02。',
    'RECOGNIZED_ASSETS': '定位认可资产分项与认可价值列（S03或认可资产表）；合计数出现不代表明细表，非认可资产表、认可负债表不能替代。允许与实际资本表共用一页。',
    'RECOGNIZED_LIABILITIES': '定位认可负债分项与认可价值列（S04或认可负债表）；合计数出现不代表明细表，认可资产表、最低资本表不能替代。允许与认可资产表或最低资本表共用一页。',
    'MINIMUM_CAPITAL': '定位最低资本明细主体及续表（S05或最低资本章节），含保险/市场/信用风险及分散效应等分项；主要指标表仅量化风险/可资本化风险、控制风险、附加资本合计标summary，不能当S05。',
    'OPERATING_METRICS': '完整覆盖实际披露的经营指标分项：主要经营指标、效益类、规模类/分渠道、前五位产品、品质类；不同公司可没有某些分项，不要求全部存在。无重复总标题的产品/品质续页也必须返回；不能看到近三年收益率标题就忽略同页上半部分经营数据。',
    'THREE_YEAR_INVESTMENT_RETURN': '仅近三年平均投资收益率及近三年平均综合投资收益率，不能拿普通本季度收益率替代；同页可能还包含经营指标，允许多目标同时命中。',
    'REGISTERED_CAPITAL': '仅本公司基本信息直接披露的注册资本，不能取股东出资、持股额或其他公司资本。',
}


def _text(value):
    return re.sub(r'\s+', '', str(value or ''))


def page_role(table_id, title, evidence, declared=''):
    """Visible-source descriptions override an inconsistent model role label."""
    title, evidence = _text(title), _text(evidence)
    if not title and not evidence:
        return 'uncertain'
    text = title or evidence
    if re.search(r'目录|contents', text, re.I) and not re.search(r'非目录|不是目录', text):
        return 'toc'
    if re.search(r'管理层分析|变化及.*原因|变动分析|变化原因|变动情况', title):
        return 'reference'
    primary_patterns = {
        'ACTUAL_CAPITAL': r'S0?2|实际资本(?:明细)?表|^[十\d一二三四五六七八九、.()（）]*实际资本$',
        'RECOGNIZED_ASSETS': r'S0?3|(?<!非)认可资产(?:明细)?表',
        'RECOGNIZED_LIABILITIES': r'S0?4|认可负债(?:明细)?表',
        'MINIMUM_CAPITAL': r'S0?5|最低资本(?:明细)?表|^[十\d一二三四五六七八九、.()（）]*最低资本$',
        'SOLVENCY_MAIN': r'偿付能力(?:充足率)?(?:主要)?指标|偿付能力状况|Solvencysummary',
        'OPERATING_METRICS': r'经营指标|效益类|规模类|品质类|前五[位大]|分渠道',
        'THREE_YEAR_INVESTMENT_RETURN': r'近三年|三年平均',
        'REGISTERED_CAPITAL': r'注册资本|公司信息|基本信息|基本情况|公司概况',
    }
    if table_id == 'RECOGNIZED_ASSETS' and re.search(r'非认可资产(?:明细)?表|认可负债(?:明细)?表', title) and not re.search(r'(?<!非)认可资产(?:明细)?表', title):
        return 'reference'
    if table_id == 'RECOGNIZED_LIABILITIES' and re.search(r'认可资产(?:明细)?表|最低资本(?:明细)?表', title) and not re.search(r'认可负债(?:明细)?表', title):
        return 'reference'
    if re.search(primary_patterns.get(table_id, r'(?!)'), title, re.I):
        return 'continuation' if declared == 'continuation' or '续表' in title else 'primary'
    if table_id in {'ACTUAL_CAPITAL', 'RECOGNIZED_ASSETS', 'RECOGNIZED_LIABILITIES', 'MINIMUM_CAPITAL'} and re.search(r'偿付能力.*指标|主要指标|资本充足率|汇总|摘要', text):
        return 'summary'
    if declared in {'primary', 'continuation', 'summary', 'reference', 'toc'}:
        return declared
    if '续表' in text or (table_id == 'OPERATING_METRICS' and re.search(r'前五[位大]|品质类|效益类|规模类', evidence)):
        return 'continuation'
    return 'uncertain'


def locator_page_hits(item, visible_pages):
    """Accept per-page evidence; preserve compatibility with old targets/pages."""
    if item.get('found') not in (True, 'true'):
        return []
    table_id = str(item.get('table_id', ''))
    detailed = item.get('page_hits')
    if isinstance(detailed, list):
        hits = [hit for hit in detailed if isinstance(hit, dict)]
    else:
        pages = item.get('pages', [])
        hits = [dict(item, page=page) for page in pages] if isinstance(pages, list) else []
    results = []
    for hit in hits:
        page = hit.get('page')
        if isinstance(page, bool) or not str(page).isdigit() or int(page) not in visible_pages:
            continue
        title = str(hit.get('source_title', '') or '')
        evidence = str(hit.get('evidence', '') or '')
        try:
            confidence = float(hit.get('confidence', item.get('confidence', 0)) or 0)
        except (TypeError, ValueError):
            confidence = 0
        if not math.isfinite(confidence):
            confidence = 0
        results.append(dict(page=int(page), source_title=title, evidence=evidence,
            role=page_role(table_id, title, evidence, str(hit.get('role', ''))),
            confidence=max(0, min(confidence, 1)),
            coverage_groups=hit.get('coverage_groups') if isinstance(hit.get('coverage_groups'), list) else []))
    return results


def select_locator_pages(table_id, hits):
    """Never union a summary/reference into a detail table's extraction range."""
    by_role = {}
    for hit in hits:
        by_role.setdefault(hit['role'], set()).add(hit['page'])
    approved = [h for h in hits if h['role'] in {'primary', 'continuation'}]
    # Legacy/ambiguous candidates remain visible for manual editing, but cannot
    # inherit high confidence from a different primary page.
    selected = approved or [h for h in hits if h['role'] == 'uncertain']
    pages = sorted({h['page'] for h in selected})
    confidence = min((h['confidence'] for h in selected), default=0)
    notes = []
    if not pages:
        notes.append('只发现汇总/引用/目录候选，未找到可确认的目标数据页' if hits else '视觉模型未找到目标页')
    elif not approved:
        notes.append('候选页缺少明确的主表/续表来源角色，请人工核对')
    if pages and confidence < .70:
        notes.append(f'包含低置信候选页（最低{confidence:.0%}），请逐页核对')
    if approved and by_role.get('uncertain', set()) - set(pages):
        notes.append('另有来源待核验候选，未自动并入主表范围，请检查是否为续表')
    sources = {'vlm_' + role: sorted(values) for role, values in by_role.items()}
    details = []
    for hit in sorted(hits, key=lambda h: (h['page'], h['role'])):
        state = '采用' if hit in selected else '辅助/未采用'
        details.append(f"物理页{hit['page']}[{state}/{hit['role']}]：{hit['source_title']}；{hit['evidence']}")
    return pages, confidence, '；'.join(notes), sources, '\n'.join(dict.fromkeys(details))
