from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, TypeVar

from .table_strategy_handlers import (
    ACTUAL_CAPITAL_BOUNDARY_HANDLER,
    ACTUAL_CAPITAL_COMPLETENESS_HANDLER,
    ACTUAL_CAPITAL_POSTPROCESS_HANDLER,
    ACTUAL_CAPITAL_PROMPT_HANDLER,
    GENERIC_BOUNDARY_HANDLER,
    GENERIC_COMPLETENESS_HANDLER,
    GENERIC_POSTPROCESS_HANDLER,
    GENERIC_PROMPT_HANDLER,
    MINIMUM_CAPITAL_BOUNDARY_HANDLER,
    MINIMUM_CAPITAL_POSTPROCESS_HANDLER,
    MINIMUM_CAPITAL_PROMPT_HANDLER,
    OPERATING_METRICS_BOUNDARY_HANDLER,
    RECOGNIZED_ASSETS_BOUNDARY_HANDLER,
    RECOGNIZED_ASSETS_COMPLETENESS_HANDLER,
    RECOGNIZED_ASSETS_POSTPROCESS_HANDLER,
    RECOGNIZED_ASSETS_PROMPT_HANDLER,
    RECOGNIZED_ASSETS_V2_BOUNDARY_HANDLER,
    RECOGNIZED_ASSETS_V2_COMPLETENESS_HANDLER,
    RECOGNIZED_ASSETS_V2_POSTPROCESS_HANDLER,
    RECOGNIZED_ASSETS_V2_PROMPT_HANDLER,
    RECOGNIZED_LIABILITIES_BOUNDARY_HANDLER,
    RECOGNIZED_LIABILITIES_COMPLETENESS_HANDLER,
    RECOGNIZED_LIABILITIES_POSTPROCESS_HANDLER,
    RECOGNIZED_LIABILITIES_PROMPT_HANDLER,
    OPERATING_METRICS_COMPLETENESS_HANDLER,
    OPERATING_METRICS_PROMPT_HANDLER,
    SOLVENCY_MAIN_BOUNDARY_HANDLER,
    SOLVENCY_MAIN_COMPLETENESS_HANDLER,
    SOLVENCY_MAIN_PROMPT_HANDLER,
    THREE_YEAR_RETURN_COMPLETENESS_HANDLER,
    THREE_YEAR_RETURN_BOUNDARY_HANDLER,
    THREE_YEAR_RETURN_POSTPROCESS_HANDLER,
    THREE_YEAR_RETURN_PROMPT_HANDLER,
    BoundaryHandler,
    BoundaryRequest,
    CompletenessHandler,
    CompletenessRequest,
    PostprocessHandler,
    PostprocessRequest,
    PromptHandler,
    PromptRequest,
)


GENERIC_GRID_STRATEGY_ID = "generic.grid_table.v1"

STRATEGY_SOLVENCY_MAIN = "life_solvency.solvency_main.v1"
STRATEGY_OPERATING_METRICS = "life_solvency.operating_metrics.v1"
STRATEGY_ACTUAL_CAPITAL = "life_solvency.actual_capital.v1"
STRATEGY_THREE_YEAR_RETURN = "life_solvency.three_year_return.v1"
STRATEGY_MINIMUM_CAPITAL = "life_solvency.minimum_capital.v1"
STRATEGY_RECOGNIZED_ASSETS = "life_solvency.recognized_assets.v1"
STRATEGY_RECOGNIZED_ASSETS_V2 = "life_solvency.recognized_assets.v2"
STRATEGY_RECOGNIZED_LIABILITIES = "life_solvency.recognized_liabilities.v1"

SUPPORTED_STAGES = (
    "locate",
    "extract_page",
    "reconstruct_table",
    "merge_pages",
    "build_prompt",
    "postprocess_rows",
    "enforce_boundaries",
    "validate_completeness",
)

T = TypeVar("T")


class StrategyRegistryError(ValueError):
    """Raised when a configured table strategy is unknown or inconsistent."""


@dataclass(frozen=True)
class TableStrategy:
    strategy_id: str
    table_id: str | None
    description: str
    prompt_handler: PromptHandler
    boundary_handler: BoundaryHandler
    postprocess_handler: PostprocessHandler
    completeness_handler: CompletenessHandler
    stages: tuple[str, ...] = SUPPORTED_STAGES

    def resolve(
        self,
        table_id: str,
        table_config: Mapping[str, Any] | None = None,
    ) -> "ResolvedTableStrategy":
        normalized_table_id = str(table_id or "").strip()
        if not normalized_table_id:
            raise StrategyRegistryError("目标表 table_id 不能为空。")
        if self.table_id and self.table_id != normalized_table_id:
            raise StrategyRegistryError(
                f"策略 {self.strategy_id} 仅适用于 {self.table_id}，"
                f"不能用于 {normalized_table_id}。"
            )
        return ResolvedTableStrategy(
            strategy_id=self.strategy_id,
            table_id=normalized_table_id,
            description=self.description,
            prompt_handler=self.prompt_handler,
            boundary_handler=self.boundary_handler,
            postprocess_handler=self.postprocess_handler,
            completeness_handler=self.completeness_handler,
            stages=self.stages,
            table_config=dict(table_config or {}),
        )


@dataclass(frozen=True)
class ResolvedTableStrategy:
    strategy_id: str
    table_id: str
    description: str
    prompt_handler: PromptHandler
    boundary_handler: BoundaryHandler
    postprocess_handler: PostprocessHandler
    completeness_handler: CompletenessHandler
    stages: tuple[str, ...]
    table_config: Mapping[str, Any]

    def _configured(self, request: T) -> T:
        if not self.table_config:
            return request
        return replace(request, table_config=dict(self.table_config))

    def build_prompt(self, request: PromptRequest) -> str:
        return self.prompt_handler(self._configured(request))

    def enforce_boundaries(
        self,
        request: BoundaryRequest,
    ) -> tuple[list[list[str]], str]:
        return self.boundary_handler(self._configured(request))

    def postprocess(
        self,
        request: PostprocessRequest,
    ) -> tuple[list[list[str]], tuple[str, ...]]:
        return self.postprocess_handler(self._configured(request))

    def validate_completeness(self, request: CompletenessRequest) -> Any:
        return self.completeness_handler(self._configured(request))

    def run(
        self,
        stage: str,
        handler: Callable[..., T],
        /,
        *args: Any,
        inject_table_id: str = "keyword",
        **kwargs: Any,
    ) -> T:
        """Run a compatibility-stage handler in the active strategy context.

        New prompt, boundary, postprocess and completeness work should use the
        dedicated handler methods. This adapter remains for shared legacy stages
        such as page extraction and cross-page merging.
        """
        if stage not in self.stages:
            raise StrategyRegistryError(
                f"策略 {self.strategy_id} 未登记处理阶段：{stage}"
            )
        configured_table_id = kwargs.pop("table_id", "")
        if configured_table_id and configured_table_id != self.table_id:
            raise StrategyRegistryError(
                f"策略 {self.strategy_id} 绑定 {self.table_id}，"
                f"但调用传入了 {configured_table_id}。"
            )
        token = _ACTIVE_TABLE_STRATEGY.set(self)
        try:
            if inject_table_id == "positional":
                result = handler(self.table_id, *args, **kwargs)
            elif inject_table_id == "keyword":
                result = handler(*args, table_id=self.table_id, **kwargs)
            else:
                raise StrategyRegistryError(
                    "inject_table_id 必须是 positional 或 keyword。"
                )
        finally:
            _ACTIVE_TABLE_STRATEGY.reset(token)
        if hasattr(result, "profile_strategy_id"):
            setattr(result, "profile_strategy_id", self.strategy_id)
        return result


class TableStrategyRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, TableStrategy] = {}
        self._default_by_table: dict[str, str] = {}

    def register(self, strategy: TableStrategy, *, default: bool = False) -> None:
        strategy_id = strategy.strategy_id.strip()
        if not strategy_id:
            raise StrategyRegistryError("strategy_id 不能为空。")
        if strategy_id in self._by_id:
            raise StrategyRegistryError(f"策略重复注册：{strategy_id}")
        self._by_id[strategy_id] = strategy
        if default:
            if not strategy.table_id:
                raise StrategyRegistryError("通用策略不能登记为具体目标表默认策略。")
            if strategy.table_id in self._default_by_table:
                raise StrategyRegistryError(
                    f"目标表 {strategy.table_id} 已有默认策略。"
                )
            self._default_by_table[strategy.table_id] = strategy_id

    def get(self, strategy_id: str) -> TableStrategy:
        normalized = str(strategy_id or "").strip()
        try:
            return self._by_id[normalized]
        except KeyError as exc:
            raise StrategyRegistryError(f"未注册的提取策略：{normalized}") from exc

    def resolve(
        self,
        table_id: str,
        strategy_id: str | None = None,
        table_config: Mapping[str, Any] | None = None,
    ) -> ResolvedTableStrategy:
        normalized_table_id = str(table_id or "").strip()
        configured_strategy_id = str(strategy_id or "").strip()
        if not configured_strategy_id:
            configured_strategy_id = self._default_by_table.get(
                normalized_table_id,
                GENERIC_GRID_STRATEGY_ID,
            )
        return self.get(configured_strategy_id).resolve(
            normalized_table_id,
            table_config,
        )

    def default_strategy_id(self, table_id: str) -> str:
        return self.resolve(table_id).strategy_id

    def all(self) -> tuple[TableStrategy, ...]:
        return tuple(self._by_id.values())


TABLE_STRATEGIES = TableStrategyRegistry()
_ACTIVE_TABLE_STRATEGY: ContextVar[ResolvedTableStrategy | None] = ContextVar(
    "active_table_strategy",
    default=None,
)

TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=GENERIC_GRID_STRATEGY_ID,
        table_id=None,
        description="通用网格表定位、逐页提取、跨页拼接和完整性校验。",
        prompt_handler=GENERIC_PROMPT_HANDLER,
        boundary_handler=GENERIC_BOUNDARY_HANDLER,
        postprocess_handler=GENERIC_POSTPROCESS_HANDLER,
        completeness_handler=GENERIC_COMPLETENESS_HANDLER,
    )
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_SOLVENCY_MAIN,
        table_id="SOLVENCY_MAIN",
        description="寿险偿付能力充足率主表原有处理逻辑。",
        prompt_handler=SOLVENCY_MAIN_PROMPT_HANDLER,
        boundary_handler=SOLVENCY_MAIN_BOUNDARY_HANDLER,
        postprocess_handler=GENERIC_POSTPROCESS_HANDLER,
        completeness_handler=SOLVENCY_MAIN_COMPLETENESS_HANDLER,
    ),
    default=True,
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_OPERATING_METRICS,
        table_id="OPERATING_METRICS",
        description="寿险主要经营指标原有披露范围与产品页排除逻辑。",
        prompt_handler=OPERATING_METRICS_PROMPT_HANDLER,
        boundary_handler=OPERATING_METRICS_BOUNDARY_HANDLER,
        postprocess_handler=GENERIC_POSTPROCESS_HANDLER,
        completeness_handler=OPERATING_METRICS_COMPLETENESS_HANDLER,
    ),
    default=True,
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_ACTUAL_CAPITAL,
        table_id="ACTUAL_CAPITAL",
        description="寿险实际资本汇总式、明细式和跨页版式原有处理逻辑。",
        prompt_handler=ACTUAL_CAPITAL_PROMPT_HANDLER,
        boundary_handler=ACTUAL_CAPITAL_BOUNDARY_HANDLER,
        postprocess_handler=ACTUAL_CAPITAL_POSTPROCESS_HANDLER,
        completeness_handler=ACTUAL_CAPITAL_COMPLETENESS_HANDLER,
    ),
    default=True,
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_THREE_YEAR_RETURN,
        table_id="THREE_YEAR_INVESTMENT_RETURN",
        description="寿险近三年投资收益率表格或句式披露原有处理逻辑。",
        prompt_handler=THREE_YEAR_RETURN_PROMPT_HANDLER,
        boundary_handler=THREE_YEAR_RETURN_BOUNDARY_HANDLER,
        postprocess_handler=THREE_YEAR_RETURN_POSTPROCESS_HANDLER,
        completeness_handler=THREE_YEAR_RETURN_COMPLETENESS_HANDLER,
    ),
    default=True,
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_MINIMUM_CAPITAL,
        table_id="MINIMUM_CAPITAL",
        description="寿险最低资本主表及风险汇总跨页版式原有处理逻辑。",
        prompt_handler=MINIMUM_CAPITAL_PROMPT_HANDLER,
        boundary_handler=MINIMUM_CAPITAL_BOUNDARY_HANDLER,
        postprocess_handler=MINIMUM_CAPITAL_POSTPROCESS_HANDLER,
        completeness_handler=GENERIC_COMPLETENESS_HANDLER,
    ),
    default=True,
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_RECOGNIZED_ASSETS,
        table_id="RECOGNIZED_ASSETS",
        description="寿险认可资产明细表（S03）两级表头版式处理逻辑（v1，保留以回退）。",
        prompt_handler=RECOGNIZED_ASSETS_PROMPT_HANDLER,
        boundary_handler=RECOGNIZED_ASSETS_BOUNDARY_HANDLER,
        postprocess_handler=RECOGNIZED_ASSETS_POSTPROCESS_HANDLER,
        completeness_handler=RECOGNIZED_ASSETS_COMPLETENESS_HANDLER,
    ),
    default=False,
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_RECOGNIZED_ASSETS_V2,
        table_id="RECOGNIZED_ASSETS",
        description="寿险认可资产明细表（S03）只取认可价值、单级表头（行次/项目/期末数/期初数）版式。",
        prompt_handler=RECOGNIZED_ASSETS_V2_PROMPT_HANDLER,
        boundary_handler=RECOGNIZED_ASSETS_V2_BOUNDARY_HANDLER,
        postprocess_handler=RECOGNIZED_ASSETS_V2_POSTPROCESS_HANDLER,
        completeness_handler=RECOGNIZED_ASSETS_V2_COMPLETENESS_HANDLER,
    ),
    default=True,
)
TABLE_STRATEGIES.register(
    TableStrategy(
        strategy_id=STRATEGY_RECOGNIZED_LIABILITIES,
        table_id="RECOGNIZED_LIABILITIES",
        description="寿险认可负债明细表（S04）只取认可价值、单级表头版式。",
        prompt_handler=RECOGNIZED_LIABILITIES_PROMPT_HANDLER,
        boundary_handler=RECOGNIZED_LIABILITIES_BOUNDARY_HANDLER,
        postprocess_handler=RECOGNIZED_LIABILITIES_POSTPROCESS_HANDLER,
        completeness_handler=RECOGNIZED_LIABILITIES_COMPLETENESS_HANDLER,
    ),
    default=True,
)


def resolve_table_strategy(
    table_id: str,
    strategy_id: str | None = None,
    table_config: Mapping[str, Any] | None = None,
) -> ResolvedTableStrategy:
    return TABLE_STRATEGIES.resolve(table_id, strategy_id, table_config)


def registered_table_strategies() -> tuple[TableStrategy, ...]:
    return TABLE_STRATEGIES.all()


def active_table_strategy(table_id: str) -> ResolvedTableStrategy:
    active = _ACTIVE_TABLE_STRATEGY.get()
    if active is not None and active.table_id == str(table_id or "").strip():
        return active
    return resolve_table_strategy(table_id)
