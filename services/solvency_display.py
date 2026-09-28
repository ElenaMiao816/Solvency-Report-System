"""Arrow-compatible display copies; never alter calculation/export data."""
import pandas as pd
import pyarrow as pa


def dataframe_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for index in range(len(result.columns)):
        column = result.iloc[:, index]
        if not (pd.api.types.is_object_dtype(column.dtype) or
                isinstance(column.dtype, pd.CategoricalDtype)):
            continue
        try:
            pa.array(column, from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Only incompatible columns become text. Numeric-only columns keep
            # their numeric sorting/formatting; None/NaN stay blank, not zero.
            result.isetitem(index, column.astype('string').fillna(''))
    return result
