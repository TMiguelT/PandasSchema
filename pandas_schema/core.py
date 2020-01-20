import abc
import math
import datetime
import pandas as pd
import numpy as np
import typing
import operator
import re

from . import column
from .errors import PanSchArgumentError, PanSchNoIndexError
from pandas.api.types import is_categorical_dtype, is_numeric_dtype


class BaseValidation(abc.ABC):
    """
    A validation is, broadly, just a function that maps a data frame to a list of errors
    """

    @abc.abstractmethod
    def validate(self, df: pd.DataFrame) -> typing.Iterable[Warning]:
        """
        Validates a data frame
        :param df: Data frame to validate
        :return: All validation failures detected by this validation
        """

    class Warning:
        """
        Represents a difference between the schema and data frame, found during the validation of the data frame
            Child classes can define their own subclass of :py:class:~pandas_schema.core.BaseValidation.Warning, but
            need only do that if the subclass needs to store additional data.
        """

        def __init__(self, validation: 'BaseValidation', message: str):
            self.message = message

        def __str__(self) -> str:
            """
            The entire warning message as a string
            """
            return self.message


class SeriesValidation(BaseValidation):
    """
    A _SeriesValidation validates a DataFrame by selecting a single series from it, and applying some validation
    to it
    """

    class Warning(BaseValidation.Warning):
        """
        Represents a difference between the schema and data frame, found during the validation of the data frame
        """

        def __init__(self, validation: BaseValidation, message: str, series: pd.Series):
            super().__init__(validation, message)
            self.series = series

        def __str__(self) -> str:
            """
            The entire warning message as a string
            """
            return '{} {}'.format(self.series.name, self.message)

    @abc.abstractmethod
    def select_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Selects a series from the DataFrame that will be validated
        """

    @abc.abstractmethod
    def validate_series(self, series: pd.Series) -> typing.Iterable[Warning]:
        """
        Validate a single series
        """

    def validate(self, df: pd.DataFrame) -> typing.Iterable[Warning]:
        series = self.select_series(df)
        return self.validate_series(series)


class IndexSeriesValidation(SeriesValidation):
    """
    Selects a series from the DataFrame, using label or position-based indexes that can be provided at instantiation
    or later
    """
    class Warning(SeriesValidation.Warning):
        """
        Represents a difference between the schema and data frame, found during the validation of the data frame
        """

        def __init__(self, validation: BaseValidation, message: str, series: pd.Series, col_index, positional):
            super().__init__(validation, message, series)
            self.col_index = col_index
            self.positional = positional

        def __str__(self) -> str:
            """
            The entire warning message as a string
            """
            return 'Column {} {}'.format(self.col_index, self.message)

    def __init__(self, index: typing.Union[int, str] = None, positional: bool = False, message: str = None):
        """
        Creates a new IndexSeriesValidation
        :param index: An index with which to select the series
        :param positional: If true, the index is a position along the axis (ie, index=0 indicates the first element).
        Otherwise it's a label (ie, index=0) indicates the column with the label of 0
        """
        self.index = index
        self.positional = positional
        self.custom_message = message

    @property
    def message(self):
        """
        Gets a message describing how the DataFrame cell failed the validation
        This shouldn't really be overridden, instead override default_message so that users can still set per-object
        messages
        :return:
        """
        return self.custom_message or self.default_message

    @property
    def readable_name(self):
        """
        A readable name for this validation, to be shown in validation warnings
        """
        return type(self).__name__

    @property
    def default_message(self) -> str:
        """
        Create a message to be displayed whenever this validation fails
        This should be a generic message for the validation type, but can be overwritten if the user provides a
        message kwarg
        """
        return 'failed the {}'.format(self.readable_name)

    def select_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Select a series using the data stored in this validation
        """
        if self.index is None:
            raise PanSchNoIndexError()

        if self.positional:
            return df.iloc[self.index]
        else:
            return df.loc[self.index]

    @abc.abstractmethod
    def validate_series(self, series: pd.Series) -> typing.Iterable[Warning]:
        pass


class BooleanSeriesValidation(IndexSeriesValidation):
    """
    Validation is defined by the function :py:meth:~select_cells that returns a boolean series.
        Each cell that has False has failed the validation.

        Child classes need not create their own :py:class:~pandas_schema.core.BooleanSeriesValidation.Warning subclass,
        because the data is in the same form for each cell. You need only define a :py:meth~default_message.
    """
    class Warning(IndexSeriesValidation.Warning):
        def __init__(self, validation: BaseValidation, message: str, series: pd.Series, col_index, positional, row_index, value):
            super().__init__(validation, message, series, col_index, positional)
            self.row_index = row_index
            self.value = value

        def __str__(self) -> str:
            return '{{row: {}, column: "{}"}}: "{}" {}'.format(self.row_index, self.col_index, self.value, self.message)

    @abc.abstractmethod
    def select_cells(self, series: pd.Series) -> pd.Series:
        """
        A BooleanSeriesValidation must return a boolean series. Each cell that has False has failed the
            validation
        :param series: The series to validate
        """
        pass

    def validate_series(self, series: pd.Series) -> typing.Iterable[Warning]:
        indices = self.select_cells(series)
        cells = series[indices]
        return (
            Warning(self, self.message, series, self.index, self.positional, row_idx, cell) for row_idx, cell in cells.items()
        )
