"""Unit tests for the process intelligence domain exception hierarchy."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from process_intelligence.core import (
    DataLeakageError,
    DataValidationError,
    InsufficientDataError,
    ProcessIntelligenceError,
)


def test_process_intelligence_error_is_exception_subclass() -> None:
    assert issubclass(ProcessIntelligenceError, Exception)


def test_domain_errors_are_process_intelligence_error_subclasses() -> None:
    assert issubclass(DataValidationError, ProcessIntelligenceError)
    assert issubclass(DataLeakageError, ProcessIntelligenceError)
    assert issubclass(InsufficientDataError, ProcessIntelligenceError)


@pytest.mark.parametrize(
    ("exc_cls", "message"),
    [
        (ProcessIntelligenceError, "base failure"),
        (DataValidationError, "invalid schema"),
        (DataLeakageError, "target leaked into features"),
        (InsufficientDataError, "need more samples"),
    ],
)
def test_exceptions_preserve_message(
    exc_cls: type[ProcessIntelligenceError],
    message: str,
) -> None:
    error = exc_cls(message)
    assert str(error) == message


def test_subclass_exceptions_can_be_handled_as_process_intelligence_error() -> None:
    raised: list[type[ProcessIntelligenceError]] = []

    for exc_cls in (DataValidationError, DataLeakageError, InsufficientDataError):
        try:
            raise exc_cls("handled as base")
        except ProcessIntelligenceError as error:
            raised.append(type(error))
            assert str(error) == "handled as base"

    assert raised == [DataValidationError, DataLeakageError, InsufficientDataError]


def test_data_validation_error_differs_from_pydantic_validation_error() -> None:
    assert DataValidationError is not PydanticValidationError
    assert not issubclass(DataValidationError, PydanticValidationError)
    assert not issubclass(PydanticValidationError, DataValidationError)
