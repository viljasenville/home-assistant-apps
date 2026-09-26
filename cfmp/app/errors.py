"""Error envelope.

The contract specifies ``{"error": "<code>", "message": "..."}`` for every
failure, which is not FastAPI's default ``{"detail": ...}``. These handlers
replace it everywhere, including the validation errors FastAPI raises before
a route is reached.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(HTTPException):
    def __init__(self, status_code: int, error: str, message: str | None = None) -> None:
        super().__init__(status_code=status_code, detail=message or error)
        self.error = error
        self.message = message


def _payload(error: str, message: str | None) -> dict:
    body = {"error": error}
    if message:
        body["message"] = message
    return body


def unauthorized(message: str = "Missing or invalid bearer token.") -> ApiError:
    return ApiError(401, "unauthorized", message)


def not_found(message: str) -> ApiError:
    return ApiError(404, "not_found", message)


def model_not_available(model: str) -> ApiError:
    return ApiError(404, "model_not_available", f"Model '{model}' is not available here.")


def bad_request(message: str) -> ApiError:
    return ApiError(400, "bad_request", message)


def conflict(error: str, message: str) -> ApiError:
    return ApiError(409, error, message)


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.error, exc.message),
            headers=exc.headers or {},
        )

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException) -> JSONResponse:
        codes = {400: "bad_request", 401: "unauthorized", 404: "not_found", 409: "conflict"}
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(codes.get(exc.status_code, "error"), str(exc.detail)),
            headers=exc.headers or {},
        )

    # A malformed body is a 400 in this contract; 422 is reserved for
    # "not enough data to train", which is a successful request with a
    # negative answer.
    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        detail = first.get("msg", "Invalid request body.")
        return JSONResponse(
            status_code=400,
            content=_payload(
                "bad_request", f"{location}: {detail}" if location else detail
            ),
        )

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content=_payload("bad_request", str(exc)))
