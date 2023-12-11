from fastapi import HTTPException, status


class BaseCustomException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg


class BoundsTypeError(BaseCustomException):
    def __init__(self, i: int) -> None:
        msg = f"Element {i} of bounds array is not a number."
        super().__init__(msg)


class BadRequestError(HTTPException):
    STATUS_CODE = status.HTTP_400_BAD_REQUEST
