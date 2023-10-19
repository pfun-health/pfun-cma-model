from fastapi import HTTPException, status


class BadRequestError(HTTPException):
    STATUS_CODE = status.HTTP_400_BAD_REQUEST
