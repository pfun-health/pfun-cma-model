from typing import TypeVar, Type


StringEnumType = TypeVar('StringEnumType', bound='StringEnum')


class StringEnum:
    @classmethod
    def __getitem__(cls: Type[StringEnumType], item: str) -> StringEnumType:
        return getattr(cls, item.upper())