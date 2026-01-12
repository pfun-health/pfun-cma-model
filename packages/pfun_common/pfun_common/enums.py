from typing import Type, TypeVar

StringEnumType = TypeVar("StringEnumType", bound="StringEnum")


class StringEnum:
    @classmethod
    def __getitem__(cls: Type[StringEnumType], item: str) -> StringEnumType:
        return getattr(cls, item.upper())
