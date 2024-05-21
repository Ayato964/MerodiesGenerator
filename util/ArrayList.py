from typing import TypeVar, List, Generic

T = TypeVar('T')


class ArrayList(Generic[T]):
    def __init__(self):
        self._items: List[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get(self, index: int) -> T:
        return self._items[index]

    def remove(self, item: T):
        self._items.remove(item)

    def size(self) -> int:
        return len(self._items)
