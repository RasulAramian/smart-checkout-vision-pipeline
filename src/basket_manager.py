from collections import Counter


class BasketManager:
    """Manage basket items independently from object detection."""

    def __init__(self) -> None:
        self.items: dict[int, str] = {}

    def add_item(self, track_id: int, product: str) -> bool:
        if track_id in self.items:
            return False

        self.items[track_id] = product
        return True

    def remove_item(self, track_id: int) -> bool:
        if track_id not in self.items:
            return False

        del self.items[track_id]
        return True

    def reset(self) -> None:
        self.items.clear()

    def get_summary(self) -> dict:
        counts = Counter(self.items.values())

        return {
            "total_items": len(self.items),
            "basket_summary": dict(counts),
        }