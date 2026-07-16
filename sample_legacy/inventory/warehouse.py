"""In-memory stand-in for the warehouse TCP protocol client (wh01)."""


class Warehouse:
    RESERVATION_TTL_S = 900          # holds auto-expire after 15 minutes

    def __init__(self, url: str):
        self.url = url
        self._stock = {"SKU-1": 100, "SKU-2": 40, "SKU-3": 0}
        self._holds: dict[str, list] = {}

    def available(self, sku: str) -> int:
        held = sum(qty for holds in self._holds.values()
                   for s, qty in holds if s == sku)
        return self._stock.get(sku, 0) - held

    def hold(self, sku: str, quantity: int, reservation_id: str) -> None:
        self._holds.setdefault(reservation_id, []).append((sku, quantity))

    def release(self, reservation_id: str) -> None:
        self._holds.pop(reservation_id, None)
