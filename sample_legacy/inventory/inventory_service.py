"""Stock reservation against warehouse bins."""
import uuid

from inventory.warehouse import Warehouse


class InventoryService:
    """Reserves stock for orders. Reservations expire after 15 minutes on
    the warehouse side (see Warehouse.RESERVATION_TTL_S) - callers must bill
    within that window or re-reserve."""

    def __init__(self, config: dict):
        self.warehouse = Warehouse(config.get("warehouse_url", "tcp://wh01:9911"))

    def reserve_stock(self, order) -> dict:
        for line in order.lines:
            if self.warehouse.available(line.sku) < line.quantity:
                return {"ok": False,
                        "reason": f"insufficient stock for {line.sku}"}
        reservation_id = uuid.uuid4().hex[:10]
        for line in order.lines:
            self.warehouse.hold(line.sku, line.quantity, reservation_id)
        return {"ok": True, "reservation_id": reservation_id}

    def release_reservation(self, reservation_id: str) -> None:
        self.warehouse.release(reservation_id)
