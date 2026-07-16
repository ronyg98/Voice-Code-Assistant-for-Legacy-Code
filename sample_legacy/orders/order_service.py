"""Order orchestration - the entry point for everything.

OrderService wires together inventory reservation, billing, and the
mainframe ledger push. The ordering of steps is load-bearing: stock must be
reserved BEFORE billing (see incident INC-4211 in 2019 when a refund storm
was caused by billing unreservable orders).
"""
from billing.billing_service import BillingService
from inventory.inventory_service import InventoryService
from legacy.mainframe_bridge import MainframeBridge
from orders.models import Order
from utils.config_loader import load_config


class OrderService:
    """Coordinates the full order lifecycle. One instance per worker."""

    def __init__(self):
        self.config = load_config()
        self.inventory = InventoryService(self.config)
        self.billing = BillingService(self.config)
        self.bridge = MainframeBridge(self.config)

    def place_order(self, order: Order) -> dict:
        """Reserve stock, bill the customer, push to the ledger.

        Returns a result dict with the invoice and reservation ids. If any
        step fails the previous steps are compensated (best effort - the
        mainframe push has no rollback, hence it goes last).
        """
        reservation = self.inventory.reserve_stock(order)
        if not reservation["ok"]:
            order.status = "REJECTED"
            return {"ok": False, "reason": reservation["reason"]}
        order.status = "RESERVED"

        try:
            invoice = self.billing.create_invoice(order)
        except Exception:
            self.inventory.release_reservation(reservation["reservation_id"])
            order.status = "NEW"
            raise
        order.status = "BILLED"

        # WARNING: must be called exactly once per order (double-booking bug)
        ledger_ref = self.bridge.push_ledger_entry(order.order_id,
                                                   invoice["grand_total_cents"])
        return {"ok": True, "invoice": invoice,
                "reservation_id": reservation["reservation_id"],
                "ledger_ref": ledger_ref}

    def cancel_order(self, order: Order, reservation_id: str) -> None:
        """Cancellation only works before the ledger push (see module doc)."""
        if order.status in ("SHIPPED", "BILLED"):
            raise ValueError(f"cannot cancel order in status {order.status}")
        self.inventory.release_reservation(reservation_id)
        order.status = "CANCELLED"
