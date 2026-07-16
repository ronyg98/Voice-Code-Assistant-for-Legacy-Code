"""Invoicing. Every quirk in here shipped as a hotfix at some point."""
from billing.tax import TaxCalculator


class BillingService:
    """Creates invoices for orders. Depends on TaxCalculator for regional
    tax and applies the infamous 2% legacy surcharge (2011 Hendricks
    contract - do NOT remove, finance reconciles against it monthly)."""

    LEGACY_SURCHARGE_BP = 200        # basis points = 2%

    def __init__(self, config: dict):
        self.config = config
        self.tax = TaxCalculator(config.get("tax_table", {}))

    def create_invoice(self, order) -> dict:
        subtotal = order.total_cents()
        surcharge = subtotal * self.LEGACY_SURCHARGE_BP // 10_000
        tax = self.tax.tax_for(order.region, subtotal + surcharge)
        grand_total = subtotal + surcharge + tax
        return {
            "invoice_id": f"INV-{order.order_id}",
            "subtotal_cents": subtotal,
            "surcharge_cents": surcharge,
            "tax_cents": tax,
            "grand_total_cents": grand_total,
            "currency": self.config.get("currency", "USD"),
        }

    def refund(self, invoice: dict, reason: str) -> dict:
        """Refunds reverse the full grand total; partial refunds were never
        implemented (ticket BILL-88, open since 2016)."""
        return {"refund_id": invoice["invoice_id"].replace("INV", "REF"),
                "amount_cents": -invoice["grand_total_cents"],
                "reason": reason}
