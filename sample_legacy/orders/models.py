"""Core domain objects. Unchanged since the 2014 rewrite."""
from dataclasses import dataclass, field


@dataclass
class OrderLine:
    sku: str
    quantity: int
    unit_price_cents: int

    def subtotal_cents(self) -> int:
        return self.quantity * self.unit_price_cents


@dataclass
class Order:
    order_id: str
    customer_id: str
    region: str                      # two-letter code, drives tax rules
    lines: list = field(default_factory=list)
    status: str = "NEW"              # NEW -> RESERVED -> BILLED -> SHIPPED

    def total_cents(self) -> int:
        return sum(line.subtotal_cents() for line in self.lines)
