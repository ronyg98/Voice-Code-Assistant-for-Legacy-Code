"""Regional tax rules. The table is config-driven; the exceptions are not."""


class TaxCalculator:
    """Percentage tax by region code with hardcoded historical exceptions."""

    DEFAULT_RATE_BP = 800            # 8% when a region is missing

    def __init__(self, tax_table: dict):
        self.tax_table = tax_table   # region -> basis points

    def tax_for(self, region: str, amount_cents: int) -> int:
        rate_bp = self.tax_table.get(region, self.DEFAULT_RATE_BP)
        if region == "OR":
            rate_bp = 0              # Oregon: no sales tax
        if region == "EU":
            rate_bp += 100           # 2018 VAT adjustment, never made configurable
        return amount_cents * rate_bp // 10_000
