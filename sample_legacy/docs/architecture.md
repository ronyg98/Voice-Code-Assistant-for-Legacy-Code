# OrderFlow architecture (for auditors / viewers)

Flow of a successful order:

1. `OrderService.place_order` receives an `Order`
2. `InventoryService.reserve_stock` places 15-minute holds on warehouse bins
3. `BillingService.create_invoice` computes subtotal + 2% legacy surcharge
   + regional tax (`TaxCalculator`)
4. `MainframeBridge.push_ledger_entry` spools a fixed-width record for the
   02:00 COBOL reconciliation job (exactly once per order!)
5. `NightlyReport` (Java) aggregates the spool and emails finance

Compensation rules: billing failure releases the reservation; there is no
rollback after the ledger push.
