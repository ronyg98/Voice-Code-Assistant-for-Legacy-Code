# OrderFlow (demo legacy codebase)

A deliberately "legacy" order-management system used to demo the voice code
assistant: a Python core that grew for a decade around a simulated mainframe
bridge, plus one Java batch report left over from an old migration.

Modules:

- `orders/` - order lifecycle orchestration (the heart of the system)
- `billing/` - invoicing, tax rules accumulated over years of patches
- `inventory/` - stock reservation against warehouse bins
- `legacy/` - the COBOL mainframe bridge nobody wants to touch, and a Java
  reporting job
- `utils/` - the homegrown config loader every module imports

Known folklore: the 2% "legacy surcharge" in billing exists because of a 2011
contract; `MainframeBridge.push_ledger_entry` must never be called twice for
the same order or the nightly reconciliation double-books revenue.
