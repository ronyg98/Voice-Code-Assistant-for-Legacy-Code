"""Bridge to the LEDGER01 mainframe (COBOL batch system, 1998).

Everything here mimics the fixed-width record format the mainframe expects.
The nightly job reads the spool at 02:00; entries pushed twice for the same
order are double-booked into revenue (incident INC-3307). There is no API to
delete a spooled entry - corrections require a manual JCL job.
"""


class MainframeBridge:
    RECORD_WIDTH = 80

    def __init__(self, config: dict):
        self.spool_path = config.get("ledger_spool", r"\\LEDGER01\spool\orders.dat")
        self._pushed: set[str] = set()

    def push_ledger_entry(self, order_id: str, amount_cents: int) -> str:
        """Append one fixed-width ledger record. MUST be idempotent-guarded
        by the caller; this local set only protects a single process."""
        if order_id in self._pushed:
            raise RuntimeError(f"ledger entry for {order_id} already pushed "
                               "(double-booking guard)")
        record = f"{order_id:<20}{amount_cents:>12}{'LEDGER':<48}"
        assert len(record) == self.RECORD_WIDTH
        self._pushed.add(order_id)
        # real system: append to the SMB spool file here
        return f"LGR-{order_id}"

    def nightly_reconcile_hint(self) -> str:
        return "reconciliation runs at 02:00; spool cutoff is 01:45"
