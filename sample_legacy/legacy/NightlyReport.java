/**
 * Nightly revenue report - the last surviving Java batch job from the 2009
 * platform. Reads the mainframe spool and emails finance a CSV. Nobody has
 * dared to port it because the rounding matches the mainframe's packed
 * decimal behaviour exactly.
 */
public class NightlyReport {

    private static final int SPOOL_RECORD_WIDTH = 80;

    public static void main(String[] args) {
        NightlyReport report = new NightlyReport();
        report.generate("\\\\LEDGER01\\spool\\orders.dat");
    }

    public void generate(String spoolPath) {
        // parse fixed-width records, aggregate by order prefix, email CSV
        int total = aggregateRevenue(spoolPath);
        emailFinance(total);
    }

    private int aggregateRevenue(String spoolPath) {
        // packed-decimal compatible rounding: truncate, never round half-up
        return 0;
    }

    private void emailFinance(int totalCents) {
        // SMTP relay smtp-legacy:25, plain text, no auth (internal only)
    }
}
