"""test_ncu_profile.py — unit tests for the ncu CSV parser + summarizer (no GPU).

Feeds a synthetic ncu `--csv --page raw` sample (the real column layout) and
asserts SOL extraction, warp-stall ranking, and the instruction-roofline point.
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ncu_profile", HERE / "ncu_profile.py")
ncu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncu)

# Minimal ncu --csv --page raw shape: a banner line, then the header, then rows.
# Columns mirror real ncu output; only the ones the parser uses must be correct.
SAMPLE = (
    '"==PROF== Disconnected"\n'
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","ad_jacobian_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","0.83"\n'
    '"0","ad_jacobian_kernel","gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","%","0.41"\n'
    '"0","ad_jacobian_kernel","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","0.22"\n'
    '"0","ad_jacobian_kernel","smsp__issue_active.avg.pct_of_peak_sustained_active","%","94.5"\n'
    '"0","ad_jacobian_kernel","smsp__inst_executed.sum","inst","2000000"\n'
    '"0","ad_jacobian_kernel","gpu__time_duration.sum","us","1000"\n'
    '"0","ad_jacobian_kernel","dram__bytes.sum","byte","4000000"\n'
    '"0","ad_jacobian_kernel","smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio","","3.1"\n'
    '"0","ad_jacobian_kernel","smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio","","5.7"\n'
    '"0","ad_jacobian_kernel","smsp__average_warps_issue_stalled_wait_per_issue_active.ratio","","1.2"\n'
)


def test_parse_extracts_kernel_and_metrics():
    parsed = ncu.parse_ncu_csv(SAMPLE)
    assert "ad_jacobian_kernel" in parsed
    m = parsed["ad_jacobian_kernel"]
    assert m["sm__throughput.avg.pct_of_peak_sustained_elapsed"][0] == 0.83
    assert m["smsp__issue_active.avg.pct_of_peak_sustained_active"][1] == "%"


def test_summarize_sol_and_issue():
    summ = ncu.summarize_kernel(ncu.parse_ncu_csv(SAMPLE)["ad_jacobian_kernel"])
    assert summ["sol"]["compute_sol_pct"] == 0.83
    assert summ["sol"]["memory_sol_pct"] == 0.41
    assert summ["sol"]["issue_active_pct"] == 94.5
    # the whole point: compute & memory SOL are <1%, issue is ~saturated
    assert summ["sol"]["compute_sol_pct"] < 1.0
    assert summ["sol"]["memory_sol_pct"] < 1.0
    assert summ["sol"]["issue_active_pct"] > 90.0


def test_summarize_top_stall_is_mio_throttle():
    summ = ncu.summarize_kernel(ncu.parse_ncu_csv(SAMPLE)["ad_jacobian_kernel"])
    reasons = [r for r, _ in summ["top_stalls"]]
    assert reasons[0] == "mio_throttle"           # 5.7 is the largest
    assert reasons[1] == "long_scoreboard"        # 3.1 next
    assert dict(summ["top_stalls"])["wait"] == 1.2


def test_instruction_roofline_point():
    summ = ncu.summarize_kernel(ncu.parse_ncu_csv(SAMPLE)["ad_jacobian_kernel"])
    ir = summ["instruction_roofline"]
    # duration is microseconds: GIPS = inst / dur_us / 1e3 = 2e6/1e3/1e3 = 2.0
    assert ir["gips"] == 2000000 / 1000 / 1e3
    assert ir["inst_per_byte"] == 2000000 / 4000000   # 0.5 inst/byte


def test_parse_handles_no_header():
    assert ncu.parse_ncu_csv("garbage\nno header here\n") == {}


# duration unit VARIES (ms for ad/fusedfd, us for revad) — must normalize (codex bug).
SAMPLE_MS = (
    '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    '"0","fd_jacobian_fused_kernel","smsp__inst_executed.sum","inst","50727633"\n'
    '"0","fd_jacobian_fused_kernel","gpu__time_duration.sum","ms","1.102432"\n'
)


def test_gips_normalizes_millisecond_duration():
    summ = ncu.summarize_kernel(ncu.parse_ncu_csv(SAMPLE_MS)["fd_jacobian_fused_kernel"])
    # 50,727,633 inst / 1.102432 ms = 46.01 GIPS (NOT 46014 — the 1000x bug)
    assert abs(summ["instruction_roofline"]["gips"] - 46.014) < 0.1


# Wide `--page raw` shape: header row of metric IDs, a units row, then kernel rows.
WIDE = (
    '"ID","Kernel Name","sm__throughput.avg.pct_of_peak_sustained_elapsed",'
    '"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",'
    '"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",'
    '"sm__inst_executed.avg.pct_of_peak_sustained_elapsed",'
    '"smsp__inst_executed.sum","gpu__time_duration.sum","dram__bytes.sum"\n'
    '"","","%","%","%","%","inst","us","byte"\n'
    '"0","rev_jacobian_kernel(const int *, float *)","7.45","5.85","0.34","7.43","2000000","1000","4000000"\n'
)


def test_parse_wide_raw_format_keys_on_bare_name():
    parsed = ncu.parse_ncu_csv(WIDE)
    assert "rev_jacobian_kernel" in parsed           # signature stripped
    m = parsed["rev_jacobian_kernel"]
    assert m["sm__throughput.avg.pct_of_peak_sustained_elapsed"][0] == 7.45


def test_summarize_wide_sol_and_issue_fallback():
    summ = ncu.summarize_kernel(ncu.parse_ncu_csv(WIDE)["rev_jacobian_kernel"])
    assert summ["sol"]["compute_sol_pct"] == 7.45
    assert summ["sol"]["dram_sol_pct"] == 0.34
    # all <10% of peak: the not-FLOP/not-BW-bound thesis
    assert summ["sol"]["compute_sol_pct"] < 10 and summ["sol"]["dram_sol_pct"] < 1
    # issue_active absent -> falls back to inst-executed %
    assert summ["sol"]["issue_active_pct"] == 7.43
    assert summ["instruction_roofline"]["gips"] == 2.0   # 2e6 inst / 1e6 ns
