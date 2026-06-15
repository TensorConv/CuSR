# EvoGPSource intentionally not re-exported here: its import pulls evogp +
# evogp.evogp_cuda at module load, which breaks CPU-only users of other
# sources. Import explicitly: `from cusr.bench.sources.evogp import EvoGPSource`.
