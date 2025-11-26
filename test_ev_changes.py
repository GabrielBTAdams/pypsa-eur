# test_ev_changes.py
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts._helpers import mock_snakemake
from scripts.prepare_sector_network import add_EVs


"""Test EV additions with mocked snakemake object"""
snakemake = mock_snakemake(
    "prepare_sector_network",
    simpl="",
    clusters="5",
    ll="v1.0",
    opts="Co2L-4H",
    sector_opts="24H",
    planning_horizons="2050",
    configfiles="config/config.test.yaml"
)

# Now you can call your functions directly
# add_EVs(
#     n,
#     avail_profile,
#     dsm_profile,
#     p_set,
#     shares["electric"],
#     number_cars,
#     temperature,
#     spatial,
#     options,
# )
print("Mock snakemake created successfully")
print(f"Input files: {snakemake.input}")
print(f"Output files: {snakemake.output}")