# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build land transport demand per clustered model region including efficiency
improvements due to drivetrain changes, time series for electric vehicle
availability and demand-side management constraints.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from scripts._helpers import (
    configure_logging,
    generate_periodic_profiles,
    get_snapshots,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

def extend_segment_distribution(eurostat_distr_fn, pop_layout, year):
    seg_keys = {
    'CAR': 'Passenger cars',
    'MOTO': 'Powered two-wheelers',
    'BUS_TOT': 'Motor coaches, buses and trolley buses',
    'LOR': 'Heavy duty vehicles',
    'TRC': 'Heavy duty vehicles',
    'TRL_STRL': 'Heavy duty vehicles',
    'VG_LE3P5': 'Light duty vehicles',
    'SPE': 'Heavy duty vehicles'
    }

    # Read Eurostat distribution data and keep only entries with unit == "NR"
    eurostat_data = pd.read_csv(eurostat_distr_fn)
    # keep only 'NR' counts
    to_drop = ['freq', 'unit']
    # drop columns explicitly and ignore if any are missing
    eurostat_data = eurostat_data[eurostat_data["unit"] == "NR"].drop(columns=to_drop, errors="ignore")

    # prepare year-specific eurostat counts
    year_col = str(year)
    ed = eurostat_data.copy()
    ed = ed.rename(columns={'geo\\TIME_PERIOD': 'geo'})

    if year_col not in ed.columns:
        logger.warning(f'year {year} not found in eurostat vehicle stock distribution columns')

    ed_year = ed[['vehicle', 'geo', year_col]].rename(columns={year_col: 'count'})

    # map vehicle codes to segment names (multiple vehicle codes may map to the same segment)
    vehicle_map = seg_keys.copy()

    ed_year['segment'] = ed_year['vehicle'].map(vehicle_map)
    # drop unmapped vehicle rows
    ed_year = ed_year.dropna(subset=['segment'])

    # segments of interest
    segments = ['Passenger cars',
                'Powered two-wheelers',
                'Motor coaches, buses and trolley buses',
                'Light duty vehicles',
                'Heavy duty vehicles'
                ]
    ed_year = ed_year[ed_year['segment'].isin(segments)]

    # counts_per_geo to have counts per geo per segment
    # group by (geo, segment) to combine multiple vehicle codes mapping to the same segment will be combined
    counts_per_geo = ed_year.groupby(['geo', 'segment'])['count'].sum().unstack()
    # make sure all segment columns exist
    for s in segments:
        if s not in counts_per_geo.columns:
            counts_per_geo[s] = np.nan

    # reindex counts_per_geo to the pop_layout nodes (names)
    counts_per_geo = counts_per_geo.reindex(pop_layout.index)

    # fill missing nodes per country with the per-country mean for that segment
    for s in segments:
        for ct, group in pop_layout.groupby('ct'):
            idx = group.index
            vals = counts_per_geo.loc[idx, s]
            mean = vals.dropna().mean()
            if np.isnan(mean):
                # fallback to overall mean for this segment
                mean = counts_per_geo[s].dropna().mean()
            counts_per_geo.loc[idx, s] = vals.fillna(mean)

    # compute fractions per country for each segment and add to pop_layout
    fractions = pd.DataFrame(index=pop_layout.index, columns=[f'frac {x}' for x in segments], dtype=float)
    for ct, group in pop_layout.groupby('ct'):
        idx = group.index
        sub = counts_per_geo.loc[idx, segments].astype(float)
        totals = sub.sum(axis=0)
        frac = sub.div(totals.replace(0, np.nan), axis=1)
        # if a segment total is zero, distribute equally among nodes
        for s in segments:
            if totals[s] == 0 or np.isnan(totals[s]):
                frac[s] = 1.0 / len(idx)
        frac = frac.fillna(1.0 / len(idx))

        for s in segments:
            col = f'frac {s}'
            fractions.loc[idx, col] = frac[s].values

    # attach fraction columns to pop_layout
    for col in fractions.columns:
        pop_layout[col] = fractions[col].astype(float)

    return pop_layout


def build_nodal_transport_data(fn, pop_layout, year):
    # read transport numbers and select year
    transport_data = pd.read_csv(fn, index_col=[0, 1])
    transport_data = transport_data.xs(year, level="year")

    # break numbers down to nodal level based on population layout
    nodal_transport_data = transport_data.loc[pop_layout.ct].fillna(0.0)
    nodal_transport_data.index = pop_layout.index

    # add nodal transport data for specified segments - avoid "efficiency" and "load factor" columns
    car_cols = transport_data.columns[~transport_data.columns.str.contains("efficiency|load factor")]

    # segments we expect and how they appear in transport_data column names
    segments = [
        'Passenger cars',
        'Powered two-wheelers',
        'Motor coaches, buses and trolley buses',
        'Light duty vehicles',
        'Heavy duty vehicles',
    ]

    # apply per-segment fractions from pop_layout where available
    for segment in segments:
        # find transport columns that correspond to this segment
        seg_cols = [c for c in car_cols if segment in c]
        if len(seg_cols) == 0:
            continue

        # prefer the explicit per-segment 'frac <segment>' column if present
        frac_col = f'frac {segment}' if f'frac {segment}' in pop_layout.columns else None

        # fallback to the original uniform fraction column if present
        if frac_col is None and 'fraction' in pop_layout.columns:
            frac_col = 'fraction'

        # if still missing, distribute equally among nodes of the country
        if frac_col is None:
            # build a temporary equal fraction Series per node within its country
            equal_frac = pop_layout.groupby('ct').apply(lambda g: pd.Series(1.0 / len(g), index=g.index)).reindex(pop_layout.index)
            nodal_transport_data[seg_cols] = nodal_transport_data[seg_cols].mul(equal_frac, axis=0)
        else:
            nodal_transport_data[seg_cols] = nodal_transport_data[seg_cols].mul(pop_layout[frac_col], axis=0)

    # ensure 'mio km-driven Rail' (or any columns containing 'Rail') are scaled by uniform pop_layout['fraction']
    rail_cols = [c for c in car_cols if ('Rail' in c or 'rail' in c)]
    if 'fraction' in pop_layout.columns and len(rail_cols) > 0:
        nodal_transport_data[rail_cols] = nodal_transport_data[rail_cols].mul(pop_layout['fraction'], axis=0)

    # fill missing stats with average data
    stats = [
        "average fuel efficiency",
        "load factor Rail passenger",
        "load factor Rail freight",
        "load factor Heavy duty vehicles",
        "load factor Motor coaches, buses and trolley buses",
    ]
    for stat in stats:
        nodal_transport_data.loc[
            nodal_transport_data[stat] == 0.0,
            stat,
        ] = transport_data[stat].mean()

    return nodal_transport_data


def get_shape(traffic_fn):
    # averaged weekly counts from the year 2010-2015
    traffic = pd.read_csv(traffic_fn, skiprows=2, usecols=["count"]).squeeze("columns")

    # create annual profile take account time zone + summer time
    transport_shape = generate_periodic_profiles(
        dt_index=snapshots,
        nodes=nodes,
        weekly_profile=traffic.values,
    )
    transport_shape = transport_shape / transport_shape.sum()

    return transport_shape

def build_transport_demand(traffic_fn_Pkw,
                           traffic_fn_Mot,
                           traffic_fn_Lfw,
                           traffic_fn_Lkw,
                           traffic_fn_Bus, airtemp_fn, nodes, nodal_transport_data): #, snapshots, options, pop_weighted_energy_totals, nyears):
    """
    Returns transport demand per bus in unit km driven [100 km].
    """
    # get transport shape per vehicle type
    transport_shape_Pkw = get_shape(traffic_fn_Pkw)
    transport_shape_Mot = get_shape(traffic_fn_Mot)
    transport_shape_Lfw = get_shape(traffic_fn_Lfw)
    transport_shape_Lkw = get_shape(traffic_fn_Lkw)
    transport_shape_Bus = get_shape(traffic_fn_Bus)

    # non-electrified rail share
    non_elec_rail = (1 - (pop_weighted_energy_totals["electricity rail"]
                          / pop_weighted_energy_totals["total rail"]))

    # total demand of driven vehicle-km [mio km]
    pkw = nodal_transport_data["mio km-driven Passenger cars"]
    mot = nodal_transport_data["mio km-driven Powered two-wheelers"]
    lfw = nodal_transport_data["mio km-driven Light duty vehicles"]
    lkw = nodal_transport_data["mio km-driven Heavy duty vehicles"] \
                + non_elec_rail * nodal_transport_data["mio km-driven Rail freight"] * (
                    nodal_transport_data["load factor Rail freight"] / nodal_transport_data["load factor Heavy duty vehicles"]
                )
    bus = nodal_transport_data["mio km-driven Motor coaches, buses and trolley buses"] \
                + non_elec_rail * nodal_transport_data["mio km-driven Rail passenger"] * (
                    nodal_transport_data["load factor Rail passenger"] / nodal_transport_data["load factor Motor coaches, buses and trolley buses"]
                )

    def get_demand(profile, total, nyears, name):
        """Returns from total demand [mio km] and given profile
        demand time-series in unit [100 km]."""

        demand = (profile.multiply(total) * 1e4 * nyears)

        return pd.concat([demand], keys=[name], axis=1)
    
    demand_pkw = get_demand(transport_shape_Pkw, pkw, nyears, name="pkw")
    demand_mot = get_demand(transport_shape_Mot, mot, nyears, name="mot")
    demand_lfw = get_demand(transport_shape_Lfw, lfw, nyears, name="lfw")
    demand_lkw = get_demand(transport_shape_Lkw, lkw, nyears, name="lkw")
    demand_bus = get_demand(transport_shape_Bus, bus, nyears, name="bus")

    return pd.concat([demand_pkw, demand_mot, demand_lfw, demand_lkw, demand_bus], axis=1)


def transport_degree_factor(
    temperature,
    deadband_lower=15,
    deadband_upper=20,
    lower_degree_factor=0.5,
    upper_degree_factor=1.6,
):
    """
    Work out how much energy demand in vehicles increases due to heating and
    cooling.

    There is a deadband where there is no increase. Degree factors are %
    increase in demand compared to no heating/cooling fuel consumption.
    Returns per unit increase in demand for each place and time
    """

    dd = temperature.copy()

    dd[(temperature > deadband_lower) & (temperature < deadband_upper)] = 0.0

    dT_lower = deadband_lower - temperature[temperature < deadband_lower]
    dd[temperature < deadband_lower] = lower_degree_factor / 100 * dT_lower

    dT_upper = temperature[temperature > deadband_upper] - deadband_upper
    dd[temperature > deadband_upper] = upper_degree_factor / 100 * dT_upper

    return dd


def bev_availability_profile(fn_Pkw,
                             fn_Mot,
                             fn_Lfw,
                             fn_Lkw,
                             fn_Bus, snapshots, nodes, options):
    """
    Derive plugged-in availability for passenger electric vehicles.
    """
    # car count in typical week
    traffic_Pkw = pd.read_csv(fn_Pkw, skiprows=2, usecols=["count"]).squeeze("columns")
    traffic_Mot = pd.read_csv(fn_Mot, skiprows=2, usecols=["count"]).squeeze("columns")
    traffic_Lfw = pd.read_csv(fn_Lfw, skiprows=2, usecols=["count"]).squeeze("columns")
    traffic_Lkw = pd.read_csv(fn_Lkw, skiprows=2, usecols=["count"]).squeeze("columns")
    traffic_Bus = pd.read_csv(fn_Bus, skiprows=2, usecols=["count"]).squeeze("columns")

    def get_avail(traffic,name):
        # maximum share plugged-in availability for respective segment
        avail_max = options["bev_avail_max"][name]
        # average share plugged-in availability for respective segment
        avail_mean = options["bev_avail_mean"][name]

        # linear scaling, highest when traffic is lowest, decreases if traffic increases
        avail = avail_max - (avail_max - avail_mean) * (traffic - traffic.min()) / (
            traffic.mean() - traffic.min()
        )

        if not avail[avail < 0].empty:
            logger.warning(
                "The BEV availability weekly profile has negative values which can "
                "lead to infeasibility."
            )

        avail_periodic = generate_periodic_profiles(
            dt_index=snapshots,
            nodes=nodes,
            weekly_profile=avail.values,
        )
    
        return pd.concat([avail_periodic], keys=[name], axis=1)
    
    return pd.concat([get_avail(traffic_Pkw,name="pkw"),
                      get_avail(traffic_Mot,name="mot"),
                      get_avail(traffic_Lfw,name="lfw"),
                      get_avail(traffic_Lkw,name="lkw"),
                      get_avail(traffic_Bus,name="bus")],
                      axis=1)


def bev_dsm_profile(snapshots, nodes, options):

    def get_dsm(name):
        dsm_week = np.zeros((24 * 7,))

        # assuming that at a certain time ("bev_dsm_restriction_time") EVs have to
        # be charged to a minimum value (defined in bev_dsm_restriction_value)
        dsm_week[(np.arange(0, 7, 1) * 24 + options["bev_dsm_restriction_time"][name])] = options[
            "bev_dsm_restriction_value"][name]

        dsm_periodic = generate_periodic_profiles(
            dt_index=snapshots,
            nodes=nodes,
            weekly_profile=dsm_week,
        )
    
        return pd.concat([dsm_periodic], keys=[name], axis=1)

    return pd.concat([get_dsm(name="pkw"),
                      get_dsm(name="mot"),
                      get_dsm(name="lfw"),
                      get_dsm(name="lkw"),
                      get_dsm(name="bus")],
                      axis=1)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_transport_demand", clusters=128)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)

    nodes = pop_layout.index

    pop_weighted_energy_totals = pd.read_csv(
        snakemake.input.pop_weighted_energy_totals, index_col=0
    )

    options = snakemake.params.sector

    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day, tz="UTC"
    )

    nyears = len(snapshots) / 8760

    energy_totals_year = snakemake.params.energy_totals_year
    pop_layout_segment_distribution = extend_segment_distribution(
        snakemake.input.vehicle_segment_distribution,
        pop_layout,
        energy_totals_year
    )
    nodal_transport_data = build_nodal_transport_data(
        snakemake.input.transport_data, pop_layout_segment_distribution, energy_totals_year
    )

    transport_demand = build_transport_demand(
        snakemake.input.traffic_data_Pkw,
        snakemake.input.traffic_data_Mot,
        snakemake.input.traffic_data_Lfw,
        snakemake.input.traffic_data_Lkw,
        snakemake.input.traffic_data_Bus,
        snakemake.input.temp_air_total,
        nodes,
        nodal_transport_data,
    )

    avail_profile = bev_availability_profile(
        snakemake.input.traffic_data_Pkw,
        snakemake.input.traffic_data_Mot,
        snakemake.input.traffic_data_Lfw,
        snakemake.input.traffic_data_Lkw,
        snakemake.input.traffic_data_Bus,
        snapshots, nodes, options
    )

    dsm_profile = bev_dsm_profile(snapshots, nodes, options)

    nodal_transport_data.to_csv(snakemake.output.transport_data)
    transport_demand.to_csv(snakemake.output.transport_demand)
    avail_profile.to_csv(snakemake.output.avail_profile)
    dsm_profile.to_csv(snakemake.output.dsm_profile)
