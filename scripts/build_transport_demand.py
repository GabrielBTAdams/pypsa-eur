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


def build_nodal_transport_data(fn, pop_layout, year):
    # get numbers of car and fuel efficiency per country
    transport_data = pd.read_csv(fn, index_col=[0, 1])
    transport_data = transport_data.xs(year, level="year")

    # break number of cars down to nodal level based on population density
    nodal_transport_data = transport_data.loc[pop_layout.ct].fillna(0.0)
    nodal_transport_data.index = pop_layout.index
    # add nodal transport data for specified segments
    car_cols = transport_data.columns[~transport_data.columns.str.contains("efficiency")]
    nodal_transport_data[car_cols] = (
        nodal_transport_data[car_cols].mul(pop_layout["fraction"], axis=0)
    )
    # fill missing fuel efficiency with average data
    nodal_transport_data.loc[
        nodal_transport_data["average fuel efficiency"] == 0.0,
        "average fuel efficiency",
    ] = transport_data["average fuel efficiency"].mean()

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

    # get heating demand for correction to demand time series
    temperature = xr.open_dataarray(airtemp_fn).to_pandas()

    # correction factors for vehicle heating
    dd_ICE = transport_degree_factor(
        temperature,
        options["transport_heating_deadband_lower"],
        options["transport_heating_deadband_upper"],
        options["ICE_lower_degree_factor"],
        options["ICE_upper_degree_factor"],
    )

    # divide out the heating/cooling demand from ICE totals
    ice_correction_Pkw = (transport_shape_Pkw * (1 + dd_ICE)).sum() / transport_shape_Pkw.sum()
    ice_correction_Mot = (transport_shape_Mot * (1 + dd_ICE)).sum() / transport_shape_Mot.sum()
    ice_correction_Lfw = (transport_shape_Lfw * (1 + dd_ICE)).sum() / transport_shape_Lfw.sum()
    ice_correction_Lkw = (transport_shape_Lkw * (1 + dd_ICE)).sum() / transport_shape_Lkw.sum()
    ice_correction_Bus = (transport_shape_Bus * (1 + dd_ICE)).sum() / transport_shape_Bus.sum()

    # non-electrified rail share
    non_elec_rail = (1 - (pop_weighted_energy_totals["electricity rail"]
                          / pop_weighted_energy_totals["total rail"]))

    # total demand of driven vehicle-km [mio km]
    pkw = nodal_transport_data["mio km-driven Passenger cars"]
    mot = nodal_transport_data["mio km-driven Powered two-wheelers"]
    lfw = nodal_transport_data["mio km-driven Light duty vehicles"]
    lkw = nodal_transport_data["mio km-driven Heavy duty vehicles"] \
                + non_elec_rail * nodal_transport_data["mio km-driven Rail"]
    bus = nodal_transport_data["mio km-driven Motor coaches, buses and trolley buses"]

    def get_demand(profile, total, nyears, ice_correction, name):
        """Returns from total demand [mio km], given profile and ICE correction
        demand time-series in unit [100 km]."""

        demand = ((profile.multiply(total) * 1e4 * nyears)
                  .divide(ice_correction))

        return pd.concat([demand], keys=[name], axis=1)
    
    demand_pkw = get_demand(transport_shape_Pkw, pkw, nyears, ice_correction_Pkw, name="pkw")
    demand_mot = get_demand(transport_shape_Mot, mot, nyears, ice_correction_Mot, name="mot")
    demand_lfw = get_demand(transport_shape_Lfw, lfw, nyears, ice_correction_Lfw, name="lfw")
    demand_lkw = get_demand(transport_shape_Lkw, lkw, nyears, ice_correction_Lkw, name="lkw")
    demand_bus = get_demand(transport_shape_Bus, bus, nyears, ice_correction_Bus, name="bus")

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
        # maximum share plugged-in availability for passenger electric vehicles
        avail_max = options["bev_avail_max"]
        # average share plugged-in availability for passenger electric vehicles
        avail_mean = options["bev_avail_mean"]

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
    dsm_week = np.zeros((24 * 7,))

    # assuming that at a certain time ("bev_dsm_restriction_time") EVs have to
    # be charged to a minimum value (defined in bev_dsm_restriction_value)
    dsm_week[(np.arange(0, 7, 1) * 24 + options["bev_dsm_restriction_time"])] = options[
        "bev_dsm_restriction_value"
    ]

    return generate_periodic_profiles(
        dt_index=snapshots,
        nodes=nodes,
        weekly_profile=dsm_week,
    )


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
    nodal_transport_data = build_nodal_transport_data(
        snakemake.input.transport_data, pop_layout, energy_totals_year
    )

    transport_demand = build_transport_demand(
        snakemake.input.traffic_data_Pkw, # ="data/bundle/emobility/Pkw__count",
        snakemake.input.traffic_data_Mot, # ="data/bundle/emobility/Pkw__count", TEMP: replace with Mot data once generated
        snakemake.input.traffic_data_Lfw, # ="data/bundle/emobility/Lfw__count",
        snakemake.input.traffic_data_Lkw, # ="data/bundle/emobility/Lkw__count",
        snakemake.input.traffic_data_Bus, # ="data/bundle/emobility/Bus__count",

        snakemake.input.temp_air_total,
        nodes,
        nodal_transport_data,
    )

    avail_profile = bev_availability_profile(
        snakemake.input.traffic_data_Pkw, # ="data/bundle/emobility/Pkw__count",
        snakemake.input.traffic_data_Mot, # ="data/bundle/emobility/Pkw__count", TEMP: replace with Mot data once generated
        snakemake.input.traffic_data_Lfw, # ="data/bundle/emobility/Lfw__count",
        snakemake.input.traffic_data_Lkw, # ="data/bundle/emobility/Lkw__count",
        snakemake.input.traffic_data_Bus, # ="data/bundle/emobility/Bus__count",
        snapshots, nodes, options
    )

    dsm_profile = bev_dsm_profile(snapshots, nodes, options)

    nodal_transport_data.to_csv(snakemake.output.transport_data)
    transport_demand.to_csv(snakemake.output.transport_demand)
    avail_profile.to_csv(snakemake.output.avail_profile)
    dsm_profile.to_csv(snakemake.output.dsm_profile)
