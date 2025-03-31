# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import importlib
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from _benchmark import memory_logger
from _helpers import (
    configure_logging,
    create_tuples,
    set_scenario_config,
    update_config_from_wildcards,
)
from prepare_sector_network import get, prepare_costs
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


class ObjectiveValueError(Exception):
    pass


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit.
    """
    logger.info("Add land-use constraint for perfect foresight")

    def compress_series(s):
        def process_group(group):
            if group.nunique() == 1:
                return pd.Series(group.iloc[0], index=[None])
            else:
                return group

        return s.groupby(level=[0, 1]).apply(process_group)

    def new_index_name(t):
        # Convert all elements to string and filter out None values
        parts = [str(x) for x in t if x is not None]
        # Join with space, but use a dash for the last item if not None
        return " ".join(parts[:2]) + (f"-{parts[-1]}" if len(parts) > 2 else "")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # drop multi entries in case p_nom_max stays constant in different periods
    # p_nom_max = compress_series(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: f"nom_max_{row['carrier']}"
        + (f"_{row['build_year']}" if row["build_year"] is not None else ""),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values

    return n


def add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in [
        "solar",
        "solar rooftop",
        "solar-hsat",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
    ]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        existing = (
            n.generators.loc[ext_i, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                        adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators["p_nom_max"] = n.generators["p_nom_max"].clip(lower=0)


def add_solar_potential_constraints(n, config):
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, tracking, ets. ) is below the region potential.

    Example:
    ES1 0: total solar potential is 10 GW, meaning:
           solar potential : 10 GW
           solar-hsat potential : 8 GW (solar with single axis tracking is assumed to have higher land use)
    The constraint ensures that:
           solar_p_nom + solar_hsat_p_nom * 1.13 <= 10 GW
    """
    land_use_factors = {
        "solar-hsat": config["renewable"]["solar"]["capacity_per_sqkm"]
        / config["renewable"]["solar-hsat"]["capacity_per_sqkm"],
    }
    rename = {"Generator-ext": "Generator"}

    solar_carriers = ["solar", "solar-hsat"]
    solar = n.generators[
        n.generators.carrier.isin(solar_carriers) & n.generators.p_nom_extendable
    ].index

    solar_today = n.generators[
        (n.generators.carrier == "solar") & (n.generators.p_nom_extendable)
    ].index
    solar_hsat = n.generators[(n.generators.carrier == "solar-hsat")].index

    if solar.empty:
        return

    land_use = pd.DataFrame(1, index=solar, columns=["land_use_factor"])
    for carrier, factor in land_use_factors.items():
        land_use = land_use.apply(
            lambda x: (x * factor) if carrier in x.name else x, axis=1
        )

    location = pd.Series(n.buses.index, index=n.buses.index)
    ggrouper = n.generators.loc[solar].bus
    rhs = (
        n.generators.loc[solar_today, "p_nom_max"]
        .groupby(n.generators.loc[solar_today].bus.map(location))
        .sum()
        - n.generators.loc[solar_hsat, "p_nom"]
        .groupby(n.generators.loc[solar_hsat].bus.map(location))
        .sum()
        * land_use_factors["solar-hsat"]
    ).clip(lower=0)

    lhs = (
        (n.model["Generator-p_nom"].rename(rename).loc[solar] * land_use.squeeze())
        .groupby(ggrouper)
        .sum()
    )

    logger.info("Adding solar potential constraint.")
    n.model.add_constraints(lhs <= rhs, name="solar_potential")


def add_co2_sequestration_limit(n, limit_dict):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """

    if not n.investment_periods.empty:
        periods = n.investment_periods
        limit = pd.Series(
            {
                f"co2_sequestration_limit-{period}": limit_dict.get(period, 200)
                for period in periods
            }
        )
        names = limit.index
    else:
        limit = get(limit_dict, int(snakemake.wildcards.planning_horizons))
        periods = [np.nan]
        names = pd.Index(["co2_sequestration_limit"])

    n.add(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


def add_carbon_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_carbon_budget_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_max_growth(n):
    """
    Add maximum growth rates for different carriers.
    """

    opts = snakemake.params["sector"]["limit_max_growth"]
    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"].keys():
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"].keys():
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period

    return n


def add_retrofit_gas_boiler_constraint(n, snapshots):
    """
    Allow retrofitting of existing gas boilers to H2 boilers.
    """
    c = "Link"
    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    p_gas = dispatch.sel(Link=gas_i)
    p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")


def prepare_network(
    n,
    solve_opts=None,
    config=None,
    foresight=None,
    planning_horizons=None,
    co2_sequestration_potential=None,
):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if load_shedding := solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.add(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("curtailment_mode"):
        n.add("Carrier", "curtailment", color="#fedfed", nice_name="Curtailment")
        n.generators_t.p_min_pu = n.generators_t.p_max_pu
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            suffix=" curtailment",
            bus=buses_i,
            p_min_pu=-1,
            p_max_pu=0,
            marginal_cost=-0.1,
            carrier="curtailment",
            p_nom=1e6,
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "myopic":
        add_land_use_constraint(n)

    if foresight == "perfect":
        n = add_land_use_constraint_perfect(n)
        if snakemake.params["sector"]["limit_max_growth"]["enable"]:
            n = add_max_growth(n)

    if n.stores.carrier.eq("co2 sequestered").any():
        limit_dict = co2_sequestration_potential
        add_co2_sequestration_limit(n, limit_dict=limit_dict)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24h]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """
    agg_p_nom_minmax = pd.read_csv(
        config["solving"]["agg_p_nom_limits"]["file"], index_col=[0, 1], header=[0, 1]
    )[snakemake.wildcards.planning_horizons]
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable").rename_axis(index="Generator-ext")
    if config["solving"]["agg_p_nom_limits"]["agg_offwind"]:
        rename_offwind = {
            "offwind-ac": "offwind-all",
            "offwind-dc": "offwind-all",
            "offwind": "offwind-all",
        }
        gens = gens.replace(rename_offwind)
    grouper = pd.concat([gens.bus.map(n.buses.country), gens.carrier], axis=1)
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    if config["solving"]["agg_p_nom_limits"]["include_existing"]:
        gens_cst = n.generators.query("~p_nom_extendable").rename_axis(
            index="Generator-cst"
        )
        gens_cst = gens_cst[
            (gens_cst["build_year"] + gens_cst["lifetime"])
            >= int(snakemake.wildcards.planning_horizons)
        ]
        if config["solving"]["agg_p_nom_limits"]["agg_offwind"]:
            gens_cst = gens_cst.replace(rename_offwind)
        rhs_cst = (
            pd.concat(
                [gens_cst.bus.map(n.buses.country), gens_cst[["carrier", "p_nom"]]],
                axis=1,
            )
            .groupby(["bus", "carrier"])
            .sum()
        )
        rhs_cst.index = rhs_cst.index.rename({"bus": "country"})
        rhs_min = agg_p_nom_minmax["min"].dropna()
        idx_min = rhs_min.index.join(rhs_cst.index, how="left")
        rhs_min = rhs_min.reindex(idx_min).fillna(0)
        rhs = (rhs_min - rhs_cst.reindex(idx_min).fillna(0).p_nom).dropna()
        rhs[rhs < 0] = 0
        minimum = xr.DataArray(rhs).rename(dim_0="group")
    else:
        minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")

    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    if config["solving"]["agg_p_nom_limits"]["include_existing"]:
        rhs_max = agg_p_nom_minmax["max"].dropna()
        idx_max = rhs_max.index.join(rhs_cst.index, how="left")
        rhs_max = rhs_max.reindex(idx_max).fillna(0)
        rhs = (rhs_max - rhs_cst.reindex(idx_max).fillna(0).p_nom).dropna()
        rhs[rhs < 0] = 0
        maximum = xr.DataArray(rhs).rename(dim_0="group")
    else:
        maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")

    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24h]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (
            p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
        ).sum("Generator")

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_variable = n.model["Generator-p_nom"].rename(
        {"Generator-ext": "Generator"}
    )
    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n):
    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    def get_backward_i(forward_i):
        return pd.Index(
            [
                (
                    re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                    if re.search(r"-\d{4}$", s)
                    else s + "-reversed"
                )
                for s in forward_i
            ]
        )

    backward_i = get_backward_i(forward_i)

    lhs = n.model["Link-p_nom"].loc[backward_i]
    rhs = n.model["Link-p_nom"].loc[forward_i]

    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_flexible_egs_constraint(n):
    """
    Upper bounds the charging capacity of the geothermal reservoir according to
    the well capacity.
    """
    well_index = n.links.loc[n.links.carrier == "geothermal heat"].index
    storage_index = n.storage_units.loc[
        n.storage_units.carrier == "geothermal heat"
    ].index

    p_nom_rhs = n.model["Link-p_nom"].loc[well_index]
    p_nom_lhs = n.model["StorageUnit-p_nom"].loc[storage_index]

    n.model.add_constraints(
        p_nom_lhs <= p_nom_rhs,
        name="upper_bound_charging_capacity_of_geothermal_reservoir",
    )


def add_co2_atmosphere_constraint(n, snapshots):
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")

# def system_res_constraints(n, year, config) -> None:
#     """
#     Set a system-wide national RES constraints based on NECPs.

#     Here CI load is not counted within country_load ->
#     this avoids avoid big overshoot of national RES targets due to CI-procured portfolio. Note that EU RE directive counts corporate PPA within NECPs.
#     """
#     country_targets = config["RES_target"][year]

#     grid_res_techs = config["global"]["grid_res_techs"]
#     weights = n.snapshot_weightings["generators"]

#     for ct in country_targets.keys():
#         country_buses = n.buses.index[(n.buses.index.str[:2] == ct)]
#         if country_buses.empty:
#             continue

#         country_loads = n.loads.index[n.loads.bus.isin(country_buses) & n.loads.carrier.isin(['electricity'])]
#         country_res_gens = n.generators.index[
#             n.generators.bus.isin(country_buses)
#             & n.generators.carrier.isin(grid_res_techs)
#         ]
#         country_res_links = n.links.index[
#             n.links.bus1.isin(country_buses) & n.links.carrier.isin(grid_res_techs)
#         ]
#         country_res_storage_units = n.storage_units.index[
#             n.storage_units.bus.isin(country_buses)
#             & n.storage_units.carrier.isin(grid_res_techs)
#         ]

#         gens = n.model["Generator-p"].loc[:, country_res_gens] * weights
#         links = (
#             n.model["Link-p"].loc[:, country_res_links]
#             * n.links.loc[country_res_links, "efficiency"]
#             * weights
#         )
#         sus = (
#             n.model["StorageUnit-p_dispatch"].loc[:, country_res_storage_units]
#             * weights
#         )
#         lhs = gens.sum() + sus.sum() + links.sum()

#         target = config["RES_target"][year][f"{ct}"]
#         total_load = (n.loads_t.p_set[country_loads].sum(axis=1) * weights).sum()

#         logger.info(
#             f"country RES constraint for {ct} {target} and total load {round(total_load/1e6, 2)} TWh"
#         )

#         n.model.add_constraints(
#             lhs == target * total_load, name=f"{ct}_res_constraint"
#         )

# def cfe_constraints(n, penetration):
#     weights = n.snapshot_weightings["generators"]
#     vls = n.links[n.links.carrier == "virtual_link"]
#     dsm = n.links[n.links.carrier == "dsm"]

#     for location, name in datacenters.items():
#         # LHS
#         clean_gens = [name + " " + g for g in clean_techs]
#         storage_dischargers = [name + " " + g for g in storage_discharge_techs]
#         storage_chargers = [name + " " + g for g in storage_charge_techs]

#         gen_sum = (n.model["Generator-p"].loc[:, clean_gens] * weights).sum()
#         discharge_sum = (
#             n.model["Link-p"].loc[:, storage_dischargers]
#             * n.links.loc[storage_dischargers, "efficiency"]
#             * weights
#         ).sum()
#         charge_sum = (
#             -1 * (n.model["Link-p"].loc[:, storage_chargers] * weights).sum()
#         )

#         ci_export = n.model["Link-p"].loc[:, [name + " export"]]
#         ci_import = n.model["Link-p"].loc[:, [name + " import"]]
#         grid_sum = (
#             (-1 * ci_export * weights)
#             + (
#                 ci_import
#                 * n.links.at[name + " import", "efficiency"]
#                 * grid_supply_cfe
#                 * weights
#             )
#         ).sum()  # linear expr

#         lhs = gen_sum + discharge_sum + charge_sum + grid_sum

#         # RHS
#         total_load = (n.loads_t.p_set[name + " load"] * weights).sum()

#         vls_snd = vls.query("bus0==@name").index
#         vls_rec = vls.query("bus1==@name").index
#         total_snd = (
#             n.model["Link-p"].loc[:, vls_snd] * weights
#         ).sum()  # NB sum over both axes
#         total_rec = (n.model["Link-p"].loc[:, vls_rec] * weights).sum()

#         dsm_delayin = dsm.query("bus0==@name").index
#         dsm_delayout = dsm.query("bus1==@name").index
#         total_delayin = (
#             n.model["Link-p"].loc[:, dsm_delayin] * weights
#         ).sum()  # NB sum over both axes
#         total_delayout = (n.model["Link-p"].loc[:, dsm_delayout] * weights).sum()

#         flex = penetration * (
#             total_rec - total_snd + total_delayout - total_delayin
#         )

#         n.model.add_constraints(
#             lhs - flex >= penetration * (total_load), name=f"CFE_constraint_{name}"
#         )

# def excess_constraints(n, config):
#     weights = n.snapshot_weightings["generators"]

#     for location, name in datacenters.items():
#         ci_export = n.model["Link-p"].loc[:, [name + " export"]]
#         excess = (ci_export * weights).sum()
#         total_load = (n.loads_t.p_set[name + " load"] * weights).sum()
#         share = config["ci"][
#             "excess_share"
#         ]  # 'sliding': max(0., penetration - 0.8)

#         n.model.add_constraints(
#             excess <= share * total_load, name=f"Excess_constraint_{name}"
#         )

# def DC_constraints(n):
#     "A general case when both spatial and temporal flexibility mechanisms are enabled"

#     flexibility = n.config["procurement"]["flexibility"]

#     delta = float(flexibility) / 100
#     weights = n.snapshot_weightings["generators"]
#     vls = n.links[n.links.carrier == "virtual_link"]
#     dsm = n.links[n.links.carrier == "dsm"]

#     for location, name in datacenters.items():
#         vls_snd = vls.query("bus0==@name").index
#         vls_rec = vls.query("bus1==@name").index
#         dsm_delayin = dsm.query("bus0==@name").index
#         dsm_delayout = dsm.query("bus1==@name").index

#         snd = n.model["Link-p"].loc[:, vls_snd].sum(dim=["Link"])
#         rec = n.model["Link-p"].loc[:, vls_rec].sum(dim=["Link"])
#         delayin = n.model["Link-p"].loc[:, dsm_delayin].sum(dim=["Link"])
#         delayout = n.model["Link-p"].loc[:, dsm_delayout].sum(dim=["Link"])

#         load = n.loads_t.p_set[name + " load"]
#         # requested_load = load + rec - snd
#         rhs_up = load * (1 + delta) - load
#         rhs_lo = load * (1 - delta) - load

#         n.model.add_constraints(
#             rec - snd + delayout - delayin <= rhs_up, name=f"DC-upper_{name}"
#         )
#         n.model.add_constraints(
#             rec - snd + delayout - delayin >= rhs_lo, name=f"DC-lower_{name}"
#         )

# def res_constraints(n, penetration):
#     weights = n.snapshot_weightings["generators"]

#     for location, name in datacenters.items():
#         res_gens = [name + " " + g for g in res_techs]
#         lhs = (n.model["Generator-p"].loc[:, res_gens] * weights).sum()
#         total_load = (n.loads_t.p_set[name + " load"] * weights).sum()

#         # Note equality sign
#         n.model.add_constraints(
#             lhs == penetration * total_load, name=f"RES_annual_matching_{name}"
#         )

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})
    if constraints["BAU"] and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if constraints["SAFE"] and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if constraints["CCL"] and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    if EQ_o := constraints["EQ"]:
        add_EQ_constraints(n, EQ_o.replace("EQ", ""))

    if {"solar-hsat", "solar"}.issubset(
        config["electricity"]["renewable_carriers"]
    ) and {"solar-hsat", "solar"}.issubset(
        config["electricity"]["extendable_carriers"]["Generator"]
    ):
        add_solar_potential_constraints(n, config)

    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    if n._multi_invest:
        add_carbon_constraint(n, snapshots)
        add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    else:
        add_co2_atmosphere_constraint(n, snapshots)

    if config["sector"]["enhanced_geothermal"]["enable"]:
        add_flexible_egs_constraint(n)

    ###################
    # Add procurement constraints here

    # if constraints.get("RES", False):
    #     year = n.params.planning_horizons
    #     system_res_constraints(n, year, config)

    # procurement_stratergy = config.get("procurement",{}).get("stratergy",False)
    # if procurement_stratergy and config["enable"].get("procurement", False):
    #     logger.info("Procurement stratergy model is activate")
    #     penetration = config["procurement"]["stratergy"][3:]

    #     if procurement_stratergy == "ref":
    #         logger.info("ref is selected")
    #     elif procurement_stratergy == "cfe":
    #         logger.info("setting CFE target of", penetration)
    #         cfe_constraints(n, penetration)
    #         excess_constraints(n, config)
    #         DC_constraints(n)
    #     elif procurement_stratergy == "res":
    #         logger.info("setting annual RES target of", penetration)
    #         res_constraints(n, penetration)
    #         excess_constraints(n, config)

        #============ add Emission based procurement stratergy here ===========#

    if n.params.custom_extra_functionality:
        source_path = n.params.custom_extra_functionality
        assert os.path.exists(source_path), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)


def check_objective_value(n, solving):
    check_objective = solving["check_objective"]
    if check_objective["enable"]:
        atol = check_objective["atol"]
        rtol = check_objective["rtol"]
        expected_value = check_objective["expected_value"]
        if not np.isclose(n.objective, expected_value, atol=atol, rtol=rtol):
            raise ObjectiveValueError(
                f"Objective value {n.objective} differs from expected value "
                f"{expected_value} by more than {atol}."
            )


# def calculate_grid_cfe(n, name: str, node: str, config) -> pd.Series:
#     """
#     Calculates the time-series of grid supply CFE score for each C&I consumer.

#     Args:
#     - n: pypsa network.
#     - name: name of a C&I consumer.
#     - node: location (node) of a C&I consumer.
#     - config: config.yaml settings

#     Returns:
#     - pd.Series: A pandas series containing the grid CFE supply score.
#     """
#     grid_buses = n.buses.index[
#         ~n.buses.index.str.contains(name) & ~n.buses.index.str.contains(node)
#     ]
#     country_buses = n.buses.index[n.buses.index.str.contains(node)]

#     clean_techs = pd.Index(config["global"]["grid_clean_techs"])
#     emitters = pd.Index(config["global"]["emitters"])

#     clean_grid_generators = n.generators.index[
#         n.generators.bus.isin(grid_buses) & n.generators.carrier.isin(clean_techs)
#     ]
#     clean_grid_links = n.links.index[
#         n.links.bus1.isin(grid_buses) & n.links.carrier.isin(clean_techs)
#     ]
#     clean_grid_storage_units = n.storage_units.index[
#         n.storage_units.bus.isin(grid_buses) & n.storage_units.carrier.isin(clean_techs)
#     ]
#     dirty_grid_links = n.links.index[
#         n.links.bus1.isin(grid_buses) & n.links.carrier.isin(emitters)
#     ]

#     clean_country_generators = n.generators.index[
#         n.generators.bus.isin(country_buses) & n.generators.carrier.isin(clean_techs)
#     ]
#     clean_country_links = n.links.index[
#         n.links.bus1.isin(country_buses) & n.links.carrier.isin(clean_techs)
#     ]
#     clean_country_storage_units = n.storage_units.index[
#         n.storage_units.bus.isin(country_buses)
#         & n.storage_units.carrier.isin(clean_techs)
#     ]
#     dirty_country_links = n.links.index[
#         n.links.bus1.isin(country_buses) & n.links.carrier.isin(emitters)
#     ]

#     clean_grid_gens = n.generators_t.p[clean_grid_generators].sum(axis=1)
#     clean_grid_ls = -n.links_t.p1[clean_grid_links].sum(axis=1)
#     clean_grid_sus = n.storage_units_t.p[clean_grid_storage_units].sum(axis=1)
#     clean_grid_resources = clean_grid_gens + clean_grid_ls + clean_grid_sus

#     dirty_grid_resources = -n.links_t.p1[dirty_grid_links].sum(axis=1)

#     # grid_cfe =  clean_grid_resources / n.loads_t.p[grid_loads].sum(axis=1)
#     # grid_cfe[grid_cfe > 1] = 1.

#     import_cfe = clean_grid_resources / (clean_grid_resources + dirty_grid_resources)
#     import_cfe = np.nan_to_num(import_cfe, nan=0.0)  # Convert NaN to 0

#     clean_country_gens = n.generators_t.p[clean_country_generators].sum(axis=1)
#     clean_country_ls = -n.links_t.p1[clean_country_links].sum(axis=1)
#     clean_country_sus = n.storage_units_t.p[clean_country_storage_units].sum(axis=1)
#     clean_country_resources = clean_country_gens + clean_country_ls + clean_country_sus

#     dirty_country_resources = -n.links_t.p1[dirty_country_links].sum(axis=1)

#     ##################
#     # Country imports |
#     # NB lines and links are bidirectional, thus we track imports for both subsets
#     # of interconnectors: where [country] node is bus0 and bus1. Subsets are exclusive.

#     line_imp_subsetA = n.lines_t.p1.loc[:, n.lines.bus0.str.contains(node)].sum(axis=1)
#     line_imp_subsetB = n.lines_t.p0.loc[:, n.lines.bus1.str.contains(node)].sum(axis=1)
#     line_imp_subsetA[line_imp_subsetA < 0] = 0.0
#     line_imp_subsetB[line_imp_subsetB < 0] = 0.0

#     links_imp_subsetA = (
#         n.links_t.p1.loc[
#             :,
#             (
#                 n.links.bus0.str.contains(node)
#                 & (n.links.carrier == "DC")
#                 & ~(n.links.index.str.contains(name))
#             ),
#         ]
#         .clip(lower=0)
#         .sum(axis=1)
#     )

#     links_imp_subsetB = (
#         n.links_t.p0.loc[
#             :,
#             (
#                 n.links.bus1.str.contains(node)
#                 & (n.links.carrier == "DC")
#                 & ~(n.links.index.str.contains(name))
#             ),
#         ]
#         .clip(lower=0)
#         .sum(axis=1)
#     )

#     country_import = (
#         line_imp_subsetA + line_imp_subsetB + links_imp_subsetA + links_imp_subsetB
#     )

#     grid_supply_cfe = (clean_country_resources + country_import * import_cfe) / (
#         clean_country_resources + dirty_country_resources + country_import
#     )

#     print(f"Grid_supply_CFE for {node} has following stats:")
#     print(grid_supply_cfe.describe())

#     return grid_supply_cfe


def solve_network(n, config, params, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.params = params

    if rolling_horizon and snakemake.rule == "solve_operations_network":
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    # elif config["enable"].get("procurement", False):
    #     n_iterations = cf_solving["min_iterations"]
    #     values = [f"iteration {i}" for i in range(n_iterations + 1)]

    #     ci_names = config["procurement"]["ci"].keys()
    #     ci_locations = [config["procurement"]["ci"][ci_name]["location"] for ci_name in ci_names]
        
    #     cols = pd.MultiIndex.from_tuples(create_tuples(ci_locations, values))
    #     grid_cfe_df = pd.DataFrame(0.0, index=n.snapshots, columns=cols)
    #     for i in range(n_iterations):
    #         for ci_location in ci_locations:
    #             grid_supply_cfe = grid_cfe_df.loc[:, (ci_location, f"iteration {i}")]
    #             logger.info(grid_supply_cfe.describe())

    #         status, condition = n.optimize(**kwargs)

    #         for ci_name in ci_names:
    #             ci_location = config["procurement"]["ci"][ci_name]["location"]
    #             grid_cfe_df.loc[:, (f"{ci_location}", f"iteration {i + 1}")] = (
    #                 calculate_grid_cfe(n, name=ci_name, node=ci_location, config=config)
    #             )
    #     grid_cfe_df.to_csv(snakemake.output.grid_cfe)
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = cf_solving["track_iterations"]
        kwargs["min_iterations"] = cf_solving["min_iterations"]
        kwargs["max_iterations"] = cf_solving["max_iterations"]
        if cf_solving["post_discretization"].pop("enable"):
            logger.info("Add post-discretization parameters.")
            kwargs.update(cf_solving["post_discretization"])
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if not rolling_horizon:
        if status != "ok":
            logger.warning(
                f"Solving status '{status}' with termination condition '{condition}'"
            )
        check_objective_value(n, solving)

    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


def strip_network(n: pypsa.Network, config: dict) -> None:
    """
    Removes unnecessary components from a pypsa network.

    Args:
    - n (pypsa.Network): The network object to be stripped.

    Returns:
    - None
    """
    ci_names = config["ci"].keys()
    ci_locations = [config["ci"][ci_name]["location"] for ci_name in ci_names]
    zone = set(n.buses.country[bus] for bus in ci_locations)

    # Perform queries and combine results into a single set
    bus_core = n.buses.query("country.isin(@zone)", engine="python").index.unique()
    combined_lines = n.lines.query("bus1.isin(@bus_core) | bus0.isin(@bus_core)", engine="python")
    combined_links = n.links.query("bus1.isin(@bus_core) | bus0.isin(@bus_core)", engine="python")
    
    # Combine the results of bus0 and bus1 in lines and links
    bus_connect = (set(combined_lines.bus0.unique()) | set(combined_lines.bus1.unique()) |
                   set(combined_links.bus0.unique()) | set(combined_links.bus1.unique()))
    
    zone_all = set(n.buses.country[bus] for bus in bus_connect)
    nodes_to_keep = n.buses.query("country.isin(@zone_all)").index.unique()

    n.remove("Bus", n.buses.index.symmetric_difference(nodes_to_keep))

    # make sure lines are kept
    n.lines.carrier = "AC"

    for c in n.iterate_components(
        ["Generator", "Link", "Line", "Store", "StorageUnit", "Load"]
    ):
        if c.name in ["Link", "Line"]:
            location_boolean = c.df.bus0.isin(nodes_to_keep) & c.df.bus1.isin(
                nodes_to_keep
            )
        else:
            location_boolean = c.df.bus.isin(nodes_to_keep)
        to_keep = c.df.index[location_boolean]
        to_drop = c.df.index.symmetric_difference(to_keep)
        n.remove(c.name, to_drop)


def load_profile(
    n: pypsa.Network,
    name: str,
    profile_shape: str,
    config,
) -> pd.Series:
    """
    Create daily load profile for C&I buyers based on config setting.

    Args:
    - n (object): object
    - profile_shape (str): shape of the load profile, must be one of 'baseload' or 'industry'
    - config (dict): config settings

    Returns:
    - pd.Series: annual load profile for C&I buyers
    """

    procurement = config["procurement"]
    scaling = n.snapshot_weightings.objective.sum() / 8760.0  # 3/1 for 3H/1H

    shapes = {
        "baseload": [1 / 24] * 24,
        "industry": [0.009] * 5
        + [0.016, 0.031, 0.07, 0.072, 0.073, 0.072, 0.07]
        + [0.052, 0.054, 0.066, 0.07, 0.068, 0.063]
        + [0.035] * 2
        + [0.045] * 2
        + [0.009],
    }

    try:
        shape = shapes[profile_shape]
    except KeyError:
        print(
            f"'profile_shape' option must be one of 'baseload' or 'industry'. Now is {profile_shape}."
        )
        sys.exit()

    # CI consumer nominal load in MW
    if procurement["stratergy"] == "ref":
        load = 0.0
    else:
        load = procurement["ci"][name]["load"] * procurement["participation"] / 100

    load_day = load * 24  # 24h
    load_profile_day = pd.Series(shape) * load_day

    if scaling != 1.0:
        load_profile_day = load_profile_day.groupby(
            np.arange(len(load_profile_day)) // scaling
        ).mean()  # 3H sampling

    load_profile_year = pd.concat([load_profile_day] * 365)
    profile = load_profile_year.set_axis(n.snapshots)

    return profile


def add_ci(n: pypsa.Network, year: str, config: dict, costs: pd.DataFrame) -> None:
    """
    Add C&I buyer(s) to the network.

    Args:
    - n: pypsa.Network to which the C&I buyer(s) will be added.
    - year: the year of optimisation based on config setting.

    Returns:
    - None
    """
    # tech_palette options
    procurement = config["procurement"]
    clean_techs = procurement["technology"]["generation_tech"]
    storage_techs = procurement["technology"]["storage_tech"]
    ci = procurement["ci"]
    stratergy = procurement["stratergy"]
    max_hours = config["max_hours"]

    for name in ci.keys():
        location = ci[name]["location"]
        profile = ci[name]["profile"]

        n.add("Bus", name)

        n.add(
            "Link",
            f"{name}" + " export",
            bus0=name,
            bus1=location,
            marginal_cost=0.1,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
        )

        n.add(
            "Link",
            f"{name}" + " import",
            bus0=location,
            bus1=name,
            marginal_cost=0.001,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
        )

        n.add(
            "Load",
            f"{name}" + " load",
            carrier="electricity",
            bus=name,
            p_set=load_profile(n, name, profile, config),
        )

        # C&I following voluntary clean energy procurement is a share of C&I load -> subtract it from node's profile
        n.loads_t.p_set[location] -= n.loads_t.p_set[f"{name}" + " load"]

        # Add clean firm advanced generators
        #if "green hydrogen OCGT" in clean_techs:
        #    n.add(
        #        "Generator",
        #        f"{name} green hydrogen OCGT",
        #        carrier="green hydrogen OCGT",
        #        bus=name,
        #        p_nom_extendable=True if stratergy == "cfe" else False,
        #        capital_cost=costs.at["OCGT", "fixed"],
        #        marginal_cost=costs.at["OCGT", "VOM"]
        #        + snakemake.config["costs"]["price_green_hydrogen"]
        #        / 0.033
        #        / costs.at["OCGT", "efficiency"],
        #    )
            # hydrogen cost in EUR/kg, 0.033 MWhLHV/kg

        if "nuclear" in clean_techs:
            n.add(
                "Generator",
                f"{name} nuclear",
                bus=name,
                carrier="nuclear",
                capital_cost=costs.loc["nuclear"]["fixed"],
                marginal_cost=costs.loc["nuclear"]["VOM"]
                + costs.loc["nuclear"]["fuel"]
                / costs.loc["nuclear"]["efficiency"],
                p_nom_extendable=True if stratergy == "cfe" else False,
                lifetime=costs.loc["nuclear"]["lifetime"],
            )

        if "allam" in clean_techs:
            n.add(
                "Generator",
                f"{name} allam",
                bus=name,
                carrier="gas",
                capital_cost=costs.at["allam", "fixed"],
                marginal_cost=costs.loc["allam"]["VOM"]
                + costs.loc["gas"]["fuel"] / costs.loc["allam"]["efficiency"],
                #+ 0.02 * costs.at["gas", "CO2 intensity"] / costs.loc["allam"]["efficiency"], # 98% of CO2 is captured
                p_nom_extendable=True if stratergy == "cfe" else False,
                lifetime=costs.loc["allam"]["lifetime"],
                efficiency=costs.loc["allam"]["efficiency"],
            )

        if "geothermal" in clean_techs:
            n.add(
                "Generator",
                f"{name} geothermal",
                bus=name,
                # carrier = '',
                capital_cost=10000, # TODO: Intergrate geothermal cost to model and config
                marginal_cost=costs.loc["geothermal"]["VOM"],
                p_nom_extendable=True if stratergy == "cfe" else False,
                lifetime=costs.at["geothermal", "lifetime"],
            )

        # Add RES generators
        for carrier in ["onwind", "solar"]:
            if carrier not in clean_techs:
                continue
            gen_template = location + " " + carrier + f"-{year}"

            n.add(
                "Generator",
                f"{name} {carrier}",
                carrier=carrier,
                bus=name,
                p_nom_extendable=False if stratergy == "ref" else True,
                p_max_pu=n.generators_t.p_max_pu[gen_template],
                capital_cost=n.generators.at[gen_template, "capital_cost"],
                marginal_cost=n.generators.at[gen_template, "marginal_cost"],
            )

        # =================== Add storage techs =================== 
        
        # check for not implemented storage technologies
        implemented = ["H2", "li-ion battery", "iron-air battery", "lfp", "vanadium", "lair", "pair"]
        not_implemented = list(set(storage_techs).difference(implemented))
        available_carriers = list(set(storage_techs).intersection(implemented))
        if len(not_implemented) > 0:
            logger.warning(
                f"{not_implemented} are not yet implemented as Storage technologies in PyPSA-Eur"
            )
        missing_carriers = list(set(available_carriers).difference(n.carriers.index))
        n.add("Carrier", missing_carriers)

        lookup_store = {"H2": "electrolysis", "li-ion battery": "battery inverter", "iron-air battery": "iron-air battery charge",
        "lfp": "Lithium-Ion-LFP-bicharger", "vanadium": "Vanadium-Redox-Flow-bicharger", "lair":  "Liquid-Air-charger", "pair": "Compressed-Air-Adiabatic-bicharger"}
        lookup_dispatch = {"H2": "fuel cell", "li-ion battery": "battery inverter", "iron-air battery": "iron-air battery discharge",
        "lfp": "Lithium-Ion-LFP-bicharger", "vanadium": "Vanadium-Redox-Flow-bicharger", "lair":  "Liquid-Air-discharger", "pair": "Compressed-Air-Adiabatic-bicharger"}

        for carrier in available_carriers:
            for max_hour in max_hours[carrier]:
                roundtrip_correction = 0.5 if carrier == "li-ion battery" else 1
                cost_carrier = "H2 tank" if carrier == "H2" else carrier
                cost_carrier = "iron-air battery storage" if carrier == "iron-air battery" else cost_carrier
                
                n.add(
                    "StorageUnit",
                    f"{name} {carrier}",
                    suffix=f" {carrier} {max_hour}h",
                    bus=name,
                    carrier=carrier,
                    p_nom_extendable=True,
                    capital_cost=costs.at[f"{cost_carrier} {max_hour}h", "fixed"],
                    marginal_cost=0., # TODO: Set this adjustable?
                    efficiency_store=costs.at[lookup_store[carrier], "efficiency"]
                    ** roundtrip_correction,
                    efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"]
                    ** roundtrip_correction,
                    max_hours=max_hour,
                    cyclic_state_of_charge=True,
                    lifetime=costs.at[f"{cost_carrier} {max_hour}h", "lifetime"],
                )


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network_perfect",
            configfiles="../config/test/config.perfect.yaml",
            opts="",
            clusters="5",
            ll="v1.0",
            sector_opts="",
            # planning_horizons="2030",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
        config=snakemake.config,
        foresight=snakemake.params.foresight,
        planning_horizons=snakemake.params.planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        #############################
        # Temporary, place the preparation for procurement here

        if snakemake.params.procurement_enable:
            print("procurement_enable is activated")
            procurement = snakemake.params.procurement


            if procurement["strip_network"]:
                print("stript_network is activated")
                strip_network(n, procurement)

            Nyears = n.snapshot_weightings.generators.sum() / 8760.0

            # TODO: (DONE) Modify the prepare_costs to apply additions in max hours
            costs = prepare_costs(
                snakemake.input.costs,
                snakemake.params,
                Nyears,
            )
            # TODO: Expand the C&I scope for renewable energy in neighbouring nodes
            add_ci(n, 
                   snakemake.wildcards.planning_horizons, 
                   snakemake.params,
                   costs
                   )

        n = solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            log_fn=snakemake.log.solver,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
