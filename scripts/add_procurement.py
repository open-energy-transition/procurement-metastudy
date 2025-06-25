# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
This script is part of a PyPSA-Eur workflow that adds Commercial & Industrial (C&I) electricity consumers 
and their clean energy procurement strategies to a power system model.
"""
import logging
import sys

import country_converter as coco
import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import (
    configure_logging,
    get,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import add_missing_carriers, load_costs

cc = coco.CountryConverter()

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

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
    bus_core = n.buses[n.buses["country"].isin(zone)].index.unique()
    combined_lines = n.lines[n.lines.bus1.isin(bus_core) | n.lines.bus0.isin(bus_core)]
    combined_links = n.links[n.links.bus1.isin(bus_core) | n.links.bus0.isin(bus_core)]

    # Combine the results of bus0 and bus1 in lines and links
    bus_connect = (
        set(combined_lines.bus0.unique())
        | set(combined_lines.bus1.unique())
        | set(combined_links.bus0.unique())
        | set(combined_links.bus1.unique())
    )

    zone_all = set(n.buses.country[bus] for bus in bus_connect)
    nodes_to_keep = n.buses[n.buses["country"].isin(zone_all)].index.unique()

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


def retrieve_ci_load(config):
    load = config["electricity"]["ci_load"]

    # 1 EUROSTAT data in GWh
    import requests
    import os
    url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/nrg_cb_e/1.0/*.*.*.*.*?c[freq]=A&c[nrg_bal]=FC,FC_IND_E,FC_OTH_CP_E&c[siec]=E7000&c[unit]=GWH&c[geo]=EU27_2020,EA20,BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,HR,IT,CY,LV,LT,LU,HU,MT,NL,AT,PL,PT,RO,SI,SK,FI,SE,IS,LI,NO,UK,BA,ME,MD,MK,GE,AL,RS,TR,UA,XK&c[TIME_PERIOD]=2023,2022,2021,2020&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name"
    file_path = load["load_path_1"]
    
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
    else:
        try:
            response = requests.get(url)
            with open(file_path, "wb") as file:
                file.write(response.content)
            data = pd.read_csv(file_path)
        except requests.ConnectionError:
            logger.warning("No internet connection and file not found locally.")
            raise FileNotFoundError(f"File {file_path} not found and cannot download from the internet.")

    # Ensure data for the specified year exists for all countries
    data["reference_year"] = int(load["load_year"])
    years = list(data["TIME_PERIOD"].unique())
    years.reverse()
    for geo in data["geo"].unique():
        if not (
            (data["TIME_PERIOD"] == int(load["load_year"])) & (data["geo"] == geo)
        ).any():
            for fallback_year in years[1:]:
                if (
                    (data["TIME_PERIOD"] == fallback_year) & (data["geo"] == geo)
                ).any():
                    fallback_row = data[
                        (data["TIME_PERIOD"] == fallback_year) & (data["geo"] == geo)
                    ].copy()
                    fallback_row["TIME_PERIOD"] = int(load["load_year"])
                    data = pd.concat([data, fallback_row], ignore_index=True)
                    data.loc[
                        (data["TIME_PERIOD"] == int(load["load_year"]))
                        & (data["geo"] == geo),
                        "reference_year",
                    ] = fallback_year
                    break

    filtered_data = data[(data["TIME_PERIOD"] == int(load["load_year"]))]

    demand_map = {
        "FC": "total_demand",
        "FC_IND_E": "industrial_demand",
        "FC_OTH_CP_E": "commercial_demand",
    }
    dfs = []
    for code, label in demand_map.items():
        df_part = (
            filtered_data[filtered_data["nrg_bal"] == code][
                ["geo", "OBS_VALUE", "reference_year"]
            ]
            .rename(columns={"geo": "country", "OBS_VALUE": label})
            .groupby("country")
            .sum()
        )
        dfs.append(df_part)

    # Merge all load data into a single DataFrame
    load_year_eurostat = pd.concat(dfs, axis=1).loc[
        :, ~pd.concat(dfs, axis=1).columns.duplicated()
    ]  # remove reference_year duplicatated columns
    load_year_eurostat.rename(
        index={"EL": "GR"}, inplace=True
    )  # Rename EL (Eurostat) to GR (PyPSA)
    load_year_eurostat["ci_demand"] = (
        load_year_eurostat["industrial_demand"]
        + load_year_eurostat["commercial_demand"]
    )
    load_year_eurostat["ci_share"] = (
        load_year_eurostat["ci_demand"] / load_year_eurostat["total_demand"]
    )

    # 2 IEA data for Switzerland (CH) and Great Britain (GB) in PJ
    years = list(range(1971, 2023)) + [str(2023) + " Provisional"]
    country_map = {"Switzerland": "CH", "United Kingdom": "GB"}
    demand_map = {
        "Total final consumption (PJ)": "total_demand",
        "Industry (PJ)": "industrial_demand",
        "Commercial and public services (PJ)": "commercial_demand",
    }
    data = pd.read_excel(
        load["load_path_2"], sheet_name="TimeSeries_1971-2023", skiprows=1
    )  #
    filtered_data = data[
        (data["Product"] == "Electricity") & (data["Country"].isin(country_map.keys()))
    ]

    # Determine the most recent available year in the 'iea' DataFrame
    for y in years[::-1]:
        if (
            y in filtered_data.columns and not (filtered_data[y] == "..").any()
        ):  # Check if at least one value is '..'
            most_recent_year = y
            break
    filtered_data = filtered_data[["Country", "Flow", most_recent_year]]
    dfs = []
    for code, label in demand_map.items():
        df_part = (
            filtered_data[filtered_data["Flow"] == code][["Country", most_recent_year]]
            .rename(columns={"Country": "country", most_recent_year: label})
            .groupby("country")
            .sum()
            * 1
            / 0.0036  # Convert PJ to GWh
        )
        dfs.append(df_part)

    # Merge all load data into a single DataFrame
    load_year_missing = pd.concat(dfs, axis=1).rename(index=country_map)
    load_year_missing["reference_year"] = most_recent_year
    load_year_missing["ci_demand"] = (
        load_year_missing["industrial_demand"] + load_year_missing["commercial_demand"]
    )
    load_year_missing["ci_share"] = (
        load_year_missing["ci_demand"] / load_year_missing["total_demand"]
    )

    # 3 Merge Eurostat and IEA data
    load_year_countries = pd.concat([load_year_eurostat, load_year_missing], axis=0)

    return load_year_countries


def load_profile(
    n: pypsa.Network,
    load_year: pd.DataFrame,
    config,
    location: str,
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
    load = config["electricity"]["ci_load"]
    scaling = n.snapshot_weightings.objective.sum() / len(
        n.snapshot_weightings.objective
    )  # e.g., 3 for 3H time resolution

    shapes = {
        "baseload": [1 / 24] * 24,
        "industry": [0.009] * 5
        + [0.016, 0.031, 0.07, 0.072, 0.073, 0.072, 0.07]
        + [0.052, 0.054, 0.066, 0.07, 0.068, 0.063]
        + [0.035] * 2
        + [0.045] * 2
        + [0.009] * 2,
        "total_daily_avg": "total_daily_avg",
        "total": "total",
    }

    try:
        shape = shapes[load["profile"]]
    except KeyError:
        print(
            f"'profile_shape' option must be one of 'baseload', 'industry', 'total_daily_avg', or 'total'. Now is {load['profile']}."
        )
        sys.exit()

    # CI consumer nominal load in MW
    
    load_year_val = (
                load_year["ci_share"].values[0]
                * (n.loads_t.p_set[location] * n.snapshot_weightings.objective).sum()
            )  # MWh
    if (snakemake.params.get("procurement_enable", False) and location in [v["location"] for v in procurement["ci"].values()]):
        print("procurement_enable is activated")
        logger.info(
            f"CI load in {load_year.index.values[0]} (raw data from Eurostat/IEA):\nannual consumption: {round((load_year['total_demand'].values[0]) / 1000)} TWh\nreference raw data year: {load_year['reference_year'].values[0]}\nshare: {round(load_year['ci_share'].values[0] * 100, 0)}%"
        )
        logger.info(
            f"CI load in {load_year.index.values[0]} (PyPSA data):\nannual consumption {round(load_year_val / 10**6)} TWh\nreference config year: {load['load_year']}"
        )
        logger.info(
            f"Only {load['share']}% of the total CI load is moved to high voltage side, which corresponds to:\nannual consumption {round((load_year_val / 10**6) * load['share'] / 100)} TWh\nreference config year: {load['load_year']}"
        )

    if procurement["strategy"] == "ref":
        profile = pd.Series(0, index = n.snapshots)
    else:
        if shape == "total":
            profile = load['share'] / 100 * load_year["ci_share"].values[0] * n.loads_t.p_set[location]
        elif shape == "total_daily_avg":
            total_daily_avg = n.loads_t.p_set[location].resample('D').mean()
            CI_daily_avg = load_year["ci_share"].values[0] * total_daily_avg
            profile = load['share'] / 100 * CI_daily_avg.reindex(n.snapshots, method="ffill")
        else:
            load = load['share'] / 100 * load_year_val / 8760 # MW

            load_day = load * 24
            load_profile_day = pd.Series(shape) * load_day
            load_profile_year = pd.concat([load_profile_day] * 365)

            if scaling != 1.0:
                load_profile_year.index = pd.date_range(
                    start=n.snapshots[0], periods=len(load_profile_year), freq="h"
                )
                profile = (
                    load_profile_year.resample(f"{int(scaling)}h")
                    .mean()
                    .reindex(n.snapshots, method="nearest")
                )
            else:
                profile = load_profile_year.set_axis(n.snapshots)

    return profile

def add_ci_load(n: pypsa.Network, config: dict) -> None:
    """
    Add C&I buyer(s) to the network.

    Args:
    - n: pypsa.Network to which the C&I buyer(s) will be added.

    Returns:
    - None
    """

    load_year_countries = retrieve_ci_load(config)  # retrieve CI load input data

    countries_chosen = []

    for bus in [loc for loc in n.buses.location.unique() if loc != "EU"]:
        
        country = n.buses.country[bus]
        if country in countries_chosen:
            continue
        countries_chosen.append(country)

        load_year = load_year_countries[
            load_year_countries.index == country
        ]  # select only the country of interest

        n.add("Bus",
              f"{bus}" + " CI",
              country=n.buses.loc[bus,"country"],
              location=bus,
              x=n.buses.loc[bus,"x"],
              y=n.buses.loc[bus,"y"],
              )

        n.add(
            "Link",
            f"{bus}" + " CI export",
            bus0=f"{bus}" + " CI",
            bus1=bus,
            marginal_cost=0.1,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
            reversed=False,
        )

        n.add(
            "Link",
            f"{bus}" + " CI import",
            bus0=bus,
            bus1=f"{bus}" + " CI",
            marginal_cost=0.001,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
            reversed=False,
        )

        n.add(
            "Load",
            f"{bus}" + " CI load",
            carrier="electricity",
            bus=f"{bus}" + " CI",
            p_set=load_profile(n, load_year, config, bus),
            ci="None"  # C&I markers used in constraints
        )

        # C&I following voluntary clean energy procurement is a share of C&I load -> subtract it from node's profile
        n.loads_t.p_set[bus] -= n.loads_t.p_set[f"{bus}" + " CI" + " load"]

    ci_load_cols = n.loads_t.p_set.filter(like="CI").columns
    non_ci_load = n.loads_t.p_set.loc[:, ~n.loads_t.p_set.columns.isin(ci_load_cols)]
    # Check for negative background load values
    negative_indices = non_ci_load.columns[(non_ci_load.min() < 0)].tolist()
    if negative_indices:
        logger.warning(
            f"Negative background load values found during some snapshots for: {negative_indices}."
        )

def add_ci_procurement(n: pypsa.Network, year: str, config: dict, costs: pd.DataFrame) -> None:
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
    strategy = procurement["strategy"]
    scope = procurement["scope"]

    for name in ci.keys():
        location = ci[name]["location"]
        participation = procurement["participation"]

        # ===================== Adding CI load to be supplied ========================
        # ============================================================================
        n.add("Bus",
              name,
              country= n.buses.country[location],
              location=location,
              x=n.buses.loc[location,"x"],
              y=n.buses.loc[location,"y"],
              )

        n.add(
            "Load",
            f"{name}" + " load",
            carrier="electricity",
            bus=name,
            p_set=n.loads_t.p_set[f"{location}" + " CI load"] * participation / 100,
            ci=name # C&I markers used in constraints
        )

        if participation == 100: 
            n.remove("Bus", f"{location}" + " CI")
            n.remove("Load",f"{location}" + " CI load")
            n.remove("Link",f"{location}" + " CI export")
            n.remove("Link",f"{location}" + " CI import")
        else:
            n.loads_t.p_set[f"{location}" + " CI load"] *= (100 - participation) / 100

        n.add(
            "Link",
            f"{name}" + " export",
            bus0=name,
            bus1=location,
            marginal_cost=0.1,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
            reversed=False,
        )

        n.add(
            "Link",
            f"{name}" + " import",
            bus0=location,
            bus1=name,
            marginal_cost=0.001,  # large enough to avoid optimization artifacts, small enough not to influence PPA portfolio
            p_nom=1e6,
            reversed=False,
        )

        # Scope Definition

        if scope == "node" or strategy == "247-cfe":
            scope = "node"
            bus = [location]
        elif scope == "country":
            zone = n.buses.loc[location, "country"]
            bus = n.buses[n.buses.country == zone].location.unique()
        else:  # scope == "all" is the default
            bus = [loc for loc in n.buses.location.unique() if loc != "EU"]

        # ===================== Adding Dispatchable Technologies =====================
        # ============================================================================

        gen_implemented = {
            "nuclear": {
                "carrier": "uranium",
                "carrier_nodes": "EU uranium",
                "unit": "MWh_th",
            },
            "allam": {
                "carrier": "gas",
                "carrier_nodes": "EU gas",
                "unit": "MWh_LHV",
            },
            "geothermal": {
                "carrier": "geothermal",
                "carrier_nodes": "EU enhanced geothermal systems",
                "unit": "MWh_th",
            },
        }
        gen_not_implemented = list(
            set(clean_techs).difference(
                list(gen_implemented.keys()) + ["onwind", "offwind-ac", "offwind-dc", "offwind-float", "solar", "solar-hsat", "solar rooftop"]
            )
        )
        gen_available_carriers = list(
            set(clean_techs).intersection(gen_implemented.keys())
        )
        if len(gen_not_implemented) > 0:
            logger.warning(
                f"{gen_not_implemented} are not yet implemented as Clean technologies for CI in PyPSA-Eur"
            )

        for generator in gen_available_carriers:
            carrier = gen_implemented[generator]["carrier"]
            carrier_nodes = gen_implemented[generator]["carrier_nodes"]

            if carrier_nodes not in n.buses.index:
                logger.info(f"Missing buses: {carrier_nodes}. Adding them now for CI")
                n.add(
                    "Bus",
                    carrier_nodes,
                    carrier=carrier,
                    location="EU",
                    unit=gen_implemented[generator]["unit"],
                )

                n.add(
                    "Generator",
                    carrier_nodes,
                    bus=carrier_nodes,
                    carrier=carrier,
                    p_nom_extendable=True,
                )

            if scope == "node":
                gen_df = pd.DataFrame({"bus1": [name]}, index=[name + " " + generator])
            else:
                index_labels = [f"{name} {b} {generator}" for b in bus]
                gen_df = pd.DataFrame({"bus1": bus}, index=index_labels)

            n.add(
                "Link",
                gen_df.index,
                bus0=carrier_nodes,
                bus1=gen_df.bus1,
                bus2="co2 atmosphere",
                marginal_cost=costs.at[generator, "efficiency"]
                * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                capital_cost=costs.at[generator, "efficiency"]
                * costs.at[generator, "capital_cost"],  # NB: fixed cost is per MWel
                p_nom_extendable=True if strategy else False,
                p_max_pu=0.7
                if carrier == "uranium"
                else 1,  # be conservative for nuclear (maintenance or unplanned shut downs)
                carrier=generator,
                efficiency=costs.at[generator, "efficiency"],
                efficiency2=costs.at[carrier, "CO2 intensity"]
                if generator != "allam"
                else 0.02 * costs.at[carrier, "CO2 intensity"],
                lifetime=costs.at[generator, "lifetime"],
                reversed=False,
                ci=name,  # C&I markers used in constraints
            )

        add_missing_carriers(n, gen_available_carriers)

        logger.info(f"Include {gen_available_carriers} for the CI: {name}.")

        # ===================== Adding Variable Renewable Technologies =====================
        # ==================================================================================

        res_available_carriers = list(
            set(clean_techs).intersection(["onwind", "offwind-ac", "offwind-dc", "offwind-float", 
                                           "solar", "solar-hsat", "solar rooftop"])
        )

        for carrier in res_available_carriers:
            bus_carrier = [b + " low voltage" for b in bus] if carrier == "solar rooftop" else bus

            mask = (
                n.generators.bus.isin(bus_carrier)
                & (n.generators.carrier == carrier)
                & (n.generators.index.astype(str).str.contains(year))
            )

            if "ci" in n.generators.columns:
                mask &= n.generators.ci.isin([np.NaN, ""])

            res_df = n.generators.loc[mask].copy()

            res_df["gen_name"] = "CI" + " " + res_df.index if scope == "continent" else name + " " + res_df.index
            res_df["bus_name"] = name if scope == "node" else res_df["bus"]
            res_df["capital_cost"] = n.generators.loc[res_df.index,"capital_cost"]
            res_df["marginal_cost"] = n.generators.loc[res_df.index,"marginal_cost"]

            p_max_pu_df = n.generators_t.p_max_pu[res_df.index]
            p_max_pu_df = p_max_pu_df.rename(columns=res_df["gen_name"].to_dict())

            res_df = res_df.set_index("gen_name")

            n.add(
                "Generator",
                res_df.index,
                carrier=carrier,
                bus=res_df["bus_name"],
                p_nom_extendable=True if strategy else False,
                p_max_pu=p_max_pu_df,
                capital_cost=res_df["capital_cost"],
                marginal_cost=res_df["marginal_cost"],
                ci="continent" if scope == "continent" else name,  # C&I markers used in constraints
            )

        logger.info(
            f"Include {res_available_carriers} for the CI: {name} with the scope: {scope}."
        )

        # ===================== Adding Storage Technologies =====================
        # =======================================================================

        max_hours = config["max_hours"]

        # check for not implemented storage technologies
        storage_implemented = [
            "H2",
            "li-ion battery",
            "iron-air battery",
            "lfp",
            "vanadium",
            "lair",
            "pair",
        ]
        storage_not_implemented = list(
            set(storage_techs).difference(storage_implemented)
        )
        storage_available_carriers = list(
            set(storage_techs).intersection(storage_implemented)
        )
        if len(storage_not_implemented) > 0:
            logger.warning(
                f"{storage_not_implemented} are not yet implemented as Storage technologies in PyPSA-Eur"
            )
        available_carriers_max_hours = [
            f"{carrier} {max_hour}h"
            for carrier in storage_available_carriers
            if carrier in max_hours
            for max_hour in max_hours[carrier]
        ]
        missing_carriers = list(
            set(available_carriers_max_hours).difference(n.carriers.index)
        )
        n.add("Carrier", missing_carriers)

        lookup_store = {
            "H2": "electrolysis",
            "li-ion battery": "battery inverter",
            "iron-air battery": "iron-air battery charge",
            "lfp": "Lithium-Ion-LFP-bicharger",
            "vanadium": "Vanadium-Redox-Flow-bicharger",
            "lair": "Liquid-Air-charger",
            "pair": "Compressed-Air-Adiabatic-bicharger",
        }
        lookup_dispatch = {
            "H2": "fuel cell",
            "li-ion battery": "battery inverter",
            "iron-air battery": "iron-air battery discharge",
            "lfp": "Lithium-Ion-LFP-bicharger",
            "vanadium": "Vanadium-Redox-Flow-bicharger",
            "lair": "Liquid-Air-discharger",
            "pair": "Compressed-Air-Adiabatic-bicharger",
        }

        for carrier in storage_available_carriers:
            for max_hour in max_hours[carrier]:
                roundtrip_correction = 0.5 if carrier == "li-ion battery" else 1
                cost_carrier = "H2 tank" if carrier == "H2" else carrier
                n.add(
                    "StorageUnit",
                    name,
                    suffix=f" {carrier} {max_hour}h",
                    bus=name,
                    carrier=f"{carrier} {max_hour}h",
                    p_nom_extendable=True if strategy else False,
                    capital_cost=costs.at[
                        f"{cost_carrier} {max_hour}h", "capital_cost"
                    ],
                    marginal_cost=0.0,
                    efficiency_store=costs.at[lookup_store[carrier], "efficiency"]
                    ** roundtrip_correction,
                    efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"]
                    ** roundtrip_correction,
                    max_hours=max_hour,
                    cyclic_state_of_charge=True,
                    lifetime=costs.at[f"{cost_carrier} {max_hour}h", "lifetime"],
                    ci=name,  # C&I markers used in constraints
                )

        logger.info(f"Include {storage_available_carriers} for the CI: {name}.")


def freeze_capacity(n, config):
    """
    Freeze capacities of expandable variable renewable generators
    """
    # Define variable renewable carriers
    res_carriers = [
        "solar", "solar-hsat", "solar rooftop", 
        "onwind", "offwind-ac", "offwind-dc", "offwind-float"
    ]

    # Freeze existing capacities for variable renewables
    mask_res = (
        n.generators.carrier.isin(res_carriers) & 
        (n.generators.p_nom_max != np.inf)
    )
    n.generators.loc[mask_res, "p_nom_extendable"] = False

    # Reallocate unused capacity to CI generators (if exist)
    if "ci" in n.generators.columns:
        logger.info("Freeze capacity activated — reallocating potential capacity to CI components")

        # Calculate remaining extendable capacity 
        remaining_cap = (
            n.generators.loc[mask_res, "p_nom_max"] - 
            n.generators.loc[mask_res, "p_nom"]
        )
        # Remove p_nom_max of existing capacities
        n.generators.loc[mask_res, "p_nom_max"] = 0

        # Candidate gen_ci: must be extendable and in res_carriers
        gen_ci = n.generators.loc[
            n.generators.carrier.isin(res_carriers) & 
            (n.generators.p_nom_extendable == True)
        ]

        # Extract (bus, carrier) pairs present in gen_ci
        gen_ci_keys = set(zip(gen_ci.bus, gen_ci.carrier))

        # Keep only (bus, carrier) pairs that exist in gen_ci
        df_remaining = n.generators.loc[mask_res, ["bus", "carrier"]]

        # Check if procurement scope is set to "node"
        if (
            config.get("procurement_enable", False) 
            and config.get("procurement", {}).get("scope") == "node"
        ):
            # Build a mapping from location codes to country names
            ci = config["procurement"]["ci"]
            ci_locations = {info.get('location'): name for name, info in ci.items()}
            df_remaining['bus'] = df_remaining['bus'].map(ci_locations)
            df_remaining = df_remaining.dropna(subset=['bus'])

        df_remaining["key"] = list(zip(df_remaining.bus, df_remaining.carrier))
        df_remaining = df_remaining[df_remaining["key"].isin(gen_ci_keys)]

        for idx, row in df_remaining.iterrows():
            bus = row["bus"]
            carrier = row["carrier"]
            rem_cap = remaining_cap.loc[idx]

            # Find matching gen_ci for that (bus, carrier)
            match = gen_ci.loc[
                (gen_ci.bus == bus) & 
                (gen_ci.carrier == carrier)
            ]

            if not match.empty:
                best_idx = match.sort_values("ci").index[0]
                n.generators.at[best_idx, "p_nom_max"] = rem_cap
                logger.info(f"Reassigned {rem_cap:.2f} MW from {idx} → {best_idx}")
    else:
        logger.info("Freeze capacity activated")

def filter_TYNDP_build_year(n, year):
    """
    Remove transmission with build year later than the planning horizon
    """
    links = n.links[(n.links.project_status != "") & (n.links.build_year > int(year))]

    logger.info(f"Remove transmission with build year later than {year}: \n{links[["build_year"]]}")

    n.remove("Link",links.index)

# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_procurement",
            run= "vol-match-2030-ci25-continent-6-3H",
            opts="",
            clusters="39",
            configfiles="config/config.meta.yaml",
            ll="v1.0",
            sector_opts="",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    if snakemake.params.get("ci_load", False):
        add_ci_load(n, snakemake.params)

    if snakemake.params.get("procurement_enable", False):
        logger.info(f"Procurement is activated for the year {planning_horizons}")
        procurement = snakemake.params.procurement

        if procurement.get("strip_network", False):
            logger.info("stript_network is activated")
            strip_network(n, procurement)

        if procurement.get("strip_snapshots", False):
            logger.info("stript_snapshots is activated")
            n.set_snapshots(n.snapshots[:168])

        Nyears = n.snapshot_weightings.objective.sum() / 8760.0
        costs = load_costs(
            snakemake.input.costs,
            snakemake.params.costs,
            snakemake.params.max_hours,
            Nyears,
        )
        add_ci_procurement(n, planning_horizons, snakemake.params, costs)

    if snakemake.params.get("electricity", {}).get("freeze_capacity", False):
        freeze_capacity(n, snakemake.params)

    if snakemake.params.get("electricity", {}).get("filter_TYNDP_build_year", False):
        filter_TYNDP_build_year(n, planning_horizons)

    n.export_to_netcdf(snakemake.output.network)