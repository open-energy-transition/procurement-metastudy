import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from pathlib import Path


def split_common_carriers(gen, country):
    PP_RESOURCE = (
        Path(__file__).parents[1] / "resources/baseline-2030-1H/powerplants_s_39.csv"
    )
    pp = pd.read_csv(PP_RESOURCE, index_col=0)
    pp["date_round"] = (pp["DateIn"] / 5).round().astype(int) * 5
    tech_map = {
        "nuclear": ("Nuclear", "Steam Turbine"),
        "hydro": ("Hydro", "Reservoir"),
        "ror": ("Hydro", "Run-Of-River"),
        "PHS": ("Hydro", "Pumped Storage"),
    }

    plants = pp[pp["Country"] == country]
    for tech, types in tech_map.items():
        plants_tech = plants[
            (plants["Fueltype"] == types[0]) & (plants["Technology"] == types[1])
        ]

        if plants_tech.empty:
            continue

        plants_cap = plants_tech["Capacity"].sum()
        plant_ratio = plants_tech.groupby("date_round")["Capacity"].sum() / plants_cap

        gen_cols = [c for c in gen.columns if tech in c]
        gen_tech = gen[gen_cols].sum(axis=1)

        gen_year = plant_ratio.apply(lambda x: x * gen_tech).T
        gen_year.columns = [f"{country} {tech}-{c}" for c in gen_year.columns]

        gen = gen.drop(columns=gen_cols)
        gen = pd.concat([gen, gen_year], axis=1)

    return gen


# def get_gen_and_em(n, country):
#     eb = n.statistics.energy_balance(
#         aggregate_time=False, groupby=["country", "carrier", "bus_carrier", "name"]
#     ).reset_index()
#     gen = eb[(eb["bus_carrier"] == "AC")]

#     if country:
#         gen = gen[(gen["country"] == country)]
# emisisons are not tagged by country
# em = eb[(eb["bus_carrier"] == "co2")]
# em = (
#     em.set_index("name")
#     .drop(columns=["component", "country", "carrier", "bus_carrier"])
#     .T
# )

#     gen_carriers = [
#         "Open-Cycle Gas",
#         "li-ion battery discharger",
#         "Combined-Cycle Gas",
#         "lignite",
#         "oil",
#         "urban central solid biomass CHP",
#         "coal",
#         "nuclear",
#         "Reservoir & Dam",
#         "Pumped Hydro Storage",
#         "Offshore Wind (AC)",
#         "Offshore Wind (Floating)",
#         "Onshore Wind",
#         "Run of River",
#         "Solar",
#         "solar-hsat",
#         "Offshore Wind (DC)",
#     ]

#     def group_carriers(_df):
#         if _df["carrier"].isin(gen_carriers).all():
#             _df["carrier"] = _df["name"]
#         return _df

#     gen = gen.groupby("carrier", as_index=False).apply(group_carriers)
#     gen = gen.reset_index(drop=True).drop(
#         columns=["component", "country", "bus_carrier", "name"]
#     )
#     gen = gen.groupby("carrier").sum()
#     gen = gen.T
#     gen = split_common_carriers(gen, country)

# em = em[[c for c in gen.columns if c in em.columns]]

#     return gen, em


def calc_mber(gens, em_by_plant):
    mber = {}
    for ctry, gen in gens.items():
        em = em_by_plant[ctry]

        cols = gen.columns
        cols = cols[cols.str.contains("20") | cols.str.contains("19")]
        years = cols.map(lambda x: x.split("-")[-1])
        gen_by_year = gen[cols].T.groupby(years).sum().T
        em_by_year = em.reindex(cols, axis=1)[cols].T.groupby(years).sum().T
        gen_total = gen_by_year.sum(axis=1)
        gen_frac_by_year = gen_by_year.cumsum(axis=1).apply(
            lambda x: x / gen_total.values, axis=0
        )
        recent_em = em_by_year[gen_frac_by_year > 0.8].sum(axis=1)
        recent_gen = gen_by_year[gen_frac_by_year > 0.8].sum(axis=1)
        mber[ctry] = recent_em / recent_gen

    return mber


def get_data(n):
    RE_FUELS = [
        "Reservoir & Dam",
        "Offshore Wind (AC)",
        "Offshore Wind (Floating)",
        "Onshore Wind",
        "Run of River",
        "Solar",
        "solar rooftop",
        "solar-hsat",
        "Offshore Wind (DC)",
    ]

    gen_carriers = [
        "Open-Cycle Gas",
        "li-ion battery discharger",
        "Combined-Cycle Gas",
        "lignite",
        "oil",
        "urban central solid biomass CHP",
        "coal",
        "nuclear",
        "Reservoir & Dam",
        "Pumped Hydro Storage",
        "Offshore Wind (AC)",
        "Offshore Wind (Floating)",
        "Onshore Wind",
        "Run of River",
        "Solar",
        "solar-hsat",
        "solar rooftop",
        "Offshore Wind (DC)",
    ]
    buses = n.buses
    eb = n.statistics.energy_balance(
        aggregate_time=False, groupby=["country", "carrier", "bus_carrier", "name"]
    ).reset_index()
    countries = buses["country"].unique()[:-1]

    gens = {}
    demand = {}  # demand
    re = {}
    ints = {}  # interchange
    em_by_country = {}
    em_by_plant = {}

    for country in countries:
        # print(f'Processing {country}')
        gen = eb[(eb["country"] == country)]
        gbf = (
            gen.drop(columns=["component", "country", "bus_carrier", "name"])
            .groupby("carrier")
            .sum()
        )
        gbf_dem = gbf[gbf < 0]
        gbf_dem = gbf_dem[
            (gbf_dem.index != "AC") & (gbf_dem.index != "DC")
        ]  # GC: we need to exclude exports, to avoide double counting in the Flow Tracing calculation
        demand[country] = gbf_dem[gbf_dem < 0].T.sum(axis=1)
        re[country] = gbf.loc[[f for f in gbf.index if f in RE_FUELS]].T.sum(axis=1)

        # get generation data
        def group_carriers(_df):
            if _df["carrier"].isin(gen_carriers).all():
                _df["carrier"] = _df["name"]
            return _df

        gen_by_plant = gen.groupby("carrier", as_index=False)[gen.columns].apply(
            group_carriers, include_groups=True
        )
        gen_by_plant = gen_by_plant.reset_index(drop=True).drop(
            columns=["component", "country", "bus_carrier", "name"]
        )
        gen_by_plant = gen_by_plant.groupby("carrier").sum()
        gen_by_plant = gen_by_plant.T
        gen_by_plant = split_common_carriers(gen_by_plant, country)
        gens[country] = gen_by_plant

        # emisisons are not tagged by country
        em = eb[(eb["bus_carrier"] == "co2")]
        em = (
            em.set_index("name")
            .drop(columns=["component", "country", "carrier", "bus_carrier"])
            .T
        )
        em_by_plant[country] = em[[c for c in gen_by_plant.columns if c in em.columns]]
        em_by_country[country] = em_by_plant[country].sum(axis=1)

        DC = (
            gen[gen["carrier"] == "DC"]
            .set_index("name")
            .drop(columns=["component", "country", "carrier", "bus_carrier"])
        )
        dc_links = n.links.loc[DC.index]
        dc_links["country0"] = dc_links["bus0"].map(buses["country"])
        dc_links["country1"] = dc_links["bus1"].map(buses["country"])
        dc_links["country"] = (
            dc_links[["country0", "country1"]].replace(country, "").max(axis=1)
        )
        DC_net = DC.groupby(dc_links["country"]).sum()

        AC = (
            gen[gen["component"] == "Line"]
            .set_index("name")
            .drop(columns=["component", "country", "carrier", "bus_carrier"])
        )
        ac_lines = n.lines.loc[AC.index]
        ac_lines["country0"] = ac_lines["bus0"].map(buses["country"])
        ac_lines["country1"] = ac_lines["bus1"].map(buses["country"])
        ac_lines["country"] = (
            ac_lines[["country0", "country1"]].replace(country, "").max(axis=1)
        )
        AC_net = AC.groupby(ac_lines["country"]).sum()
        int_net = pd.concat([AC_net, DC_net])
        int_net = int_net.groupby(int_net.index).sum().T
        ints[country] = int_net

    em_by_country = pd.DataFrame(em_by_country)

    return gens, demand, re, ints, em_by_country, em_by_plant


def flow_trace(ints, ems, demand, countries):
    consumed_em = []
    N = ints[countries[0]].shape[0]
    for i in range(N):
        int_matrix = [ints[country].iloc[i] for country in countries]
        int_matrix = pd.concat(int_matrix, axis=1)
        int_matrix.columns = countries
        int_matrix = int_matrix.loc[int_matrix.columns]
        int_matrix = int_matrix.fillna(0).T
        imp_matrix = int_matrix.where(int_matrix > 0, 0)
        exp = int_matrix.where(int_matrix < 0, 0).sum(axis=1)

        flow_matrix = imp_matrix
        for country in demand.keys():
            flow_matrix.loc[country, country] = demand[country].iloc[i] + exp[country]

        countries = flow_matrix.index
        gen_em = pd.Series([ems[c].iloc[i] for c in countries], index=countries)

        em_rate = np.linalg.inv(flow_matrix.values) @ gen_em.values
        d = pd.Series([demand[c].iloc[i] for c in countries], index=countries)

        consumed_em.append(em_rate * d)
    consumed_em = pd.concat(consumed_em, axis=1).T
    consumed_em.index = ems[country].index[:N]

    return consumed_em


def calc_moer(demand, consumed_em, re):
    # Implementing Zohrabian 2023
    # "A data-driven framework for quantifying consumption-based monthly and hourly marginal emissions factors"
    moers = {}

    for country in demand.keys():
        df = pd.concat([demand[country], consumed_em[country], re[country]], axis=1)
        df.index = pd.to_datetime(df.index)
        df.columns = ["demand", "em", "re"]
        df["demand"] = -1 * df["demand"]
        df["net_demand"] = df["demand"] - df["re"]
        deltas = df.diff(-1)

        bins = pd.qcut(df["demand"], 10, duplicates="drop")

        slopes = {}
        for bin, _df in deltas.groupby([deltas.index.month, bins], observed=True):
            _df = _df.dropna()
            results = smf.ols("em ~ demand + re", data=_df).fit()
            slopes[bin] = results.params["demand"]

        h_slope = (
            pd.MultiIndex.from_arrays([deltas.index.month, bins]).map(slopes).values
        )
        h_slope = pd.Series(h_slope, index=deltas.index)
        moers[country] = (
            h_slope.astype(float)
            .groupby([h_slope.index.month, h_slope.index.hour])
            .transform("mean")
        )
        moers[country] = moers[country].where(moers[country] > 0, 0)

    return moers


def calc_aer(demand, consumed_em):
    aer = {}
    for ctry, d in demand.items():
        aer[ctry] = -1 * consumed_em[ctry] / d
    return aer
