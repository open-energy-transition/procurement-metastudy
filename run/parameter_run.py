# SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
import os
import shutil

import yaml

# ---------------------- Utility Functions ----------------------


def deep_update(original, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            deep_update(original[key], value)
        else:
            original[key] = value


def select_scenario(scenarios, name="scenarios"):
    print(f"Available {name}:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    if name == "scenarios":
        print("(Scenarios cannot include 'baseline' in their name)")
    elif name == "baseline":
        print("(Baselines must include 'baseline' in their name)")
    print("(You can also select the list number)")

    while True:
        answer = input("Select a scenario (or press Enter to cancel): ").strip()
        if not answer:
            return None
        if answer in scenarios:
            print(f"'{answer}' is selected.")
            return answer
        elif answer.isdigit():
            scenario_selected = scenarios[int(answer) - 1]
            print(f"'{scenario_selected}' is selected.")
            return scenario_selected
        else:
            print(f"'{answer}' not found in the list. Please try again.")


def select_profile():
    while True:
        answer = input("Is this in a computer cluster [y/n]?: ").strip().lower()
        if not answer:
            return None
        elif answer in ["y", "yes"]:
            return "--profile slurm"
        elif answer in ["n", "no"]:
            cpu = (
                input("How many CPUs do you want to use (all or a number)? ")
                .strip()
                .lower()
            )
            if cpu == "all":
                return "-call"
            elif cpu.isdigit():
                return f"-c{cpu}"
        print(f"'{answer}' is not a valid option. Please try again.")


def select_multiruns():
    while True:
        answer = (
            input("is there more runs that you want to make [y/n]?: ").strip().lower()
        )
        if not answer:
            return None
        elif answer in ["y", "yes"]:
            return True
        elif answer in ["n", "no"]:
            return False
        print(f"'{answer}' is not a valid option. Please try again.")


def duplicate_run_delete(scenario_name, selected_baseline, selected_profile):
    # Prepare resources
    new_folder = f"resources/{scenario_name}"
    old_folder = f"resources/{selected_baseline}"

    year = 2025 if "2025" in scenario_name else 2030

    files_to_copy = [f"costs_{year}.csv", f"networks/base_s_39___{year}_brownfield.nc"]

    os.makedirs(os.path.join(new_folder, "networks"), exist_ok=True)

    for f in files_to_copy:
        old_path = os.path.join(old_folder, f)
        new_path = os.path.join(new_folder, f)
        if os.path.isfile(old_path):
            shutil.copy(old_path, new_path)
            print(f"Copied {old_path} to {new_path}")
        else:
            print(f"WARNING: File not found: {old_path}")

    # Run snakemake
    run_cmd = f"snakemake {selected_profile} solve_sector_networks --configfile run/config.meta_temp.yaml --rerun-trigger mtime"
    os.system(run_cmd + " --touch")
    os.system(run_cmd)

    # Clean up
    if os.path.exists(new_folder):
        shutil.rmtree(new_folder)
        print(f"Deleted folder: {new_folder}")
    else:
        print(f"Folder does not exist (already removed?): {new_folder}")

    temp_file = "run/config.meta_temp.yaml"

    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Deleted file: {temp_file}")
    else:
        print(f"File does not exist: {temp_file}")


# ---------------------- Main Script ----------------------


def main():
    scenarios = []
    for res in [True, False]:
        res_tag = "-NoResTargets-" if not res else "-"
        baseline_scenario = "baseline-2025-1H"  # No RES in 2025
        # (
        #     "baseline-2025-1H" if res else "baseline-2025-NoResTargets-1H"
        # )
        for ci_participation in [25]:  # [10, 25, 50]:
            for em_signal in ["mber", "moer", "cmer", "aer", "lrmer_c"]:
                em_name = (
                    f"emi-match-2025-ci{ci_participation}-model-{em_signal}{res_tag}1H"
                )
                em_config = {
                    "scenario": {"planning_horizons": [2025]},
                    "costs": {"year": 2025},
                    "electricity": {
                        "powerplants_filter": "(DateOut >= 2024 or DateOut != DateOut) and not (Country == 'Germany' and Fueltype == 'Nuclear')"
                    },
                    "enable": {"procurement": True},
                    "clustering": {"temporal": {"resolution_sector": "1H"}},
                    "res_target": {
                        "EU_share_target": res,
                        "country_cap_target": res,
                        "res_additionality": False,
                    },
                    "procurement": {
                        "strategy": "emi-match",
                        "scope": "continent",
                        "energy_matching": 100,
                        "participation": ci_participation * 2,
                        "emissionality": {
                            "emissions_matching": 100,
                            "emission_signal": em_signal,
                            "signal_source": "historical",
                        },
                    },
                }
                scenarios.append(([em_name, em_config], baseline_scenario))
            # for cfe_level in [70, 80, 90, 100]:
            #     cfe_name = f"247-cfe-2030-ci{ci_participation}-cfe{cfe_level}{res_tag}grid-use-SSS-1H"
            #     cfe_config = {
            #         "scenario": {"planning_horizons": [2030]},
            #         "costs": {"year": 2030},
            #         # "electricity": {
            #         #     "powerplants_filter": "(DateOut >= 2024 or DateOut != DateOut) and not (Country == 'Germany' and Fueltype == 'Nuclear')"
            #         # },
            #         "enable": {"procurement": True},
            #         "clustering": {"temporal": {"resolution_sector": "1H"}},
            #         "res_target": {
            #             "EU_share_target": res,
            #             "country_cap_target": res,
            #             "res_additionality": False,
            #         },
            #         "procurement": {
            #             "strategy": "247-cfe",
            #             "scope": "node",
            #             "energy_matching": cfe_level,
            #             "participation": ci_participation * 2,
            #             "min_iterations": 3,
            #             "excess_share": 1,
            #             "use_SSS": True,
            #         },
            #     }
            #     scenarios.append(([cfe_name, cfe_config], baseline_scenario))

    # Iterate runs
    for scenario, selected_baseline in scenarios:
        selected_scenario, scenario_config = scenario
        print("\n=================================================================")
        print("Currently running:")
        print(f"Scenario: {selected_scenario}")
        print(f"Baseline: {selected_baseline}")

        with open("config/config.meta.yaml") as file:
            config = yaml.safe_load(file)

        deep_update(config, scenario_config)

        # Modify config per country
        scenario_name = selected_scenario

        config_update = {
            "run": {"name": selected_scenario, "scenarios": {"enable": False}}
        }

        deep_update(config, config_update)

        # Write temp config
        with open("run/config.meta_temp.yaml", "w") as file:
            yaml.safe_dump(config, file, default_flow_style=False)

        duplicate_run_delete(scenario_name, selected_baseline, "-call")


if __name__ == "__main__":
    main()
