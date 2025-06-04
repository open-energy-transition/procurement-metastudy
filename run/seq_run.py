import os
import shutil
import yaml
import pandas as pd

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
        print("(Scenarios must include '--' in their name)")
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
            scenario_selected = scenarios[int(answer)-1]
            print(f"'{scenario_selected}' is selected.")
            return scenario_selected
        else:
            print(f"'{answer}' not found in the list. Please try again.")

def select_country():
    df = pd.read_csv("run/name_location_country.csv")
    available_countries = set(df["country"].unique())
    main_countries = {"DE", "FR", "PL", "ES", "IE", "DK"}

    print("Select which country or group of countries to execute:")
    print("  - main: DE, FR, PL, ES, IE, DK")
    print("  - all: all 34 countries individually")
    print("  - or enter country codes separated by commas (e.g., DE,FR,PL)")

    while True:
        answer = input("Select a country or group (or press Enter to cancel): ").strip().lower()
        if not answer:
            return None, df

        if answer == "all":
            print("All 34 countries selected.")
            return "all", df

        elif answer == "main":
            print(f"Selected main countries: {', '.join(sorted(main_countries))}")
            return "main", df[df["country"].isin(main_countries)]

        else:
            # Split and normalize input
            selected = {code.strip().upper() for code in answer.split(",")}
            invalid = selected - available_countries

            if invalid:
                print(f"Invalid country code(s): {', '.join(sorted(invalid))}. Please try again.")
            else:
                print(f"Selected countries: {', '.join(sorted(selected))}")
                return ",".join(sorted(selected)), df[df["country"].isin(selected)]

def select_procedure():
    while True:
        answer = input("Do you want all countries in the same model [y/n]?: ").strip().lower()
        if not answer:
            return None
        elif answer in ["y", "yes"]:
            return "combined"
        elif answer in ["n", "no"]:
            return "sequential"

        print(f"'{answer}' is not a valid option. Please try again.")

def select_profile():
    while True:
        answer = input("Is this in a computer cluster [y/n]?: ").strip().lower()
        if not answer:
            return None
        elif answer in ["y", "yes"]:
            return "--profile slurm"
        elif answer in ["n", "no"]:
            cpu = input("How many CPUs do you want to use (all or a number)? ").strip().lower()
            if cpu == "all":
                return "-call"
            elif cpu.isdigit():
                return f"-c{cpu}"
        print(f"'{answer}' is not a valid option. Please try again.")

def duplicate_run_delete(scenario_name, selected_baseline, selected_profile):
    # Prepare resources
    new_folder = f"resources/{scenario_name}"
    old_folder = f"resources/{selected_baseline}"
    files_to_copy = ["costs_2030.csv", "networks/base_s_39___2030_brownfield.nc"]

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

    temp_file = 'run/config.meta_temp.yaml'
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Deleted file: {temp_file}")
    else:
        print(f"File does not exist: {temp_file}")

# ---------------------- Main Script ----------------------

def main():
    # Load scenario configuration
    with open('config/scenarios.meta.yaml', 'r') as file:
        config_s = yaml.safe_load(file)

    # Extract scenarios with '--'
    scenario_list = [key for key in config_s if "--" in key]
    baseline_list = [key for key in config_s if "baseline" in key]

    # User input
    selected_scenario = select_scenario(scenario_list, name="scenario")
    if not selected_scenario:
        print("\nOperation cancelled by user.")
        return
    
    print("\n=================================================================")
    selected_baseline = select_scenario(baseline_list, name="baseline")
    if not selected_baseline:
        print("\nOperation cancelled by user.")
        return

    print("\n=================================================================")
    selected_country, df = select_country()
    if not selected_country:
        print("\nOperation cancelled by user.")
        return

    print("\n=================================================================")    
    selected_procedure = select_procedure()
    if not selected_procedure:
        print("\nOperation cancelled by user.")
        return

    print("\n=================================================================")
    selected_profile = select_profile()
    if not selected_profile:
        print("\nOperation cancelled by user.")
        return

    print("\nFinal selection:")
    print(f"Scenario: {selected_scenario}")
    print(f"Baseline: {selected_baseline}")
    print(f"Country group: {selected_country}")
    print(f"Country inclusion: {selected_procedure}")
    print(f"Profile: {selected_profile}")

    # Iterate runs
    if selected_procedure == "combined":
        with open('config/config.meta.yaml', 'r') as file:
                config = yaml.safe_load(file)

        deep_update(config, config_s[selected_scenario])

        # Modify config per country
        scenario_name = selected_scenario.replace("--", f"-{selected_country}-")
        ci_list = df.set_index("name")[["location"]].to_dict('index')

        config_update = {
            "run": {
                "name": scenario_name,
                "scenarios": {"enable": False}
            },
            "procurement": {
                "ci": ci_list
            }
        }

        deep_update(config, config_update)

        # Write temp config
        with open('run/config.meta_temp.yaml', 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)

        duplicate_run_delete(scenario_name, selected_baseline, selected_profile)


    elif selected_procedure == "sequential":
        for i, row in df.iterrows():
            # Load base config
            with open('config/config.meta.yaml', 'r') as file:
                config = yaml.safe_load(file)

            deep_update(config, config_s[selected_scenario])

            # Modify config per country
            country = row["country"]
            scenario_name = selected_scenario.replace("--", f"-{country}-")

            config_update = {
                "run": {
                    "name": scenario_name,
                    "scenarios": {"enable": False}
                },
                "procurement": {
                    "ci": {
                        row["name"]: {"location": row["location"]}
                    }
                }
            }

            deep_update(config, config_update)

            # Write temp config
            with open('run/config.meta_temp.yaml', 'w') as file:
                yaml.safe_dump(config, file, default_flow_style=False)

            duplicate_run_delete(scenario_name, selected_baseline, selected_profile)

if __name__ == "__main__":
    main()