..
   SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
   SPDX-License-Identifier: CC-BY-4.0
##########################################
Scenario Configuration
##########################################

This is the scenario configuration used in the **WattTime Impact Metastudy**.

.. note::
   The PyPSA-Eur configuration files follow a pyramid-like structure, where the parameters in the highest configuration file add to and override those in the configuration file below it.
   The order is as follows:

   1. scenarios.meta-{xH}.yaml (**this section**)
   2. `config.meta.yaml <https://open-energy-transition.github.io/procurement-metastudy/11-baseline-config.html>`_
   3. `config.default.yaml <https://pypsa-eur.readthedocs.io/en/latest/configuration.html>`_

Thus, for example changes specified in `scenario.meta.yaml` will add to and override configurations in `config.meta.yaml` and so on.

Baseline Scenarios
==================

The following table provide an overview of the baseline scenarios:

.. list-table:: Baseline
   :widths: 5 45 25 25
   :header-rows: 1

   * - #
     - Scenario name
     - Planning horizon
     - RES targets
   * - 1
     - baseline-2025-nH
     - 2025
     - No
   * - 2
     - baseline-2030-nH
     - 2030
     - Yes
   * - 3
     - baseline-2030-NoResTargets-nH
     - 2030
     - No

Key similarities across all baseline scenarios that distinguish them from the default PyPSA-Eur results:

* Load demand is split into low-voltage demand and commercial/industrial (CI) demand at the high-voltage level for all nodes.
* Transmission capacities between country buses are fixed, including additional capacities specified in the TYNDP.
* The model considers electricity only (no sector coupling).
* Fossil fuel prices are customized to reflect estimates for the year 2025.

Key differences by year:

* 2025: No coal phase-outs have been initiated.
* 2030: 17 countries have achieved their coal phase-out targets. See `Beyond Fossil Fuels Europe's Coal Exit<https://beyondfossilfuels.org/europes-coal-exit/>`_.

When the RES target is referenced in this table, it implies that:

* The model enforces the EU-wide renewable generation share target of 72%.
* Country-specific renewable capacity targets are applied.
* National renewable generation share targets are not considered (i.e., deactivated).


Annual Volume Matching Scenarios
================================

The following table provide an overview of the annual volume matching scenarios:

.. list-table:: Annual Volume Matching
   :widths: 5 40 20 10 10 15
   :header-rows: 1

   * - #
     - Scenario name
     - Baseline
     - Planning horizon
     - RES targets
     - Location scope
   * - 1
     - vol-match-2025-ci25-nH
     - baseline-2025-nH
     - 2025
     - No
     - Continent
   * - 2
     - vol-match-2030-ci25-nH
     - baseline-2030-nH
     - 2030
     - Yes
     - Continent
   * - 3
     - vol-match-2030-country-ci25-nH
     - baseline-2030-nH
     - 2030
     - Yes
     - Country
   * - 4
     - vol-match-2030-NoResTargets-ci25-nH
     - baseline-2030-NoResTargets-nH
     - 2030
     - No
     - Continent

Key similarities across these annual volume matching scenarios:

* Energy matching is set at 100%.
* Each country sources 25% of its commercial and industrial (CI) demand domestically.
* All procurement counts toward renewable energy (RES) targets (no additionality).

The sensitivities explore two main aspects:

* The impact of limiting procurement strictly to within individual countries.
* The consequences of not meeting renewable energy (RES) targets.

24/7 Carbon Free Energy Scenarios
=================================

The following table provide an overview of the 24/7 carbon free energy scenarios:

.. list-table:: 24/7 Carbon Free Energy
   :widths: 5 30 20 10 10 10 15
   :header-rows: 1

   * - #
     - Scenario name
     - Baseline
     - Hourly Matching
     - Planning horizon
     - RES targets
     - Location scope
   * - 1
     - 247-cfe-2025-ci25-cfe90-nH
     - baseline-2025-nH
     - 90%
     - 2025
     - No
     - Node
   * - 2
     - 247-cfe-2025-ci25-cfe100-nH
     - baseline-2025-nH
     - 100%
     - 2025
     - No
     - Node
   * - 3
     - 247-cfe-2030-ci25-cfe90-nH
     - baseline-2030-nH
     - 90%
     - 2030
     - Yes
     - Node
   * - 4
     - 247-cfe-2030-ci25-cfe100-nH
     - baseline-2030-nH
     - 100%
     - 2030
     - Yes
     - Node

Key similarities across these annual volume matching scenarios:

* All procurement counts toward renewable energy (RES) targets (no additionality).

The sensitivities explore two main aspects:

* Different energy matching levels, set at either 90% or 100%.
* The year of procurement, which affects whether RES targets are in place.


Emission Matching Scenarios
===========================

The following table provide an overview of the emission matching scenarios:

.. list-table:: Emission Matching
   :widths: 5 30 20 9 9 9 9 9
   :header-rows: 1

   * - #
     - Scenario name
     - Baseline
     - Signal Source
     - Signal Type
     - Planning horizon
     - RES targets
     - Location scope
   * - 1
     - emi-match-2025-ci25-data-moer-nH
     - baseline-2025-nH
     - Statistics
     - MOER
     - 2025
     - No
     - Continent
   * - 2
     - emi-match-2025-ci25-data-mber-nH
     - baseline-2025-nH
     - Statistics
     - MBER
     - 2025
     - No
     - Continent
   * - 3
     - emi-match-2025-ci25-data-cmer-nH
     - baseline-2025-nH
     - Statistics
     - CMER
     - 2025
     - No
     - Continent
   * - 4
     - emi-match-2025-ci25-data-aer-nH
     - baseline-2025-nH
     - Statistics
     - AER
     - 2025
     - No
     - Continent
   * - 5
     - emi-match-2030-ci25-data-moer-nH
     - baseline-2030-nH
     - Model
     - MOER
     - 2030
     - No
     - Continent
   * - 6
     - emi-match-2030-ci25-data-mber-nH
     - baseline-2030-nH
     - Model
     - MBER
     - 2030
     - No
     - Continent
   * - 7
     - emi-match-2030-ci25-data-cmer-nH
     - baseline-2030-nH
     - Model
     - CMER
     - 2030
     - No
     - Continent
   * - 8
     - emi-match-2030-ci25-data-aer-nH
     - baseline-2030-nH
     - Model
     - AER
     - 2030
     - No
     - Continent

Key similarities across these emission matching scenarios:

* Energy matching is set at 100%.
* Each country sources 25% of its commercial and industrial (CI) demand domestically.
* All procurement counts toward renewable energy (RES) targets (no additionality).

The sensitivities explore two main aspects:

* The type of emission matching signal used (MOER, MBER, CMER, AER).
* Whether the signal is derived from recent historical data or from long-term model projections.