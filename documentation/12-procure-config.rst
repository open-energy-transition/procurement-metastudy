..
   SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
   SPDX-License-Identifier: CC-BY-4.0
##########################################
Procurement Configuration
##########################################

This is the procurement configuration used in the **WattTime Impact Metastudy**.
These features were not available in the default PyPSA-Eur at the time of documentation.

.. note::
   The PyPSA-Eur configuration files follow a pyramid-like structure, where the parameters in the highest configuration file add to and override those in the configuration file below it.
   The order is as follows:

   1. `scenarios.meta-{xH}.yaml <https://open-energy-transition.github.io/procurement-metastudy/13-scenario-config.html>`_
   2. config.meta.yaml (**this section**)
   3. `config.default.yaml <https://pypsa-eur.readthedocs.io/en/latest/configuration.html>`_

Thus, for example changes specified in `scenario.meta.yaml` will add to and override configurations in `config.meta.yaml` and so on.

``res_target``
==============

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: res_target:
   :end-at: res_path:

This configuration provides the option to set renewable energy targets for the year 2030, using the `EMBER 2030 Global Renewable Energy Target Tracker <https://ember-energy.org/data/2030-global-renewable-target-tracker/>`_ as a reference. 
The targets are set by means of a the new constraint ``ember_res_target`` defined in ``solve_network.py/extra_functionality``.
While it is possible to enable all available targets, please note that doing so may increase the likelihood of curtailment and inefficient allocation of generators.

* ``EU_share_target``: If ``true``, applies the EU-wide renewable generation share target of 72%.
* ``country_share_target```: If ``true``, applies country-specific renewable generation share targets.
* ``country_cap_target``: If ``true``, applies country-specific renewable capacity targets.
* ``res_additionality```: If ``true``, excludes procurement strategies from the targets, indicating that all procurements are additional to the background system.
* ``res_path``: File path to the EMBER 2030 Global Renewable Energy Target data.


``grid_policy``
===============

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: grid_policy:
   :end-at: emitters:

This configuration defines how the model categorizes generation technologies as renewable, clean, or emitting.
This affects the implementation of renewable targets as well as the assessment of background grid quality in 24/7 CFE (carbon-free energy) calculations.

``procurement``
===============

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-after: emitters: ["CCGT", "OCGT", "coal", "lignite", "oil"]
   :end-at: location: "XK0 0"

These configurations define the core components of procurement strategies:

* ``strategy``: Name of the procurement strategy (e.g., annual volume matching, 24/7 carbon-free energy, emission matching).
.. note::
   Depending on the procurement strategy selected, the following constraints defined in ``solve_network.py/extra_functionality`` will be applied:

   * ``vol-match`` (annual volume matching): ``res_annual_matching_constraints``.
   * ``24/7-cfe`` (24/7 carbon-free energy): ``cfe_constraints``.
   * ``emi-match`` (emission matching): ``emission_matching_constraints`` and ``res_annual_matching_constraints``.

   Then, common constraints for all strategies are also applied:

   * ``res_capacity_constraints``: Restricts the deployment of renewable capacities for the same carrier within the same buses.
   * ``excess_constraints``: Ensures that each CI bus must meet its own load consumption before exporting any energy back to the grid based on the proportion of the procured CI load demand.
   * ``import_constraints``: Ensures that each CI bus can only import electricity based on the proportion of the procured CI load demand.

* ``scope``: Spatial scope of the procurement (e.g., node, country, all, continent).
.. note::
   Consider that 24/7 CFE is meant for a ``node`` scope, while annual volume and emission matching can include all the scopes. In particular:

   * ``node``: carbon-free energy can be only procured at the level of the participating CI load bus.
   * ``country``: carbon-free energy can be procured at the level of the participating CI load country.
   * ``all``: carbon-free energy can be procured anywhere in the system. This scope allows to track the origin country of the procured energy separately for each participating CI load.
   * ``continent``: carbon-free energy can be procured anywhere in the system. Differently from the ``all`` scope, this one is not able to track the origin country separately, but aggregates the procured energy as well as the participating CI load.
     This dramatically reduces the number of variables in the optimization problem (i.e., the computational burden), while still allowing to fulfill transmission grid constraints.
     Also, when it is selected, the annual volume and emission matching constraints selected are, respectively, ``res_annual_matching_constraints_continent`` and ``emission_matching_constraints_continent``.

* ``energy_matching``: Percentage of CI energy demand to be procured in a given year (e.g., enter 10 for 10%).
* ``emissionality``
   * ``emission_matching``: Percentage of CI emissions to be offset or matched in a given year.
   * ``emission_signal``: Type of emission signal used (e.g., AER, MBER, MOER, CMER).
   * ``signal_source``: Source of the emission signal (e.g., model-generated or historical data).
   * ``signal_model``: Folder path to model-generated signals (used when AER, MBER, or MOER is selected).
   * ``signal_historical``: Folder path for historical emission signal data (used when historical is selected).
.. note::
   There are two types of emission signals that can be used in the emission matching strategy:

   * *Model-based emission signals*: these signals are used if ``siganl_source`` is set to ``model``.
     In particular, they are generated from baseline scenarios results by means of the ``model-based-signals.ipynb`` notebook in the ``notebooks/emission-signals/`` directory.
     This notebook directly pulls the desired baseline network from the ``results/`` directory within the repository (but this can be easily adapted to be used outside the repository).
     Then, it stores the generated signals in dedicated files for each country involved in the analysis in the ``data/emission-signals-model/`` directory (i.e., the path set in ``signal_model``).
   * *Historical data-based emission signals* (under development): these signals are used if ``siganl_source`` is set to ``historical``.
     In particular, they are generated from historical data by means of ``historical-based-signals.ipynb`` notebook in the ``notebooks/emission-signals/`` directory.
     In order to use the notebook, one needs to ask WattTime for specific credentials. 
     Then, it stores the generated signals in a single file in the ``data/emission-signals-model/`` directory (i.e., the path set in ``signal_historical``).

* ``participation``: Share of CI load participating in the procurement, expressed as a percentage.
  Consider that this share is applied to the CI load modelled in the baseline configuration, which is in turn only a fraction of the total CI load (set in ``share`` in the `electricity base configuration settings <https://open-energy-transition.github.io/procurement-metastudy/11-baseline-config.html#electricity>`_).
  For instance, if one aims to account for 25% of participation rate, while only 50% of the total CI load is explicitly modelled, ``participation`` should be set to ``50``.

These configurations determine which technologies are included in the procurement strategy.
In particular, they are modelled through the ``add_ci_procurement`` function in the ``add_procurement.py`` script:

* ``technology``
   * ``generation_tech``: List of renewable and clean generation technologies to include.
   .. note::
      Consider that solar-rooftop modeling involves a simplification when ``scope`` is set to ``node``.
      Indeed, it would be connected to a high-voltage bus (since the CI load is connected to a high-voltage bus, as pointed out below for the ``ci`` setting), even though it is a low-voltage technology.
      It would be more correct to have a dedicated CI load low-voltage bus. However, as far it is assumed that solar-rooftop is built directly onsite to supply CI loads, the electricity distribution grid is not needed.
      On the other hand, potential connection costs would be neglected (instead, they are considered for utility-scale solar, which is connected to high-voltage buses).

   * ``storage_tech``: List of eligible storage technologies (defined as storage_units).

These settings are particularly relevant for testing or simplified model resolution:

* ``strip_network``: If ``true``, models only countries with CI loads and their immediate neighbors.
* ``strip_snapshots``: If ``true``, limits the model to the first 168 time steps.

``cap_premium`` serves dual purposes: it can be used to guide CI procurement technologies to optimal locations and can also reflect real-world conditions where these technologies may incur higher capital costs.

These configurations are primarily relevant for strategies with nodal scope, such as 24/7 carbon-free energy:

* ``excess_share``: Proportion of excess CI generation at the CI bus that can be sold back to the grid (range: 0 to 1). It is used in ``excess_constraints``.
* ``import_share``: Proportion of electricity demand that can be imported from the grid (range: 0 to 1). It is used in ``import_constraints``.
* ``min_iterations``: Setting this above 1 allows the background grid quality to influence the strategy implementation in iterative runs.

This configuration specifies the location of CI procurement. The list includes all countries represented in the model:

* ``ci``: {name} with {location: bus}.
.. note::
   As previously mentioned, the participating CI load is a fraction of the CI load already modelled in the `baseline configuration <file:///home/user/OET/projects/metastudy/procurement-metastudy/documentation/_build/html/11-baseline-config.html#electricity>`_.
   In particular, it is  generated through the ``add_ci_procurement`` function in the ``add_procurement.py`` script.
   
   To understand the rationale behind its modelling, let's consider the following example: a participating CI load with ``name`` set to ``Germany`` and ``location`` set to ``DE0 0``.
   
   * The modelled participating CI load would be a portion of ``DE0 0 CI load``, that might be called ``Germany load`` and is connected to a dedicated high-voltage bus ``Germany CI``.
   * Also, dedicated import and export links are modelled to connect the ``DE0 0`` and ``Germany CI`` buses.
   

