##########################################
Procurement Configuration
##########################################

This is the base configuration outline used in the analysis "The Role of Energy Storage in Germany".
The complete explaination of each configuration can be found in `PyPSA-Eur: Configuration <https://pypsa-eur.readthedocs.io/en/latest/configuration.html>`_.

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
* ``scope``: Spatial scope of the procurement (e.g., node, country, all, continent).
* ``energy_matching``: Percentage of CI (Critical Infrastructure) energy demand to be procured in a given year (e.g., enter 10 for 10%).
* ``emissionality``
   * ``emission_matching``: Percentage of CI emissions to be offset or matched in a given year.
   * ``emission_signal``: Type of emission signal used (e.g., AER, MBER, MOER, CMER).
   * ``signal_source``: Source of the emission signal (e.g., model-generated or historical data).
   * ``signal_model``: Folder path to model-generated signals (used when AER, MBER, or MOER is selected).
   * ``signal_historical``: Folder path for historical emission signal data (used when historical is selected).
* ``participation``: Share of CI load participating in the procurement, expressed as a percentage (e.g., 10 for 10%).

These configurations determine which technologies are included in the procurement strategy:

* ``technology``
   * ``generation_tech``: List of renewable and clean generation technologies to include.
   * ``storage_tech``: List of eligible storage technologies (defined as storage_units).

These settings are particularly relevant for testing or simplified model resolution:

* ``strip_network``: If ``true``, models only countries with CI loads and their immediate neighbors.
* ``strip_snapshots``: If ``true``, limits the model to the first 168 time steps.

``cap_premium`` serves dual purposes: it can be used to guide CI procurement technologies to optimal locations and can also reflect real-world conditions where these technologies may incur higher capital costs.

These configurations are primarily relevant for strategies with nodal scope, such as 24/7 carbon-free energy:

* ``excess_share``: Proportion of excess CI generation at the CI bus that can be sold back to the grid (range: 0 to 1).
* ``import_share``: Proportion of electricity demand that can be imported from the grid (range: 0 to 1).
* ``min_iterations``: Setting this above 1 allows the background grid quality to influence the strategy implementation in iterative runs.

This configuration specifies the location of CI procurement. The list includes all countries represented in the model:

* ``ci``: {name} with {location: bus}.