##########################################
Base Configuration
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

``run``
=============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Run Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#run>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: run:
   :end-before: # docs

This is the only few section in the file that needs to be changed in order to run the scenario.

* prefix: (optional) directory for output results
* name: Scenario name from ``config/scenarios.meta.yaml``

``foresight``
=============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Foresight Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#foresight>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: foresight:
   :end-at: foresight:

The scope of this work is based on myopic foresight.

``scenario``
============

* For a comprehensive explanation, refer to the upstream PyPSA-Eur `Scenario Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#scenario>`_
* For a comprehensive explanation, refer to the upstream PyPSA-Eur `Wildcard Documentation <https://pypsa-eur.readthedocs.io/en/latest/wildcards.html>`_

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: scenario:
   :end-before: # docs

* **clusters**: The model outputs are organized into 39 nodes, which is the default configuration in PyPSA-Eur. Models with higher spatial resolution are currently not supported.
* **planning_horizons**: The model simulates either the year 2025 or 2035.

``countries``
=============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Countries Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#countries>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: countries:
   :end-before: # docs

The analysis includes all of the default countries in PyPSA-Eur.

``snapshots``
=============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Snapshots Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#snapshots>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: snapshots:
   :end-before: # docs

The baseline scenario is based on the climate year 2013.

``enable``
==========

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Enable Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#enable>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-after: #enable
   :end-before: # docs

For the first run, it is recommended to set ``retrieve_databundle``, ``retrieve_cost_data`` and ``retrieve_cutout`` as true.

``electricity``
===============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Electricity Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#electricity>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: electricity:
   :end-before: # docs

Configuration changes made:

* If the year 2030 is selected, the ``powerplants_filter`` excludes countries that have committed to phasing out coal power plants by 2030.
* By default, Germany is set to phase out its nuclear power plants by 2025 and beyond.
* ``transmission_limit`` is set to ``v1.0``, meaning no transmission line expansions are allowed beyond those specified in ``transmission_projects``.

New configuration options introduced in this repository:

* ``ci_load``: Settings for generating non-procuring CI (Critical Infrastructure) loads and buses.
* ``freeze_capacity``: Option to prevent the expansion of renewable energy technologies (used for 2025 scenarios).
* ``filter_TYNDP_build_year``: Option to exclude TYNDP network components scheduled for construction after 2025 or 2030.


``transmission_projects``
=========================

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Transmission Projects Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#transmission_projects>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: transmission_projects:
   :end-before: # docs

Configuration changes made:

* The Ten Year Network Development Plan (TYNDP) 2020 transmission plan has been included from the model.
* The Netzentwicklungsplan (NEP) of Germany has been excluded in the model.
* The capacities of the newly added transmission lines are based on the targets specified for their planned build year.

``sector``
=======================

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Sector Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#sector>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: sector:
   :end-before: # docs

All sector-related configurations are disabled to model an electricity-only system.

``costs``
=============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Costs Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#costs>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: costs:
   :end-before: # docs

Configuration changes made:

* The cost data from PyPSA technology cost database is based on the year 2035, instead of 2030.
* Gas price is set at 31.98 EUR/MWh. `EU Natural Gas TTF in April 2025 <https://tradingeconomics.com/commodity/eu-natural-gas>`_
* Coal price is set at 10.08 EUR/MWh `API2 Rotterdam Coal Futures in April 2025 <https://www.tradingview.com/symbols/ICEEUR-ATW1!/>`_
* Lignite price is set at 22.11 EUR/MWh `Source from Business Analytiq (2024 Lignite price) <https://businessanalytiq.com/procurementanalytics/index/lignite-coal-price-index/>`_

This configuration is relevant if those storage technologies are included in the model:

* Iron-air battery price is set at 23,500 EUR/MWh (2024).
* Storage energy cost for CAES is set at 30,000 EUR/MWh (2022).
* Storage power cost for CAES is set at 1,725,000 EUR/MW (2022).
* The lifetime of CAES is reduced to 40 years.

``clustering``
==============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Clustering Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#clustering>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: clustering:
   :end-before: # docs

The temporal resolution is clustered to a 3H resolution by default, will change depending on the scenarios.

``adjustments``
===============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Plotting Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#adjustments>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: adjustments:
   :end-before: # docs

Configuration changes made:

* The marginal cost for DC links and the electricity distribution grid is set to 0.5 EUR/MWh to avoid unintended simultaneous exports and imports.

``solving``
=============

For a comprehensive explanation, refer to the upstream PyPSA-Eur `Solving Documentation <https://pypsa-eur.readthedocs.io/en/latest/configuration.html#solving>`_.

.. literalinclude:: ../config/config.meta.yaml
   :language: yaml
   :start-at: solving:
   :end-before: # =

.. note::
   As noted in the `Installation <https://open-energy-transition.github.io/procurement-metastudy/01-installation.html>`_ section, 
   there are several solvers compatible with PyPSA. Please choose the ones that are available to you.

   Each solver has a solver-specific parameter settings (``options``) to chose from:

   * **gurobi**: ``gurobi-default``, ``gurobi-numeric-focus``, ``gurobi-fallback``
   * **highs**: ``highs-default``
   * **cplex**: ``cplex-default``
   * **copt**: ``copt-default``, ``copt-gpu``
   * **cbc**: ``cbc-default``
   * **glpk**: ``glpk-default``