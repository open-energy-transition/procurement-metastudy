##########################################
Tutorials
##########################################

This is a brief tutorial on how to recreate the scenarios of the study **WattTime Impact Metastudy**.
Unlike the original PyPSA-Eur repository, this version comes pre-configured with scenarios that you can test run.

Create the baseline models
==========================

To make it easier for users to replicate results, we provide a simple command-line interface for generating the baseline model.
Once the environment is activated, you can create the baseline scenario by running:

.. code:: console

   python run/baseline_run.py

During execution, you will be prompted to specify the following:

* Desired time resolution (1H or 3H)
* Which baseline scenarios to generate (you can select more than one)
* Whether you're using a computing cluster (yes/no)
* (If not using a cluster) The number of CPUs available on your local machine

The script will then generate the baseline scenarios from scratch.

.. note::
    If an error occurs due to missing data, the first solution is to check the ``enable:`` setting in ``config.meta.yaml``.

    .. literalinclude:: ../config/config.meta.yaml
       :language: yaml
       :start-after: #enable
       :end-before: # docs

    * ``retrieve_databundle``: set this to true if a file in ``data/`` is missing
    * ``retrieve_cost_data``: set this to true if ``resources/{run}/cost_{}.csv`` is missing
    * ``retrieve_cutout``: set this to true if ``cutouts/europe-{weather year}-sarah3-era5.nc`` is missing

    The `weather data cutouts for PyPSA-Eur <https://zenodo.org/records/14936211>`_ are available for all the weather years included in the scenarios.
    If the weather year you've selected is not on that list, you'll need to set ``build_cutout`` to true.

Create the scenario models
==========================

Once the baseline scenario is created, you can streamline the workflow by using a shortcut to avoid re-running redundant steps. To do this, run:

.. code:: console

   python run/multi_run.py

You will again be prompted to specify:

* Desired time resolution (1H or 3H)
* Which non-baseline scenarios to generate (you can select more than one)
* The baseline scenario to duplicate resources from (must match the correct year)
* Whether you're using a computing cluster (yes/no)
* (If not using a cluster) The number of CPUs available on your local machine

With this setup, you can efficiently generate your own scenario results.

Analyze the model
=========================

You can use the pre-configured Jupyter notebooks located in the ``notebooks/`` directory, or any Jupyter notebook of your choice. There are two types of notebooks:

**Emission Signals Analysis**

These notebooks are located in ``notebooks/emission-signals/`` and are designed to calculate the emission signals used for the emission matching procurement strategy. More details are provided in `emissionality <file:///home/user/OET/projects/metastudy/procurement-metastudy/documentation/_build/html/12-procure-config.html>`_.

**Procurement Results Analysis**

These notebooks are located in ``notebooks/`` and are designed to analyze the procurement results, directly pulling the networks from the ``results/`` directory within the repository (but they can be easily adapted to be used outside the repository). Running these notebooks will generate ready-made overview tables and plots on energy and technology mixes, costs, emissions, and other relevant metrics. In particular:

* `energy-mix.ipynb`: evaluates the energy mix by source, bus, and carriers
* `system-level-results_and_CI-allocation.ipynb`: evaluates system-level impacts of the procurement strategies in terms of costs, emissions, operations, and capacity expansion, as well as the CI procurement allocation by carrier and bus
* `costs.ipynb`: evaluates costs and revenues for all the CI components
* `CI-time-series.ipynb`: evaluates the nodal dispatch by source of the CI procuring actors


