##########################################
Scenario Configuration
##########################################

This is the base configuration outline used in the analysis "The Role of Energy Storage in Germany".
The complete explaination of each configuration can be found in `PyPSA-Eur: Configuration <https://pypsa-eur.readthedocs.io/en/latest/configuration.html>`_.

.. note::
   The PyPSA-Eur configuration files follow a pyramid-like structure, where the parameters in the highest configuration file add to and override those in the configuration file below it.
   The order is as follows:

   1. scenarios.meta-{xH}.yaml (**this section**)
   2. `config.meta.yaml <https://open-energy-transition.github.io/procurement-metastudy/11-baseline-config.html>`_
   3. `config.default.yaml <https://pypsa-eur.readthedocs.io/en/latest/configuration.html>`_

Thus, for example changes specified in `scenario.meta.yaml` will add to and override configurations in `config.meta.yaml` and so on.