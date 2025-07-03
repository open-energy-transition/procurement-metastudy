##########################################
Features
##########################################

Here is a list of changes made to this repository specifically for this work, with the potential to be upstreamed to the main PyPSA-Eur repository.

**Baseline Enhancements**

* Cherry-picked storage features and technologies from Form Energy Storage Project (https://github.com/open-energy-transition/procurement-metastudy/pull/2)

* Added global and national renewable energy share targets (https://github.com/open-energy-transition/procurement-metastudy/pull/11)

* Implemented coal phase-out through ``powerplants_filter`` (https://github.com/open-energy-transition/procurement-metastudy/pull/15)

* Added ``res_additionallity`` (https://github.com/open-energy-transition/procurement-metastudy/pull/16)

* Improved retrieving of CI load input data (https://github.com/open-energy-transition/procurement-metastudy/pull/17)

* Added national renewable capacity targets (https://github.com/open-energy-transition/procurement-metastudy/pull/18)

* Established a standardized background scenario prior to implementing procurement strategies (https://github.com/open-energy-transition/procurement-metastudy/pull/22)

* Added the impact of electricity imports from neighboring buses into the grid quality score (https://github.com/open-energy-transition/procurement-metastudy/pull/23)

* Set capacities based on the TYNDP transmission grid (https://github.com/open-energy-transition/procurement-metastudy/pull/24)

* Added ``freeze_capacity`` option to represent system conditions in 2025 (https://github.com/open-energy-transition/procurement-metastudy/pull/27)

* Added a new constraint called ``import_constraints`` (https://github.com/open-energy-transition/procurement-metastudy/pull/30)

* Centralized all CI-related processes under a new Snakemake rule named ``add_procurement`` (https://github.com/open-energy-transition/procurement-metastudy/pull/31)

* Added a new procurement scope: ``continent`` (https://github.com/open-energy-transition/procurement-metastudy/pull/34)

* Added ``cap_premium`` to support optimal spatial allocation of CI procurement technologies (https://github.com/open-energy-transition/procurement-metastudy/pull/36)

**Procurement Strategy Development**

* Added annual volume matching strategy (https://github.com/open-energy-transition/procurement-metastudy/pull/8)

* Added 24/7 carbon-free energy strategy (https://github.com/open-energy-transition/procurement-metastudy/pull/9)

* Added mmission-matching constraint and retrieve CI load share from Eurostat (https://github.com/open-energy-transition/procurement-metastudy/pull/12)

* Improved emission_matching_constraints and refactored the CI load workflow (https://github.com/open-energy-transition/procurement-metastudy/pull/26)

**Model Usability Improvements**

* Developed the first command-line interface for running the model (https://github.com/open-energy-transition/procurement-metastudy/pull/25)

* Added user documentation via GitHub Pages

**Bug Fixes**

* Fixed issues with emissionality signal handling (https://github.com/open-energy-transition/procurement-metastudy/pull/28)

* Removed requirement for specifying a procurement year in ``extra_functionality`` (https://github.com/open-energy-transition/procurement-metastudy/pull/29)

* Corrected configuration loading to enable procurement via ``config`` instead of ``params`` (https://github.com/open-energy-transition/procurement-metastudy/pull/33)

* Addressed final scenario preparation bug (https://github.com/open-energy-transition/procurement-metastudy/pull/35)