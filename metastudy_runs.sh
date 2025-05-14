#!/bin/bash

snakemake --profile slurm solve_sector_networks --configfile config/config.meta_PL.yaml --rerun-trigger mtime
snakemake --profile slurm solve_sector_networks --configfile config/config.meta_DE.yaml --rerun-trigger mtime
snakemake --profile slurm solve_sector_networks --configfile config/config.meta_FR.yaml --rerun-trigger mtime
snakemake --profile slurm solve_sector_networks --configfile config/config.meta_ES.yaml --rerun-trigger mtime
snakemake --profile slurm solve_sector_networks --configfile config/config.meta_DK.yaml --rerun-trigger mtime
snakemake --profile slurm solve_sector_networks --configfile config/config.meta_IE.yaml --rerun-trigger mtime
snakemake --profile slurm solve_sector_networks --configfile config/config.meta_PL-DE-FR-ES-DK-IE.yaml --rerun-trigger mtime
