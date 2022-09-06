#!/usr/bin/env bash
### Quickly test train_vae and analyze with default parameters (most common commands)

# set shell output
set -e
set -x
rm -rf output
mkdir -p output

tomodrgn train_vae data/10076_both_32_sim.star -o output/01_vae_both_sim --zdim 8 --uninvert-data --seed 42 --log-interval 100 --enc-dim-A 64 --enc-layers-A 2 --out-dim-A 64 --enc-dim-B 32 --enc-layers-B 4 --dec-dim 256 --dec-layers 3 -n 10
tomodrgn analyze output/01_vae_both_sim 9 --Apix 13.1 --ksample 2

echo "train_vae and analyze functional"
