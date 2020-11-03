#!/bin/bash

herringrun -n 8 --homogeneous -c /shared/roshanin/conda RUN_HERRING=1 bash 8script_checkpoint.sh & python3 eval_rosh.py &
