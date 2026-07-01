#!/usr/bin/env bash

python src/ocularrigidity/scripts/registration/infer.py
python src/ocularrigidity/scripts/pulsation/infer.py
python src/ocularrigidity/scripts/cohort_analysis/segment_n_cycles.py
python src/ocularrigidity/scripts/cohort_analysis/extract_deltaA.py
python src/ocularrigidity/scripts/cohort_analysis/flag_misregistration.py