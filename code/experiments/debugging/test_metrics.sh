#!/bin/bash

python -m experiments.thesis.domain_shift --split test_r\
    --max_samples_per_class 30
