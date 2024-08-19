#!/bin/bash

DUMP=$1
grep -A 1 "TIMESTEP" $DUMP | grep  "[0-9]" > timesteps.txt
