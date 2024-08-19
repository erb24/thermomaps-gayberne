#!/bin/bash

awk 'NF==9 { print $0}' tee.log | grep "[0-9] * [0-9]" | grep -v "[A-Z]" > thermo.txt
