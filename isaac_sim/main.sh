#!/bin/bash

SIM_PYTHON_PATH=" $HOME/isaacsim/_build/linux-x86_64/release/python.sh"


# $SIM_PYTHON_PATH ./main.py --condition "single_optimal" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/full_warehouse/single/roadmap_single_optimal.pkl" \
#                            --usd_path "$HOME/Downloads/map/full_warehouse_v1.usd"

# $SIM_PYTHON_PATH ./main.py --condition "dual_case_shadow" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/full_warehouse/dual/path_case_shadow/" \
#                            --usd_path "$HOME/Downloads/map/full_warehouse_v1.usd" 

# $SIM_PYTHON_PATH ./main.py --condition "dual_case_distance" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/full_warehouse/dual/path_case_distance/" \
#                            --usd_path "$HOME/Downloads/map/full_warehouse_v1.usd"



# $SIM_PYTHON_PATH ./main.py --condition "single_optimal" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/hospital/single/roadmap_single_optimal.pkl" \
#                            --usd_path "$HOME/Downloads/map/hospital_v1.usd"

$SIM_PYTHON_PATH ./main.py --condition "dual_case_shadow" \
                           --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/hospital/dual/path_case_shadow/" \
                           --usd_path "$HOME/Downloads/map/hospital_v1.usd" 

# $SIM_PYTHON_PATH ./main.py --condition "dual_case_distance" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/hospital/dual/path_case_distance/" \
#                            --usd_path "$HOME/Downloads/map/hospital_v1.usd"



# $SIM_PYTHON_PATH ./main.py --condition "single_optimal" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/office/single/roadmap_single_optimal.pkl" \
#                            --usd_path "$HOME/Downloads/map/office_v1.usd"

# $SIM_PYTHON_PATH ./main.py --condition "dual_case_shadow" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/office/dual/path_case_shadow/" \
#                            --usd_path "$HOME/Downloads/map/office_v1.usd"

# $SIM_PYTHON_PATH ./main.py --condition "dual_case_distance" \
#                            --roadmap "/home/uuugaga/_my_code/map/planning/trajectory/office/dual/path_case_distance/" \
#                            --usd_path "$HOME/Downloads/map/office_v1.usd"   