conda activate ncps
cd E:\ncps_research
python synthetic_fit.py --num_neurons 8 --connect_policy ncp
python synthetic_fit.py --num_neurons 16 --connect_policy ncp
python synthetic_fit.py --num_neurons 24 --connect_policy ncp
python synthetic_fit.py --num_neurons 32 --connect_policy ncp
