conda activate ncps
cd E:\ncps_research
# python record_test.py --num_neurons 8
# python record_test.py --num_neurons 12
# python record_test.py --num_neurons 16
# python record_test.py --num_neurons 20
# python record_test.py --num_neurons 24
# python record_test.py --num_neurons 28
# python record_test.py --num_neurons 32
#
# python record_test.py --num_neurons 8 --suffix _noRecur
# python record_test.py --num_neurons 12 --suffix _noRecur
# python record_test.py --num_neurons 16 --suffix _noRecur
# python record_test.py --num_neurons 20 --suffix _noRecur
# python record_test.py --num_neurons 24 --suffix _noRecur
# python record_test.py --num_neurons 28 --suffix _noRecur
#
# python record_test.py --num_neurons 20 --suffix _fixTC
# python record_test.py --num_neurons 32 --suffix _fixTC
#
# python record_test.py --num_neurons 8 --suffix _50epochs
# python record_test.py --num_neurons 12 --suffix _50epochs

python record_test.py --num_neurons 16 --suffix _addition0
python record_test.py --num_neurons 16 --suffix _addition1
python record_test.py --num_neurons 16 --suffix _addition2
python record_test.py --num_neurons 16 --suffix _addition3
python record_test.py --num_neurons 16 --suffix _addition4



