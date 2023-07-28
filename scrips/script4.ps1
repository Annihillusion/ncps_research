conda activate ncps
cd E:\ncps_research

# python visual_weights.py --num_neurons 8
# python visual_weights.py --num_neurons 12
# python visual_weights.py --num_neurons 16
# python visual_weights.py --num_neurons 20
# python visual_weights.py --num_neurons 24
# python visual_weights.py --num_neurons 28
# python visual_weights.py --num_neurons 32
#
# python visual_weights.py --num_neurons 8 --suffix _noRecur
# python visual_weights.py --num_neurons 12 --suffix _noRecur
# python visual_weights.py --num_neurons 16 --suffix _noRecur
# python visual_weights.py --num_neurons 20 --suffix _noRecur
# python visual_weights.py --num_neurons 24 --suffix _noRecur
# python visual_weights.py --num_neurons 28 --suffix _noRecur
#
# python visual_weights.py --num_neurons 20 --suffix _fixTC
# python visual_weights.py --num_neurons 32 --suffix _fixTC

python visual_weights.py --num_neurons 16 --suffix addition0
python visual_weights.py --num_neurons 16 --suffix addition1
python visual_weights.py --num_neurons 16 --suffix addition2
python visual_weights.py --num_neurons 16 --suffix addition3
python visual_weights.py --num_neurons 16 --suffix addition4