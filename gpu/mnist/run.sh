# Run main.py with many arguments
# Usage: sh run.sh

# For training resnet model
# python main.py --epochs 5 --model-name resnet
# For plotting resnet results
# python main.py --epochs 5 --model-name resnet --read-only

python main.py --epochs 5 --batch-size 64 --lr 1.0 --gamma 0.7 --model-name resnet
python main.py --epochs 5 --batch-size 64 --lr 1.0 --gamma 0.7 --model-name resnet --reverse-data
python main.py --epochs 5 --batch-size 64 --lr 1.0 --gamma 0.7 --model-name resnet --reverse-data --disable-dropout
python main.py --epochs 5 --batch-size 64 --lr 1.0 --gamma 0.7 --model-name resnet --disable-dropout
python main.py --epochs 5 --batch-size 64 --lr 1.0 --gamma 0.7 --model-name resnet --merge-plot


# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 0.7
# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 0.7 --reverse-data
# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 0.7 --reverse-data --disable-dropout
# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 0.7 --disable-dropout
# python merge_plots.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 0.7

# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7
# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7 --reverse-data
# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7 --reverse-data --disable-dropout
# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7 --disable-dropout
# python merge_plots.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7

# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7 --no-cuda
# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7  --no-cuda --reverse-data
# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7 --no-cuda --reverse-data --disable-dropout
# python main.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7 --no-cuda --disable-dropout
# python merge_plots.py --epochs 5 --lr 1.0 --batch-size 2048 --gamma 0.7 --no-cuda

# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 1.0
# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 1.0 --reverse-data
# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 1.0 --reverse-data --disable-dropout
# python main.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 1.0 --disable-dropout
# python merge_plots.py --epochs 5 --lr 1.0 --batch-size 1000 --gamma 1.0
