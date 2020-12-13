# Deep Lagrangian Networks (DeLaN)

Simple PyTorch implementation for [Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning](https://arxiv.org/abs/1907.04490v1).
Note that the model in this repo is slightly different from the one in the original paper, and for simplicity we use automatic differentiation to implement the derivative operation. We introduced and improved this model at [the 63rd Japan Joint Automatic Control Conference](https://www.sice.jp/rengo63/).

## Quick start

```
git clone git@github.com:xth430/DeepLagrangian.git
cd DeepLagrangian
pip install -r requirements.txt
```

## Dataset setup
Generate cosine trajectory datasets by running:
```
python gen_cosine_data.py
```

## Train and test the model 
```
python main.py --epochs 200 --lr 0.005
```
For detailed arguments please refer to `main.py`.