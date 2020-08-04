# JoCoR/Co-teaching

## Enviornments

 - python 3.7
 - PyTorch 1.3.1
 - torchvision 0.4.2
 - Numpy
 
## How to run

```
git clone https://github.com/Moukahou/JoCoR.git
```
and then 
```
cd JoCoR
python train.py
```

you can adjust the noise rate of the training data and some other parameters by modifying the config.py

for example, if you want to try original co-teaching, you can just set `loss = 'co-teaching'`  in config.py

Now this implementation support co-teaching and JoCoR loss.

