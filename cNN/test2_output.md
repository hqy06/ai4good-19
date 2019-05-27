Train Epoch: 0 [6400/42000 (15%)] Loss: 1.980280
Train Epoch: 0 [12800/42000 (30%)] Loss: 1.853076
Train Epoch: 0 [19200/42000 (46%)] Loss: 1.815983
Train Epoch: 0 [25600/42000 (61%)] Loss: 1.740305
Train Epoch: 0 [32000/42000 (76%)] Loss: 1.769417
Train Epoch: 0 [38400/42000 (91%)] Loss: 1.783690

Average loss: 1.7292, Accuracy: 30063/42000(71.000%)

```
C:\Users\cappu\AppData\Local\Programs\Python\Python37\lib\site-packages\torch\nn_reduction.py:46: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
warnings.warn(warning.format(ret))
```

---

````

n_epochs = 1
for epoch in range(n_epochs):
train(epoch)
evaluate(train_loader)

```

```

Train Epoch: 0 [6400/42000 (15%)] Loss: 2.034397
Train Epoch: 0 [12800/42000 (30%)] Loss: 1.836204
Train Epoch: 0 [19200/42000 (46%)] Loss: 1.846570
Train Epoch: 0 [25600/42000 (61%)] Loss: 1.770751
Train Epoch: 0 [32000/42000 (76%)] Loss: 1.923445
Train Epoch: 0 [38400/42000 (91%)] Loss: 1.814658
C:\Users\cappu\AppData\Local\Programs\Python\Python37\lib\site-packages\ipykernel_launcher.py:42: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
C:\Users\cappu\AppData\Local\Programs\Python\Python37\lib\site-packages\torch\nn_reduction.py:46: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
warnings.warn(warning.format(ret))

```

```
````
