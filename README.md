# Решение для кейса "Определение вида отходов строительства в кузове транспортного средства"

```train.py``` - для тренировки - коннстанта DATASET_PATHS - определяет папку с данными (в папке должны быть train, test, val подпапки)

```inference.py``` - для инференса моделью на вход подается папка с видео и два пути к весам для двух моделей

```visualize_opticalflow.py```
пример:

```python inference.py videos checkpoints/epoch=3-val_loss=0.29499.ckpt  checkpoints/epoch=9-val_loss=0.01546.ckpt```