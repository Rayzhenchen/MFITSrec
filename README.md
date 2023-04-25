# MFITSrec: Multi-Aspect Features of Items for Time-Ordered Sequential Recommendation
The code is tested under a Windows 11 desktop (w/ RTX 3060 GPU) with PyTorch 1.11.0 and Python 3.8.

## Datasets
Our data file (e.g. `data/Beauty_item.txt`) contains one `user id`, `item id`,`timestamp`, `category`, and `brand` per line, representing an interaction along with two features of the corresponding item and a timestamp of the interaction.

Among them, `user id`, `item id`, `category`, and `brand` start from 1.
* Example
```
    1 10077 1405296000 1116 66
    1 11753 1405296000 14 66
    1 9450 1405296000 713 66
    1 11864 1405296000 14 66
    1 9840 1405296000 1116 66
    1 11156 1405296000 713 66
```
## Model Training

To train our model on `Beauty` (with default hyper-parameters): 

```
python main.py --dataset=Beauty_item --train_dir=default
```
