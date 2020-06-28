# seq2point-nilm
**Sequence-to-point (seq2point) learning for non-intrusive load monitoring (NILM)**

Seq2point learning is a generic and as well as simple framework for NILM [1-2]. It learns a mapping from the mains window Y to the midpoint x of the curresponding appliance. From probabilistic perspective, seq2point learns a condistional distribution p(x|Y) (see details in [2]).

Similarly, the seq2seq learning proposed in [2] learns a mapping from sequence to sequence which could be seen as an extension of seq2point.

Note that seq2point learning is a framework and so you can choose any architectures including CNN, RNN and AutoEncoders if you are employing deep neural networks. Indeed, you can also choose logistic regression, SVM, and Gaussian Process regression models because all these models are instances for representing a mapping.

This code is written by Mingjun Zhong adapted from Michele D'Incecco and Jack Barber:

https://github.com/MingjunZhong/transferNILM.

https://github.com/JackBarber98/pruned-nilm.
 

References:

[1] DIncecco, Michele, Stefano Squartini, and Mingjun Zhong. "Transfer Learning for Non-Intrusive Load Monitoring." arXiv preprint arXiv:1902.08835 (2019).

[2] Chaoyun Zhang, Mingjun Zhong, Zongzuo Wang, Nigel Goddard, and Charles Sutton. "Sequence-to-point learning with neural networks for nonintrusive load monitoring."
Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18), Feb. 2-7, 2018.

Seq2point model: the input is the mains windows (599 timepoints); and output is the midpoint of the corresponding appliance windows. Note that you can choose other sizes of the inpurt window, for example, 299, 399, etc.

![](images/s2p.png)


**Requirements**

0. This software was tested on Ubuntu 16.04 LTS

1. Create your virtual environment Python 3.5-3.8

2. Install Tensorflow = 2.0.0

    * Follow official instruction on https://www.tensorflow.org/install/
    
    * Remember a GPU support is highly recommended for training
    
3. Install Keras > 2.1.5 (Tested on Keras 2.3.1)

    * Follow official instruction on https://keras.io/
    
4. Clone this repository
    

For instance, the environments we used are listed in the file `environment.yml` - 
you could find all the packages there. If you use `conda`, 
you may type `conda env create -f environment.yml` to set up the environment.
    

# How to use the code and examples
With this project you will be able to use the Sequence to Point network. You can prepare the dataset from the
most common in NILM, train the network and test it. Target appliances taken into account are kettle, microwave, fridge, dish washer and
washing machine.
Directory tree:

``` bash
seq2point-nilm/
├── appliance_data.py
├── data_feeder.py
├── dataset_management/
│   ├── functions.py
│   ├── redd/
│   │   ├── create_trainset_redd.py
│   │   ├── house_plot.py
│   │   ├── redd_create_tes-set.py
│   │   ├── redd_parameters.py
│   │   └── redd_raw_plot.py
│   ├── refit/
│   │   ├── create_dataset.py
│   │   ├── dataset_infos.py
│   │   ├── dataset_plot.py
│   │   ├── excelExporter.py
│   │   ├── merge_fridges.py
│   │   └── raw_house_data_plot.py
│   └── ukdale/
│       ├── create_test_set.py
│       ├── create_trainset_ukdale.py
│       ├── excelExporterUK.py
│       ├── house_data_plot.py
│       ├── import_ext.py
│       ├── testset_plot.py
│       └── ukdale_parameters.py
├── environment.yml
├── images/
│   ├── model.png
│   ├── s2p.png
│   └── washingmachine.png
├── model.png
├── model_structure.py
├── README.md
├── remove_space.py
├── saved_models/
├── seq2point_test.py
├── seq2point_train.py
├── test_main.py
└── train_main.py
```

## **Create REFIT, UK-DALE or REDD dataset**

This script allows the user to create CSV files of training dataset of power measurments.
The output will be 3 CSV files for training, validation and test. 

You should select the following arguments for the argument parser:
`python create_dataset -h`

```
--data_dir DATA_DIR             The directory containing the CLEAN REFIT data

--appliance_name APPLIANCE_NAME which appliance you want to train: kettle,
                                microwave,fridge,dishwasher,washingmachine

--aggregate_mean AGGREGATE_MEAN Mean value of aggregated reading (mains)

--aggregate_std AGGREGATE_STD   Std value of aggregated reading (mains)

`--save_path SAVE_PATH           The directory to store the training data
```


Example:

Create a REFIT dataset (mains and appliance power measurments) for kettle:

`python create_dataset.py --data_dir './' --appliance_name 'kettle' --aggregate_mean 522 --aggregate_std 814 --save_path './'`
    
### **REFIT**

Download the REFIT raw data from the original website (https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned). 
Appliances and training set composition for this project:

| Appliances      |      training                    |  validation | test   |
|-----------------|:--------------------------------:|:-----------:|:------:|
| kettle          | 3, 4, 6, 7, 8, 9, 12, 13, 19, 20 |     5       |   2    |
| microwave       | 10, 12, 19                       |    17       |   4    |
| fridge          | 2, 5, 9                          |     12      |   15   |
| dish washer     | 5, 7, 9, 13, 16                  |     18      |   20   |
| washing machine | 2, 5, 7, 9, 15, 16, 17           |      18     |   8    |


### **UK-DALE**

Download the UK-DALE raw data from the original website (http://jack-kelly.com/data/). 
Validation is a 13% slice from the final training building. 
Appliances and training set composition for this project:

| Appliances      |      training   |  validation | test   |
|-----------------|:---------------:|:-----------:|:------:|
| kettle          | 1               |     1       |   2    |
| microwave       | 1               |     1       |   2    |
| fridge          | 1               |     1       |   2    |
| dishwasher      | 1               |     1       |   2    |
| washingmachine  | 1               |     1       |   2    |


### **REDD**

Download the REDD raw data from the original website (http://redd.csail.mit.edu/).
Validation is a 10% slice from the final training building. 
Appliances and training set composition for this project:

| Appliances      |      training   |  validation | test   |
|-----------------|:---------------:|:-----------:|:------:|
| microwave       | 2,3             |     3       |   1    |
| fridge          | 2,3             |     3       |   1    |
| dishwasher      | 2,3             |     3       |   1    |
| washingmachine  | 2,3             |     3       |   1    |


**I will write instructions how to use the code with more details. Currently, you just run train_main.py and test_main.py. Do remember to choose your parameters in these two files correspondingly.**

To train the modoel, just run `python train_main.py` or in IDE environment, e.g., Spyder, run train_main.py

Any questions, please write email to me: mingjun.zhong@abdn.ac.uk
