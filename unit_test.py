#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:09:24 2020

@author: mingjun
"""
from model_structure import create_model
from data_feeder import TrainSlidingWindowGenerator
import pandas as pd
import numpy as np

__appliance = 'kettle'
__test_directory = "~/mingjun/research/housedata/refit/" + __appliance + "/" + __appliance + "_test_H2.csv"
__crop = 1000
__window_size = 601
__window_offset = int((0.5 * __window_size) - 1)
def load_dataset(directory):
    """Loads the testing dataset from the location specified by file_name.

    Parameters:
    directory (string): The location at which the dataset is stored, concatenated with the file name.

    Returns:
    test_input (numpy.array): The first n (crop) features of the test dataset.
    test_target (numpy.array): The first n (crop) targets of the test dataset.

    """

    data_frame = pd.read_csv(directory, nrows=__crop, skiprows=0, header=0)
    test_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
    test_target = np.round(np.array(data_frame.iloc[__window_offset: -__window_offset, 1], float), 6)
    
    del data_frame
    return test_input, test_target
    
test_input, test_target = load_dataset(__test_directory)
model = create_model()