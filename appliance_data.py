""" A collection of all appliances available. Each appliance has a consumption mean 
and standard deviation, as well as a list of houses containing the appliance. """

appliance_data = {
    "kettle": {
        "mean": 700,
        "std": 1000,
        "houses": [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
        "channels": [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
        "test_house": 2,
        "validation_house": 5,
    },
    "fridge": {
        "mean": 200,
        "std": 400,
        "houses": [2, 5, 9, 12, 15],
        "channels": [1, 1, 1, 1, 1],
        "test_house": 15,
        "validation_house": 12
    },
    "dishwasher": {
        "mean": 700,
        "std": 1000,
        "houses": [5, 7, 9, 13, 16, 18, 20],
        "channels": [4, 6, 4, 4, 6, 6, 5],
        "test_house": 9,
        "validation_house": 18,     
    },
    "washingmachine": {
        "mean": 400,
        "std": 700,
        "houses": [2, 5, 7, 8, 9, 15, 16, 17, 18],
        "channels": [2, 3, 5, 4, 3, 3, 5, 4, 5],
        "test_house": 8,
        "validation_house": 18,
    },
    "microwave": {
        "mean": 500,
        "std": 800,
        "houses": [4, 10, 12, 17, 19],
        "channels": [8, 8, 3, 7, 4],
        "test_house": 4,
        "validation_house": 17,
    },
}

# The std and mean values for normalising the mains data which are used
# as input to the networks
mains_data = {
    "mean": 522,
    "std":  814        
    }