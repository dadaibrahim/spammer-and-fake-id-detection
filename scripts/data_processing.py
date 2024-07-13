import pandas as pd

def read_datasets():
    """Reads users profile from csv files."""
    genuine_users = pd.read_csv("data/users.csv")
    fake_users = pd.read_csv("data/fusers.csv")
    x = pd.concat([genuine_users, fake_users])
    y = [0] * len(fake_users) + [1] * len(genuine_users)
    return x, y
