import pandas as pd
import numpy as np


def setImagename(path):
    df = pd.read_csv(path)

    # Update the 'path' column by removing the '/kaggle/working/' prefix
    df['path'] = df['path'].str.replace('/kaggle/working/', '', regex=False)

    # Save the updated DataFrame if needed
    df.to_csv("updatedValidation.csv", index=False)

    print("Updated 'path' column:")
    print(df['path'].head())

if __name__ =="__main__":
    setImagename("C:/Users/bhara/Desktop/clothfilter/Dataset/DeepFashion2/input/validation.csv")

