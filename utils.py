import json
import pandas as pd


def json_to_df(filepath: str):
    """
    convert json to pandas dataframe
    """
    data_file = open(filepath)
    data = []
    for line in data_file:
        data.append(json.loads(line))
    data_file.close()

    assert len(data) > 0, "json file is empty"
    return pd.DataFrame(data)
