def segment_data(data, column, values_group_a, values_group_b):
    """Segment data into control (Group A) and test (Group B) for comparison."""

    group_a = data[data[column].isin(values_group_a)]
    # group_a = data[data[column] == value_a]
    group_b = data[data[column].isin(values_group_b)]
    # group_b = data[data[column] == value_b]
    return group_a, group_b

def aggregate_by_group(data, group_column, metrics):
    """Aggregate data by group and calculate the provided metrics."""

    return data.groupby(group_column)[metrics].agg(["sum", "mean"]).reset_index()