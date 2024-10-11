import pandas as pd


def distance_pixel_to_meter(pixel_distance, height_meters, height_pixels):
    return (pixel_distance * height_meters) / height_pixels


def distance_meter_to_pixel(meter_distance, height_meters, height_pixels):
    return (meter_distance * height_pixels) / height_meters


def convert_to_dataframe(array):
    # .get(1, []) returns [], when there is no ball position
    # and a list of elements it there is a position
    array = [x.get(1, []) for x in array]
    return pd.DataFrame(array, columns=['x1', 'y1', 'x2', 'y2'])


def changes_detector(df, column, start, end):
    negative_change = False
    positive_change = False

    if (
        df[column].iloc[start] > 0
        and df[column].iloc[end] < 0
    ):
        negative_change = True
    if (
        df[column].iloc[start] < 0
        and df[column].iloc[end] > 0
    ):
        positive_change = True

    return negative_change, positive_change
