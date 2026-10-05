AGE_LABELS = [
    "0-12",
    "13-19",
    "20-29",
    "30-39",
    "40-54",
    "55+",
]

NUM_AGE_CLASSES = len(AGE_LABELS)


def age_to_class(age):
    if age <= 12:
        return 0
    if age <= 19:
        return 1
    if age <= 29:
        return 2
    if age <= 39:
        return 3
    if age <= 54:
        return 4
    return 5