
import numpy as np

def get_aromatic_rings_value(value):
    if value == 0:
        return 0.336376
    elif value == 1:
        return 0.816016
    elif value == 2:
        return 1
    elif value == 3:
        return 0.691115
    elif value == 4:
        return 0.199399
    elif value > 4:
        return 0
    else:
        raise ValueError("Aromatic rings must not be", value)


def get_heavy_atoms(value):
    if value > 5 and value <= 45:
        return (1 / 0.624231) * (0.0000443 * value ** 3 - 0.004556 * value ** 2 + 0.12775 * value - 0.463)
    else:
        return 0


def get_mwhbn(value):
    if value >= 0.05 and value <= 0.45:
        return (1 / 0.72258) * (26.733 * value ** 3 - 31.495 * value ** 2 + 9.5202 * value - 0.1358)
    else:
        return 0


def get_tpsa(value):
    if value > 0 and value <= 120:
        return (1 / 0.9598) * (-0.0067 * value + 0.9598)
    else:
        return 0


def get_pka(value):
    if value > 3 and value <= 11:
        return (1 / 0.597488) * (
                0.00045068 * value ** 4 - 0.016331 * value ** 3 + 0.18618 * value ** 2 - 0.71043 * value + 0.8579)
    else:
        return 0


def calculate_gupta_score(_data_array: np.ndarray) -> np.float:
    mw = _data_array[0]
    hbd = _data_array[2]
    hba = _data_array[3]

    hac = get_heavy_atoms(_data_array[6])
    ar = get_aromatic_rings_value(_data_array[9])
    tpsa = get_tpsa(_data_array[5])
    pka = get_pka(_data_array[7])
    mwhbn = get_mwhbn(mw ** (-0.5) * (hba + hbd))

    return ar + hac + 1.5 * mwhbn + 2 * tpsa + 0.5 * pka

