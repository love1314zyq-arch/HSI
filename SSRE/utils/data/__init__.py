from utils.data.prepare_data import prepare_CL, prepare_HSI
from utils.data.prepare_loader import prepare_loader_normal


datatypes = {
    'cl': prepare_CL,
    'hsi_cl': prepare_HSI,
}

loadertypes = {
    'normal': prepare_loader_normal,
}
