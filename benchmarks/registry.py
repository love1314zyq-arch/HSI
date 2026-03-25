from benchmarks.adapters import FEICACILAdapter, FeTrILAdapter, GFRILAdapter, HyperKDAdapter, LWFAdapter, OursAdapter, PlaceholderAdapter, SSREAdapter


def build_registry():
    return {
        "ours": OursAdapter(),
        "feica_cil": FEICACILAdapter(),
        "hyperkd": HyperKDAdapter(),
        "ssre": SSREAdapter(),
        "lwf": LWFAdapter(),
        "fetril": FeTrILAdapter(),
        "gfr_il": GFRILAdapter(),
    }
