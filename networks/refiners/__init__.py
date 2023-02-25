from .refiner import MultiInstRefiner

__all__ = [
    'multi_inst_refiner',
]


def multi_inst_refiner():
    return MultiInstRefiner(32, 16)
