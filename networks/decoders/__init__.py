from .resnet_dec import ResNet_D_Dec, BasicBlock
from .res_shortcut_dec import ResShortCut_D_Dec
from .res_shortcut_dec import ResShortCut_D_Dec_Color

__all__ = [
    'res_shortcut_decoder_22', 
    'res_shortcut_decoder_22_2ch', 
    'res_shortcut_decoder_22_3ch',
    'res_shortcut_decoder_22_3ch_color',
]


def _res_shortcut_D_dec(block, layers, **kwargs):
    model = ResShortCut_D_Dec(block, layers, **kwargs)
    return model

def _res_shortcut_D_dec_color(block, layers, **kwargs):
    model = ResShortCut_D_Dec_Color(block, layers, **kwargs)
    return model


def res_shortcut_decoder_22(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec(BasicBlock, [2, 3, 3, 2], **kwargs)


def res_shortcut_decoder_22_2ch(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec(BasicBlock, [2, 3, 3, 2], out_channel=2, **kwargs)


def res_shortcut_decoder_22_3ch(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec(BasicBlock, [2, 3, 3, 2], out_channel=3, **kwargs)


def res_shortcut_decoder_22_3ch_color(**kwargs):
    """Constructs a resnet_encoder_14 model.
    """
    return _res_shortcut_D_dec_color(BasicBlock, [2, 3, 3, 2], out_channel=3, **kwargs)
