import sys

def select_model(arg_obj):
    model_type = arg_obj.model_type.lower()
    channels = arg_obj.channels
    input_channels = len(channels)
    dropout = float(arg_obj.dropout)

    if model_type == 'rpnet':
        from models.RPNet import RPNet
        model = RPNet(input_channels=input_channels, drop_p=dropout)
    elif model_type == 'physnet_rppg_enc':
        # Encodeur du PhysNet de rPPG-Toolbox ; pas de dropout, comme au preentrainement.
        from models.PhysNetRPPGEncoder import PhysNetRPPGEncoder
        pretrained = None if str(arg_obj.pretrained).lower() in ('', 'none') else arg_obj.pretrained
        model = PhysNetRPPGEncoder(input_channels=input_channels, pretrained=pretrained, drop_p=0.0)
    elif model_type == 'physnet':
        from models.PhysNet import PhysNet
        model = PhysNet(input_channels=input_channels, drop_p=dropout)
    else:
        print('Could not find model specified.')
        sys.exit(-1)

    return model
