from easydict import EasyDict as edict


def set_config(config_path):
    with open(config_path) as source:
        for line in source:
            line = line.strip()
            arg,value = line.split('=')
            arg = arg.strip()
            value = value.strip()
            if value == 'True' or value == 'False':
                value = bool(value)
            else:
                value = int(value)
            var = exec("%s = %d" % (arg,value))
            args.var = value
    return args

config_path = 'config.txt'
args = edict({'seq_len':30, 'offset':4, 'cuda':False, 'batch_size':1, 'num_layers':3, 'hidden_size':128})
set_config(config_path)
print(args)
print(args.seq_len)