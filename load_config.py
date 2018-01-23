from easydict import EasyDict as edict


def set_config(config_path):
    with open(config_path) as source:
        for line in source:
            line = line.strip()
            arg, value = line.split('=')
            arg = arg.strip()
            value = value.strip()
            if value == 'True':
                value == True
            elif value == 'False':
                value = False
            else:
                value = int(value)

            args[arg] = value
    return args

config_path = 'config.txt'
args = edict()
set_config(config_path)
print(args)
print(args.seq_len)