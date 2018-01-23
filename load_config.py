config_path = 'config.txt'
def set_config(config_path):
    args = edict()
    with open(config_path) as source:
        for line in source:
            line = line.strip()
            arg,value = line.split('=')
            arg = arg.strip()
            value = arg.strip()
            value = int(value)
            args.arg = value
    return args

print(args)