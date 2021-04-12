import os

def configuration(f_config):

    Config = open(f_config)
    for line in Config:
        line = line.rstrip()
        p = line.split(' = ')[0]
        val = line.split(' = ')[1]

        if p == 'mode':
            mode = val
            print(f'mode [{mode}]')
        elif p == 'checkpoint':
            checkpoint = val
            print(f'checkpoint [{checkpoint}]')
        elif p == 'prefix':
            prefix = val
            print(f'prefix [{prefix}]')

        elif p == 'train_list':
            train_list = val
            print(f'train_list [{train_list}]')
        elif p == 'valid_list':
            valid_list = val
            print(f'valid_list [{valid_list}]')
        elif p == 'test_list':
            test_list = val
            print(f'test_list [{test_list}]')

        elif p == 'pocket_dir':
            pocket_dir = val
            print(f'pocket_dir [{pocket_dir}]')

        elif p == 'max_poc_node':
            max_poc_node = int(val)
            print(f'max_poc_node [{max_poc_node}]')

        elif p == 'epochs':
            epochs = int(val)
            print(f'epochs [{epochs}]')
        elif p == 'learning_rate':
            lr = float(val)
            print(f'learning_rate [{lr}]')
        elif p == 'batch_size':
            batch_size = int(val)
            print(f'batch_size [{batch_size}]')
        elif p == 'l2_regularization':
            l2_param = float(val)
            print(f'L2 regularization [{l2_param}]')


    if mode == 'train':
        return train_list, valid_list, pocket_dir, max_poc_node, epochs, lr, batch_size, l2_param, checkpoint, prefix
    elif mode == 'eval' or mode == 'featurize':
        return test_list, pocket_dir, max_poc_node, batch_size, checkpoint, prefix