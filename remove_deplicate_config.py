import json
import os
from typing import Union


def remove_duplicate_key(config, path='.'):
    file_name = os.path.join(path, config)
    try:
        with open(file_name, 'r') as infile:
            cfg = json.load(infile)
        with open(file_name, 'w') as outfile:
            json.dump(cfg, outfile, indent=4, sort_keys=True)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"{file_name} is not a valid json file.")
    except FileNotFoundError:
        raise ValueError(f"{file_name} is not exist.")


def application(config: str = 'config.json', suite: Union[str, None] = None):
    """
    remove duplicate keys and sort config file in all paths listed in the suite file.
    :param config:
    :param suite:
    :return:
    """
    if not suite:
        remove_duplicate_key(config)
    elif os.path.isfile(suite):
        try:
            with open(suite, 'r') as suitefile:
                sf = json.load(suitefile)
            for key in sf:
                if key == 'tests' or key == 'science':
                    for path_dict in sf[key]:
                        path = path_dict['path']
                        remove_duplicate_key(config, path)
        except (KeyError, json.decoder.JSONDecodeError):
            raise ValueError(f"suite = {suite} is not a valid suite file.")
    else:
        raise ValueError(f"suite = {suite} is not a valid input.")
    print(f"Done with removing duplicate parameters in {config}.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.json', help='config.json file path')
    parser.add_argument('-s', '--suite', default=None, help='json file contains simulation paths(generic.json)')

    args = parser.parse_args()
    application(config=args.config, suite=args.suite)
