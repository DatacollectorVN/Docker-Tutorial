import yaml
import argparse
def main(yml_file_path):
    with open(yml_file_path) as file:
        params = yaml.load(file, Loader = yaml.FullLoader)

    print(params)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ymlfile", dest = "yml_file", type = str,
                        default = None, help = "Yml file path")
    args = parser.parse_args()
    main(args.yml_file)

