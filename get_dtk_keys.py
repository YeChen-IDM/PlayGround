import json
import os

def application( output_folder="output", stdout_filename="StdOut.txt", insetchart_name="InsetChart.json",
                 config_filename="config.json", campaign_filename="campaign.json",
                 debug=False):
    with open(config_filename) as infile_1:
        cdj = json.load(infile_1)["parameters"]
    with open("configkeys.txt", 'w') as file:
        for param in cdj:
            file.write(f"{param} = '{param}'\n")

    with open(os.path.join(output_folder, insetchart_name)) as infile_2:
        isj = json.load(infile_2)["Channels"]
    with open("insetchartkeys.txt", 'w') as file:
        for param in isj:
            paramkey = param
            for ch in [' (', ') ', ' ']:
                paramkey = str(paramkey).replace(ch, '_')
            file.write(f"{paramkey} = '{param}'\n")

if __name__ == "__main__":
    # execute only if run as a script
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default="output", help="Folder to load outputs from (output)")
    parser.add_argument('-s', '--stdout', default="StdOut.txt", help="Name of stdoutfile to parse (StdOut.txt)")
    parser.add_argument('-j', '--jsonreport', default="InsetChart.json", help="Json report to load (InsetChart.json)")
    parser.add_argument('-c', '--config', default="config.json", help="Config name to load (config.json)")
    parser.add_argument('-C', '--campaign', default="campaign.json", help="campaign name to load (campaign.json)")
    parser.add_argument('-d', '--debug', help="debug flag", action='store_true')
    args = parser.parse_args()


    application(output_folder=args.output, stdout_filename=args.stdout, insetchart_name=args.jsonreport,
                config_filename=args.config, campaign_filename=args.campaign,
                debug=args.debug)
