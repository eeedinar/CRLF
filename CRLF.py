import pandas as pd
import concurrent.futures
import time
import shutil
import argparse
from functools import partial
from essential_func import *


def compare_groud_truth(output_directory, folder):

    ### Parameters and Analysis of model with the labels.csv file for CRLF only
    gnd_file = 'labels.csv'
    df_col_name_category = 'Category'
    df_col_name_file = 'File'
    df_col_name_call_start = 'Start'
    frog_type1 = 'CRLF'
    frog_type2 = 'BOTH'
    gen_comparison_file = 'comparison.csv'

    ### read labels.csv, and results.csv files and write comparison.csv file
    df2 = pd.read_csv(gnd_file)                     # read labels file
    df2 = df2[(df2[df_col_name_category]==frog_type1 ).values + (df2[df_col_name_category]==frog_type2).values]            # filter only CRLF only
    df2.sort_values(by=[df_col_name_file], inplace=True)      # sort dataframe to be similar with df
    n   = len(np.unique(df2[df_col_name_file].tolist()))      # get unique files 
    X = np.zeros((n,2),dtype=object)

    grp = df2.groupby(df_col_name_file)                       # group by file name only

    for idx, ng in enumerate(grp):
        name, group = ng
        X[idx,0] = name
        X[idx,1] = np.array(group[df_col_name_call_start])
        
    df = pd.read_csv(os.path.join(output_directory,folder,'results.csv'))         # read CRLF analyzed file    
    df2 = pd.DataFrame(X)
    df2.columns = [df_col_name_file, df_col_name_call_start]
    df_outer = pd.merge(df, df2, on = df_col_name_file, how="outer")   # make sure all values are retained
    df_outer.columns = ['File', 'Analyzed', 'Ground']
    df_outer.to_csv(os.path.join(output_directory,folder,gen_comparison_file), index=False,)

    print("Analysis is completed")

def main(args):
    print(f"Program started on {time.strftime('%m/%d/%Y %A %H:%M:%S', time.localtime())}")

    #### Parameters - showing plot and results
    show_plot    = False
    show_results = False

    ### create folder for data output
    folder = f"{time.strftime('%Y-%m-%d-%A-%H-%M-%S', time.localtime())}"

    ### reading parameters and setting it to variables
    file = open('parameters.txt','r')
    file.seek(0)
    parameters = file.readlines()
    file.close()

    ### set variables from parameters.txt
    for i in range(len(parameters)):
        var, val = parameters[i].split("=")[0].strip() , parameters[i].split("=")[1].strip()
        try:
            globals()[var] = int(val)
        except:
            globals()[var] = val

    ### create folder and copy parameters.txt
    os.makedirs(os.path.join(output_directory,folder))
    shutil.copy2('parameters.txt', os.path.join(output_directory,folder))     # copy parameters.txt to the folder

    ### look for wav and WAV extensions in the specified directory and produce results.csv file
    files_sorted = cwd_files_search_with('.wav', directory = audio_directory)
    files_sorted.append(cwd_files_search_with('.WAV', directory = audio_directory))
    files_sorted = flatten(files_sorted)
    n = len(files_sorted)
    X = np.zeros((n,2),dtype=object)

    ### Serial Processing - iterate over file and do signal detection
    if args.mode == 'serial':
        for idx, file in enumerate(files_sorted):
            freq, t = signal_detection(filename = os.path.join(audio_directory, file), sr= sr, fmax=fmax, n_mels=n_mels, dB_thr=dB_thr, freq_input=freq_input, n_serials=n_serials, block=block, show_plot=show_plot, show_results=show_results, play_audio=play_audio )
            X[idx,0] = file
            X[idx,1] = t  

    ### Multi-Processing - iterate over file and do signal detection
    elif args.mode == 'parallel':
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            arg = [os.path.join(audio_directory, file) for file in files_sorted]
            worker = partial(signal_detection, sr= sr, fmax=fmax, n_mels=n_mels, dB_thr=dB_thr, freq_input=freq_input, n_serials=n_serials, block=block, show_plot=show_plot, show_results=show_results, play_audio=play_audio )
            results = executor.map(worker, arg)

            for idx, (file, t) in enumerate(zip(files_sorted, results)):
                X[idx,0] = file
                freq, X[idx,1] = t  

    ### generates results.csv and parameters to the folder
    df = pd.DataFrame(X)
    df.columns = ['File', 'Start']
    df.sort_values(by=['File'], inplace=True)
    df.to_csv(os.path.join(output_directory,folder,'results.csv'), index=False)

    ### log the outcome of this program
    file = open(os.path.join(output_directory,folder,'output.log'), 'w')
    file.write(f'Frequency used for computation is {freq:0.3f} Hz\nProgram was executed in {args.mode} mode')
    file.close()

    return output_directory, folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for CRLF detection program")
    parser.add_argument('-m', '--mode',   type=str,   metavar='', required=False, default="parallel",  choices = ["serial", "parallel"],help = "serial or parallel operation on directory")
    args = parser.parse_args()

    output_directory, folder = main(args)
    debug = False
    if debug:
        try:
            compare_groud_truth(output_directory, folder)
        except:
            print("debug failed")
    
    print(f"Program completed on {time.strftime('%m/%d/%Y %A %H:%M:%S', time.localtime())}")