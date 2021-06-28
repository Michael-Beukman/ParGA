from typing import List, Tuple
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
FILES = {
    'mpi_standard_sa': [
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-06-07-21__ranks_1.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-06-15-35__ranks_2.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-06-22-10__ranks_4.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-06-28-21__ranks_8.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-06-34-34__ranks_14.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-06-42-37__ranks_28.csv',
        
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-07-08-30__ranks_28.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-07-23-19__ranks_56.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-07-46-01__ranks_112.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_TSP_TSP/2021-06-23-08-48-09__ranks_224.csv',
    ],
    'mpi_exp3_strong_sa': [
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-06-08-26__ranks_1.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-06-16-08__ranks_2.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-06-22-28__ranks_4.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-06-28-30__ranks_8.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-06-34-40__ranks_14.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-06-42-44__ranks_28.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-07-08-40__ranks_28.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-07-23-25__ranks_56.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-07-46-06__ranks_112.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp3_strong_scaling_TSP/2021-06-23-08-48-28__ranks_224.csv',
    ],
    'cuda_standard_sa': [
        '../code/results/proper_exps/v41_cuda_all_v4/cuda/sa/cuda_sa_experiment4_allTSP/2021-06-28-09-01-06__.csv'
    ],
    'cuda_standard_sa_more':[
        '../code/results/proper_exps/v41_cuda_all_v4/cuda/sa/cuda_sa_experiment4_allTSP/2021-06-28-09-01-06__.csv'
    ],

    'mpi_ga_exp1': [
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/ga/ga_exp1_standard_TSP_TSP/2021-06-23-06-14-14__ranks_1.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/ga/ga_exp1_standard_TSP_TSP/2021-06-23-06-20-39__ranks_2.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/ga/ga_exp1_standard_TSP_TSP/2021-06-23-06-26-35__ranks_4.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/ga/ga_exp1_standard_TSP_TSP/2021-06-23-06-32-24__ranks_8.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/ga/ga_exp1_standard_TSP_TSP/2021-06-23-06-38-45__ranks_14.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/ga/ga_exp1_standard_TSP_TSP/2021-06-23-06-49-58__ranks_28.csv',
    ],
    'cuda_ga': [
        '../code/results/proper_exps/v33_cuda_good_ga/cuda/ga/cuda_ga_experiment_1_standard_TSP/2021-06-23-18-29-07__.csv'
    ],
    'mpi_rosen': [
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_Rosen_Rosen/2021-06-23-06-09-12__ranks_1.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_Rosen_Rosen/2021-06-23-06-16-55__ranks_2.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_Rosen_Rosen/2021-06-23-06-23-20__ranks_4.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_Rosen_Rosen/2021-06-23-06-29-25__ranks_8.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_Rosen_Rosen/2021-06-23-06-35-40__ranks_14.csv',
        '../code/results/proper_exps/v31_mpi_good_exps_0623_test/mpi/sa/exp1_standard_Rosen_Rosen/2021-06-23-06-44-06__ranks_28.csv',
    ],
    'serial': [
        '../code/results/proper_exps/v32_serial_good_exps_0623_test/serial/sa/exp1_standard_TSP_TSP/2021-06-23-09-29-34__serial.csv'
    ],
    'serial_rosen':[
        '../code/results/proper_exps/v32_serial_good_exps_0623_test/serial/sa/exp1_standard_Rosen_Rosen/2021-06-23-09-30-25__serial.csv'
    ],
    'serial_ga':[
        '../code/results/proper_exps/v32_serial_good_exps_0623_test/serial/ga/ga_exp1_standard_TSP_TSP/2021-06-23-09-32-32__serial.csv'
    ],
    'cuda_rosen': [
        '../code/results/proper_exps/v31_cuda_good/cuda/sa/cuda_sa_experiment1_standard_Rosen/2021-06-23-17-22-06__.csv'
    ],
    'cuda_4_bs':[
        '../code/results/proper_exps/v41_cuda_all_v4/cuda/sa/cuda_sa_experiment4_allTSP/2021-06-28-09-01-06__.csv'
    ],
    'mpi_strong_exp4_corrected':[
        '../code/results/proper_exps/v38_mpi_good_exps_0627_test_corrected/mpi/sa/exp4_strong_scaling_corrected_TSP/2021-06-27-08-23-26__ranks_1.csv',
        '../code/results/proper_exps/v38_mpi_good_exps_0627_test_corrected/mpi/sa/exp4_strong_scaling_corrected_TSP/2021-06-27-08-34-24__ranks_2.csv',
        '../code/results/proper_exps/v38_mpi_good_exps_0627_test_corrected/mpi/sa/exp4_strong_scaling_corrected_TSP/2021-06-27-08-42-47__ranks_4.csv',
        '../code/results/proper_exps/v38_mpi_good_exps_0627_test_corrected/mpi/sa/exp4_strong_scaling_corrected_TSP/2021-06-27-08-45-49__ranks_8.csv',
        '../code/results/proper_exps/v38_mpi_good_exps_0627_test_corrected/mpi/sa/exp4_strong_scaling_corrected_TSP/2021-06-27-08-47-43__ranks_14.csv',
        '../code/results/proper_exps/v38_mpi_good_exps_0627_test_corrected/mpi/sa/exp4_strong_scaling_corrected_TSP/2021-06-27-08-49-24__ranks_28.csv'
    ]
}

def get_csv(name: str) -> pd.DataFrame:
    """Reads a csv file and returns it as a df

    Args:
        name (str): filename

    Returns:
        pd.DataFrame:
    """
    df = pd.read_csv(name)
    dic = [k.strip() for k in df.keys()]
    df.columns = dic
    df.keys()
    keys = df.select_dtypes(include='object').keys()
    for key in keys:
        df[key] = df[key].map(lambda s: s.strip())
    return df

def do_analysis_v2_final_score_vs_time(do_multiple_nodes=False, prob_size=1000):
    """Plots final score vs time for multiple methods, SA.

    Args:
        do_multiple_nodes (bool, optional). Defaults to False.
        prob_size (int, optional): 100 or 1000 for TSP. Defaults to 1000.
    """
    def single(name, label):
        df = get_csv(name)
        if label == 'CUDA 4 block size':
            df = df[df['CUDA_BlockSize'] == 4]
            df = df[df['CUDA_GridSize'] == 256]
        if label == 'CUDA':
            if prob_size == 1000:
                df = df[df['CUDA_BlockSize'] == 32]
                df = df[df['CUDA_GridSize'] == 32]
            else:
                if prob_size == 100:
                    pass
                    df = df[df['CUDA_BlockSize'] == 16]
                    df = df[df['CUDA_GridSize'] == 64]
                else:
                    df = df[df['CUDA_BlockSize'] == 16]
                    df = df[df['CUDA_GridSize'] == 64]

        df = df[df['ProbSize'] == prob_size]
        df = df.groupby("Num Iterations", as_index=False).mean()
        # if prob_size == 100:print(df)
        ins = list(df['Time Ops'] + df['Time Init'])
        outs = list(df['Final Score'])
        if prob_size == 100 and (label == 'CUDA' or label == 'CUDA 4 block size'):
            outs = [1e4] + outs
            ins = [0] + ins
         
        # print(label, list(ins), list(outs))
        plt.plot(ins, outs, label=label)
    single(FILES['serial'][0], 'Serial')
    single(FILES['mpi_standard_sa'][0], '1 Rank')
    single(FILES['mpi_standard_sa'][4], '14 Ranks')
    if do_multiple_nodes:
        single(FILES['mpi_standard_sa'][-1], '224 Ranks')
    if not do_multiple_nodes:
        if prob_size == 1000:
            single(FILES['cuda_standard_sa'][0], 'CUDA')
        if prob_size == 100:
            single(FILES['cuda_standard_sa'][0], 'CUDA')
        single(FILES['cuda_4_bs'][0], 'CUDA 4 block size')
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Final Distance (Lower is better)")
    if prob_size==1000:
        plt.xlim(right=2100)
    plt.title(
        f"TSP Final Distance vs Total time (ms).\nProblem Size = {prob_size}. Simulated Annealing")
    plt.legend()
    plt.yscale('log')
    s = '_mult_nodes' if do_multiple_nodes else ''
    plt.savefig(
        f'./plots/proper_v2_final_score_vs_iters_mpi{s}_{prob_size}.png')
    pass


def do_analysis_v3_flops_vs_problem_size(do_multiple_nodes=False):
    """Plots 'operations' per second for TSP SA

    Args:
        do_multiple_nodes (bool, optional). Defaults to False.
    """
    def single(name, label):
        df = get_csv(name)
        if label == 'CUDA':
            # df = df[df['CUDA_BlockSize'] == 16]
            # df = df[df['CUDA_GridSize'] == 1024]
            df = df[df['CUDA_BlockSize'] == 32]
            df = df[df['CUDA_GridSize'] == 256]
        elif label == 'CUDA 4 block size':
            df = df[df['CUDA_BlockSize'] == 4]
            df = df[df['CUDA_GridSize'] == 1024]
        df = df[df['Num Iterations'] == df['Num Iterations'].max()]
        df = df.groupby("ProbSize", as_index=False).mean()
        ins = df['ProbSize']
        # Flops
        outs = (df['Procs'] * df['Num Iterations'] * df['ProbSize']
                ) / (df['Time Ops'] + df['Time Init'])
        plt.plot(ins, outs, label=label)

    single(FILES['serial'][0], 'Serial')

    single(FILES['mpi_standard_sa'][4], '14 Ranks')
    single(FILES['mpi_standard_sa'][5], '28 Ranks on one node')
    if do_multiple_nodes:
        single(FILES['mpi_standard_sa'][-1], '224 Ranks')
    single(FILES['cuda_standard_sa_more'][0], 'CUDA')

    plt.xlabel("Problem Size")
    plt.ylabel("Operations per Second")
    plt.title("TSP Operations per Second vs Problem Size (SA)")
    plt.legend()
    plt.yscale('log')
    s = '_mult_nodes' if do_multiple_nodes else ''
    plt.savefig(f'./plots/proper_v3_flops_vs_problem_size{s}.png')


def do_analysis_v4_mpi_weak_scaling(do_multiple_nodes=False):
    # Show time taken vs ranks, where problem size is proportional to ranks.
    ins = []
    outs = []
    init = None
    do_flops = False
    N = 6 if not do_multiple_nodes else len(FILES['mpi_standard_sa'])+1
    fig, ax = plt.subplots()
    scores = []

    for i, n in enumerate(FILES['mpi_standard_sa'][:N]):
        # if not do_multiple_nodes:print(n)
        if do_multiple_nodes and i == 5:
            continue
        df = get_csv(n)
        df['Time'] = df['Time Init'] + df['Time Ops']
        df['Ops'] = df['Procs'] * df['Num Iterations'] * df['ProbSize']
        df['Flops'] = df['Ops'] / df['Time']
        df = df[df['Num Iterations'] == df['Num Iterations'].max()]
        df = df[df['ProbSize'] == df['ProbSize'].max()]
        ranks = list(df['Procs'])[0]
        scores.append(df.mean()['Final Score'])
        if do_flops:
            time = df.mean()['Flops']
        else:
            time = df.mean()['Time']  # / ranks
        if init is None:
            init = time
        ins.append(ranks)
        outs.append(time)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    ax1.plot(ins, outs, label='Actual')
    ax1.plot(ins, outs[:1] * len(outs), label='Ideal', linestyle='--')
    ax2.plot(ins, scores, label='scores', color='green', linestyle='dotted')
    word = 'Time' if not do_flops else 'FLOPS'
    ax1.set_title(f"MPI Ranks vs {word} for a per-rank constant problem size (SA)")
    ax1.set_xlabel("MPI Ranks")
    ax1.set_ylabel(f"{word} for fixed number of iterations per rank")
    ax1.legend()
    ax2.legend()
    ax2.set_ylabel("Final Scores")
    ax1.set_ylim(bottom=0)
    s = '_mult_nodes' if do_multiple_nodes else ''
    plt.savefig(
        f"./plots/proper_v4_weak_scaling_mpi_ranks_vs_{word}_v1_{s}.png")
    plt.close()
    pass


def do_analysis_v5_mpi_strong_scaling(do_multiple_nodes: bool):
    """Shows strong scaling.

    Args:
        do_multiple_nodes (bool):
    """
    ins = []
    outs = []
    ideal = []
    start = None
    N = 6 if not do_multiple_nodes else len(FILES['mpi_exp3_strong_sa'])
    fig, ax = plt.subplots()
    scores = []
    for i, n in enumerate(FILES['mpi_exp3_strong_sa'][:N]):
        if do_multiple_nodes and i == 5:
            continue
        df = get_csv(n)
        df = df[df['ProbSize'] == df['ProbSize'].max()]
        df = df[df['Num Iterations'] == df['Num Iterations'].max()]
        df = df.mean()
        ins.append((df['Procs']))
        outs.append((df['Time Ops'] + df['Time Init']))
        scores.append(df['Final Score'])
        if start is None:
            start = outs[-1]
        ideal.append(start / ins[-1])
    
    outs2 = []
    ins2 = []
    start2 = None
    if not do_multiple_nodes:
        for i, n in enumerate(FILES['mpi_strong_exp4_corrected']):
            if do_multiple_nodes and i == 5:
                continue
            elif not do_multiple_nodes and i == 6: 
                continue
            df = get_csv(n)
            df = df[df['ProbSize'] == df['ProbSize'].max()]
            # print(df.groupby("Num Iterations", as_index=False).mean())
            for time, score in zip(df['Time Ops'] + df['Time Init'], df['Final Score']):
                if score <= 443:
                    outs2.append(time)
                    ins2.append(list(df['Procs'])[0])
                    break
            if start2 is None:
                start2 = outs2[-1]

    ax.plot(ins, outs, label='Actual')
    if not do_multiple_nodes:
        ax.plot(ins2, outs2, label='Corrected')
    ax.plot(ins, ideal, label='Ideal', linestyle='--')
    ax.legend()
    ax.set_xlabel("Number of Ranks")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Number of Ranks vs Time for a total constant problem size (SA)")
    s = '_mult_nodes' if do_multiple_nodes else ''
    if do_multiple_nodes:
        ax.tick_params(color='r', which='minor', length=6, width=2)
        ax.set_xticks([i * 14 for i in [1, 2, 4, 8, 16]], minor=True)
    plt.savefig(
        f"./plots/proper_v5_strong_scaling_mpi_ranks_vs_time_v1{s}.png")

def ga_do_analysis_v6_genetic_algorithms():
    """Does GA analysis, final score and performance.
    """
    def single(name, label):
        df = get_csv(name)
        df = df[df['ProbSize'] == 1000]
        # print(df)
        if label == 'CUDA':
            df = df[df['Procs'] == 1024]
        df = df.groupby("Num Iterations", as_index=False).mean()
        # ins = df['Num Iterations']
        ins = df['Time Ops'] + df['Time Init']
        outs = df['Final Score']
        plt.plot(ins, outs, label=label)
    
    single(FILES['serial_ga'][0], 'Serial')
    single(FILES['mpi_ga_exp1'][-2], '14 Ranks')
    
    single(FILES['mpi_standard_sa'][4], '14 Ranks SA')
    single(FILES['cuda_ga'][0], 'CUDA')

    plt.xlabel("Time (ms)")
    plt.ylabel("Final Distance (lower is better)")
    plt.title("TSP Final Distance vs Total time (ms).\nProblem Size = 1000. Genetic Algorithm")
    plt.legend()
    do_multiple_nodes = False
    s = '_mult_nodes' if do_multiple_nodes else ''

    plt.savefig(f'./plots/proper_v6_ga{s}.png')
    plt.close()
    pass
    #   weak scaling GA
    ins = []
    outs = []
    ideal = []
    fix, ax = plt.subplots()
    start = None
    scores = []
    for i, n in enumerate(FILES['mpi_ga_exp1'][:-1]):
        df = get_csv(n)
        df = df[df['Num Iterations'] == df['Num Iterations'].max()]
        df = df.mean()
        ins.append((df['Procs']))
        outs.append((df['Time Ops'] + df['Time Init']))
        scores.append(df['Final Score'])
        if start is None:
            start = outs[-1]
        ideal.append(start)
    ax.plot(ins, outs, label='Actual')
    ax.plot(ins, ideal, label='Ideal', linestyle='--')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Number of Ranks")
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        "Number of Ranks vs Time for a constant amount of work per rank (GA)")
    s = '_mult_nodes' if do_multiple_nodes else ''
    if do_multiple_nodes:
        ax.tick_params(color='r', which='minor', length=6, width=2)
        ax.set_xticks([i * 14 for i in [1, 2, 4, 8, 16]], minor=True)
    plt.savefig(f"./plots/proper_v6_ga_scaling{s}.png")


def v8_rosen():
    """Does rosen plots
    """
    li = FILES['mpi_rosen']
    li = [li[1], li[-2], li[-1]]
    ins = []
    outs = []
    
    
    df = get_csv(FILES['serial_rosen'][0])
    df = df[df['Num Iterations'] == df['Num Iterations'].max()]
    df = df.groupby(['ProbSize'], as_index=False).mean()
    plt.plot(df['ProbSize'], df['Final Score'], label=f"Serial")
    for name in li:
        df = get_csv(name)
        df = df[df['Num Iterations'] == df['Num Iterations'].max()]
        df = df.groupby(['ProbSize'], as_index=False).mean()
        plt.plot(df['ProbSize'], df['Final Score'], label=f"{list(df['Procs'])[0]} Ranks")
    
    df = get_csv(FILES['cuda_rosen'][0])
    df = df[df['Num Iterations'] == df['Num Iterations'].max()]
    df = df[df['Procs'] == 65536]
    df = df.groupby(['ProbSize'], as_index=False).mean()
    plt.plot(df['ProbSize'], df['Final Score'], label=f"CUDA")



    plt.legend()
    plt.title("Rosenbrock Score (Simulated Annealing)")
    plt.xlabel("Problem Size")
    plt.ylabel("Final Value")
    plt.savefig("./plots/v8_rosen.png");
    plt.close()

def main():
    do_analysis_v2_final_score_vs_time(prob_size=100)
    plt.close()
    do_analysis_v2_final_score_vs_time()
    plt.close()
    v8_rosen()
    plt.close()
    ga_do_analysis_v6_genetic_algorithms();
    plt.close()
    do_analysis_v3_flops_vs_problem_size(True)
    plt.close()
    funcs = [
        do_analysis_v4_mpi_weak_scaling, do_analysis_v5_mpi_strong_scaling
    ]
    for mult in [False, True]:
        for func in funcs:
            print((func).__name__)
            func(mult)
            plt.close()

if __name__ == "__main__":
    main()
