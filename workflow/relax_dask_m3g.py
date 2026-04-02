from pymatgen.core.structure import Structure

# import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask
import dask.distributed

import time
import shutil
import yaml
import subprocess
import os.path, os
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure, Element, PeriodicSite
from tqdm import tqdm
import matgl
from matgl.ext.ase import PESCalculator, MolecularDynamics, Relaxer
import os
import torch
import csv
import numpy as np

def submit_relax(cur_range):
    """
    Perform M3GNet-based structure relaxation and property prediction on a range of unrelaxed structures.

    This function iterates over a given range of structure indices, loads corresponding unrelaxed structures 
    from a predefined directory, relaxes them using the M3GNet potential, and predicts formation energy 
    and band gap using pretrained MEGNet models. The results are appended to a CSV file and the relaxed 
    structures are saved as POSCAR files.

    Only structures that do not contain the elements Polonium (Po), Astatine (At), Radium (Ra), or 
    Francium (Fr) are processed.

    Parameters:
        cur_range (iterable): A list or range of integer indices corresponding to structure files 
                              named POSCAR_{i} in the source directory.

    Returns:
        int: The first index in the input range, useful for tracking batch execution.
    """    
    par_el = 'run_full'
    root = f'/blue/hennig/jasongibson/diff_model/materials/{par_el}/unrelaxed/'    
    root_relaxed = f'/blue/hennig/jasongibson/diff_model/materials/{par_el}/m3gnet_relaxed/'

    pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    model_form = matgl.load_model("MEGNet-MP-2018.6.1-Eform")
    model_bg = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")
    
    relaxer = Relaxer(potential=pot)    

    for i in cur_range:
        print(i,flush=True)
        struct = Structure.from_file(root + f'POSCAR_{i}')
        meta = [i]
        if (Element('Po') not in struct.species and Element('At') not in struct.species and Element('Ra') not in struct.species and Element('Fr') not in struct.species):
            try:
                relax_results = relaxer.relax(struct, fmax=0.001)
                final_structure = relax_results["final_structure"]
                final_energy = relax_results["trajectory"].energies[-1]
            except:
                final_structure = struct
                final_energy = np.nan
            meta.append(final_energy)
        
            # For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
            for j, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
                graph_attrs = torch.tensor([j])
                try:
                    bandgap = model_bg.predict_structure(structure=final_structure, state_attr=graph_attrs)
                except:
                    bandgap = np.nan
                meta.append(float(bandgap))
            try:
                eform = model_form.predict_structure(final_structure)
            except:
                eform = np.nan
            meta.append(float(eform))
            with open(root_relaxed + 'data.csv', 'a') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(meta)    
            final_structure.to(root_relaxed + f'POSCAR_{i}')
    return cur_range[0]

def update_futures(all_futures,Job_status_fp):
    # remove all futures with an exception
    calc_done_index, calc_wo_exceptn_index = [], []
    for i, future in enumerate(all_futures):
        if future.done():
            calc_done_index.append(i)
            if not future.exception():
                calc_wo_exceptn_index.append(i)
            else:
                Job_status_fp.write("Exception: "+str(future.exception())+"\n")

    # relaxed futures that are to be processed
    done_futures = [all_futures[i] for i in calc_wo_exceptn_index]
    # futures that are still running
    active_futures = [all_futures[i] for i in range(len(all_futures)) \
                                                if i not in calc_done_index]

    return active_futures, done_futures

def main(par_el):
    num_calcs_at_once = 150
    job_specs = {"cores":1,
		"memory":'16GB',
		"account": "benjamin.geisler",
		"queue": "hpg-default",
		"walltime":"3-00:00:00",
                "job_extra_directives":["--ntasks=8","--nodes=1","--qos=benjamin.geisler-b"]}
                
                #"job_extra_directives":["--ntasks=1","--nodes=1","--qos=hennig","--partition=gpu","--gpus=a100:1"]}


    root = f'/blue/hennig/jasongibson/diff_model/materials/{par_el}/unrelaxed/'
    root_relaxed = f'/blue/hennig/jasongibson/diff_model/materials/{par_el}/m3gnet_relaxed/'
    files = os.listdir(root)
    with open(root_relaxed + 'data.csv', 'a') as myfile:
       wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
       wr.writerow(['id','final_E','PBE','GLLB-SC','HSE','SCAN','E_form'])

    tot_calcs = 750
    ranges = np.array_split(range(len(files)), tot_calcs)
    futures = []

    # start cluster and scale jobs
    cluster_job = SLURMCluster(cores=job_specs['cores'],
                           memory=job_specs['memory'],
                           account=job_specs['account'],
                           queue=job_specs['queue'],
                           walltime=job_specs['walltime'],
                           job_extra_directives=job_specs['job_extra_directives'])
    print(cluster_job.job_script())

    cluster_job.scale(num_calcs_at_once) # number of parallel jobs
    client  = Client(cluster_job)

    num_submitted_calcs = 0
    num_finished_calcs = 0

    job_status_fp = open(f"job_status_relax.log","w")

    for i in range(0,num_calcs_at_once):
        cur_range = ranges[num_submitted_calcs]

        job_status_fp.write("Submitting "+str(cur_range[-1])+"\n")
        out = client.submit(submit_relax,cur_range)
        futures.append(out)
        num_submitted_calcs += 1

    job_status_fp.close()

    while True:
        job_status_fp = open("job_status_relax.log","a")
        futures, done_Futures = update_futures(futures,job_status_fp)
        for i in done_Futures:
            job_status_fp.write("Finished "+str(i.result())+"\n")
        num_finished_calcs += len(done_Futures)
        job_status_fp.write("Num finished calcs:"+str(num_finished_calcs)+"\n")
        if num_finished_calcs >= tot_calcs:
            job_status_fp.write("Finished all calculations\n")
            break

        else:
            if len(futures) < num_calcs_at_once:
                for i in range(0,num_calcs_at_once-len(futures)):
                    if num_submitted_calcs < tot_calcs:
                        cur_range = ranges[num_submitted_calcs]
                        job_status_fp.write("Submitting "+str(cur_range[-1])+"\n")
                        out = client.submit(submit_relax,cur_range)
                        futures.append(out)
                        num_submitted_calcs += 1

                    else:
                        break
                job_status_fp.write(str(len(futures))+" calculations running\n")
            else:
                job_status_fp.write(str(len(futures))+" calculations running\n")

        job_status_fp.close()
        time.sleep(30)

if __name__ == "__main__":
    main('run_full')
