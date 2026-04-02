import ase
import ase.io
import ase.io.espresso
from ase.data import atomic_masses, atomic_numbers
import json
import subprocess

import matplotlib.pyplot as plt

def get_k_point_density():
    k_point_density = 40 #min kpoints per inv A
    return k_point_density

def get_q_point_density():
    q_point_density = 15 #min qpoints per inv A
    return q_point_density

def get_k_q_grid(k_point_density,q_point_density,reciprocal_cell):
    k_grid = []
    q_grid = []
    for lat_i in range(0,3):
        k = round(reciprocal_cell.lengths()[lat_i]*k_point_density)
        q = round(reciprocal_cell.lengths()[lat_i]*q_point_density)
    
        if k%q != 0:
            while True:
                if k%q == 0:
                    break
                if (k+1)%q == 0:
                    k = k + 1
                    break
                elif (k-1)%q == 0:
                    k = k - 1
                    break
                else:
                    q -=1
        k_grid.append(k)
        q_grid.append(q)

    return k_grid, q_grid


def get_ibrav_celldm(cell):
    lat_str = ""
    for i in cell:
        for j in i:
            lat_str += str(round(j,6))+" "
        lat_str += "\n"
        
    lat_str += "1.88973"
    path = '/blue/hennig/jasongibson/qe_gpu/qe-7.3.1/bin/cell2ibrav.x'     
    proc = subprocess.Popen(path,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    
    output, error = proc.communicate(bytes(lat_str,'utf-8'))
    ibrav = 0
    celldm = {}
    if error==None:
        output = output.decode("utf-8")
        output = output.split("\n")
        #print(output)
        for i in output:
            if "ibrav" in i:
                line = i.split()
                ibrav = int(line[2])
            elif "celldm" in i:
                line = i.split()
                celldm[line[0]] = float(line[2])
        
    return ibrav, celldm


def update_dict_relax(param,k_grid,atom_obj):
    param["CONTROL"]["tstress"] = True
    param["CONTROL"]["tprnfor"] = True

    param["CONTROL"]["prefix"] = atom_obj.get_chemical_formula()
    param["SYSTEM"]["ntyp"] = len(set(atom_obj.get_atomic_numbers()))
    param["SYSTEM"]["nat"] = atom_obj.get_global_number_of_atoms()

    count = 0
    w_str_sp = ""
    w_str_pos = ""
    w_str_cell = ""
    cell_list = atom_obj.cell.tolist()

    for i in atom_obj.get_chemical_symbols():
        param["ATOMIC_SPECIES"][i] = str(atomic_masses[atomic_numbers[i]]) + " " + str(i)+"_ONCV_PBE-1.2.upf"
        w_str_pos += i+" "+" ".join([str(round(j,6)) for j in atom_obj.get_scaled_positions()[count]])
        w_str_pos += "\n"
    
        w_str_cell = " ".join([str(j) for j in cell_list[0]])
        w_str_cell += "\n"
        w_str_cell += " ".join([str(j) for j in cell_list[1]])
        w_str_cell += "\n"
        w_str_cell += " ".join([str(j) for j in cell_list[2]])
        w_str_cell += "\n"
    
        count += 1
    
    param["ATOMIC_POSITIONS (crystal)"] = w_str_pos
    param["CELL_PARAMETERS (angstrom)"] = w_str_cell
    param["K_POINTS (automatic)"] = " ".join([str(i) for i in k_grid])+" 0 0 0"

    cell = atom_obj.get_cell()
    cell.tolist()
    ibrav, celldm = get_ibrav_celldm(cell)

    if ibrav != 0:
        #print("ibrav =",ibrav,celldm)
        param["SYSTEM"]["ibrav"] = ibrav
        for key in celldm.keys():
            param["SYSTEM"][key] = celldm[key]
        param.pop("CELL_PARAMETERS (angstrom)")

    return param

def update_dict_ph(ph, q_grid, atom_obj):
    ph["INPUTPH"]["reduce_io"] = True
    ph["INPUTPH"]["ldisp"] = True

    #count = 1
    #for i in atom_obj.get_chemical_symbols():
    #    ph_elph["INPUTPH"]["amass("+str(count)+")"] = atomic_masses[atomic_numbers[i]]
    #    ph["INPUTPH"]["amass("+str(count)+")"] = atomic_masses[atomic_numbers[i]]
    #    count += 1

    ph["INPUTPH"]["prefix"] = atom_obj.get_chemical_formula()
    ph["INPUTPH"]["fildyn"] = ph["INPUTPH"]["prefix"]+".dyn"
    ph["INPUTPH"]["nq1"] = q_grid[0]
    ph["INPUTPH"]["nq2"] = q_grid[1]
    ph["INPUTPH"]["nq3"] = q_grid[2]

    return ph

def update_dict_ph_elph(ph_elph,q_grid,atom_obj):
    ph_elph["INPUTPH"]["reduce_io"] = True
    ph_elph["INPUTPH"]["ldisp"] = True
    ph_elph["INPUTPH"]["trans"] = True

    #count = 1
    #for i in atom_obj.get_chemical_symbols():
    #    ph_elph["INPUTPH"]["amass("+str(count)+")"] = atomic_masses[atomic_numbers[i]]
    #    matdyn["input"]["amass("+str(count)+")"] = atomic_masses[atomic_numbers[i]]
    #    count += 1
    ph_elph["INPUTPH"]["prefix"] = atom_obj.get_chemical_formula()
    ph_elph["INPUTPH"]["fildyn"] = ph_elph["INPUTPH"]["prefix"]+".dyn"
    ph_elph["INPUTPH"]["nq1"] = q_grid[0]
    ph_elph["INPUTPH"]["nq2"] = q_grid[1]
    ph_elph["INPUTPH"]["nq3"] = q_grid[2]

    ph_elph["INPUTPH"]["electron_phonon"] = "interpolated"
    ph_elph["INPUTPH"]["el_ph_sigma"] = 0.001
    ph_elph["INPUTPH"]["el_ph_nsigma"] = 30

    return ph_elph

def update_q2r_matdyn_elph(q2r, matdyn,q_grid,atom_obj):

    q2r["input"]["fildyn"] = atom_obj.get_chemical_formula()+".dyn"
    q2r["input"]["flfrc"] = atom_obj.get_chemical_formula() + "".join([str(i) for i in q_grid])+".fc"
    q2r["input"]["la2F"] = True
    
    matdyn["input"]["la2F"] = True
    matdyn["input"]["dos"] = True
    matdyn["input"]["flfrc"] = q2r["input"]["flfrc"]
    matdyn["input"]["flfrq"] = atom_obj.get_chemical_formula()+ "".join([str(i) for i in q_grid])+".freq"
    matdyn["input"]["nk1"] = q_grid[0]
    matdyn["input"]["nk2"] = q_grid[1]
    matdyn["input"]["nk3"] = q_grid[2]
    matdyn["input"]["el_ph_nsigma"] = 30

    return q2r, matdyn

def write_qe_file(param,filename):
    with open(filename,"w") as fp:
        for i in param.keys():
            if type(param[i]) == dict:
                if i == "ATOMIC_SPECIES":
                    fp.write(i+"\n")
                    for j in param[i]:
                        fp.write(j+" "+param[i][j])
                        fp.write("\n")
                    fp.write("\n")
                    
                else:
                    fp.write("&"+i+"\n")
                    for j in param[i].keys():
                        if type(param[i][j]) == str:
                            fp.write("\t"+j+" = '"+str(param[i][j])+"',\n")
                        elif type(param[i][j]) == bool:
                            fp.write("\t"+j+" = ."+str(param[i][j]).lower()+".,\n")
                        else:
                            fp.write("\t"+j+" = "+str(param[i][j])+",\n")
                    fp.write("/\n")
                
            elif type(param[i]) == str:
                fp.write(i+"\n")
                fp.write(param[i])
                fp.write("\n")

            else:
                raise ValueError("Something went wrong")

def plot_ph_disp(line_count,dir_loc,plot_file):
    plt.rcParams.update({'font.size': 25})
    plt.rcParams["font.family"] = "Times New Roman"

    fig, axs = plt.subplots(1,1,figsize=(16,10))#,gridspec_kw={'width_ratios': [3, 1]})
    fig.tight_layout(h_pad=0.5,w_pad=0.1,rect=[0.02,0,1,1.0])

    with open(dir_loc+'/plotband.out','r') as fp:
        lines = fp.readlines()

    no_q_pt = int(lines[0].split()[-2])
    no_bnds = int(int(lines[0].split()[4]))
    lab_xcor = []
    for i in range(1,line_count+1):
        line = lines[i].split()
        lab_xcor.append(float(line[-1]))

    with open(dir_loc+'/'+plot_file,'r') as fp:
        lines = fp.readlines()

    x = []
    y = []
    for j in range(0,no_bnds):
        x.append([])
        y.append([])
        str_line = (j*(no_q_pt+1))
        for i in range(str_line,str_line+no_q_pt):
            line = lines[i].split()
            x[j].append(float(line[0]))
            y[j].append(float(line[1]))

    for i in range(0,len(x)):
        axs.plot(x[i],y[i],'r-',linewidth=3.4)
    for i in range(0,len(lab_xcor)):
        axs.plot([lab_xcor[i],lab_xcor[i]],[0,max(y[-1])+20],'k-',linewidth=1.4)

    axs.set_ylim(min(y[0])-20,max(y[-1])+20)
    axs.set_xlim(0,lab_xcor[-1])
    axs.plot([0,max(x[-1])],[0,0],'k--',linewidth=1.4)
    axs.set_xticks([])
    axs.set_ylabel('Frequency (cm^-1)')
    plt.savefig(dir_loc+"/ph_disp.png")


