import argparse
import os
import re
import yaml

def parse_mace_log(log_file_path):
    # Determine the directory of the input file to save the YAML in the same place
    input_dir = os.path.dirname(os.path.abspath(log_file_path))
    base_name = os.path.splitext(os.path.basename(log_file_path))[0]
    output_yaml_path = os.path.join(input_dir, f"{base_name}_hyperparameters.yaml")

    # Dictionaries to store extracted values
    hyperparameters = {
        "model_details": {},
        "loss_function": {},
        "optimizer_information": {}
    }
    
    # Compile Regex Patterns
    patterns = {
        # Model Details
        "channels_max_l": re.compile(r"Message passing with (\d+) channels and max_L=(\d+)", re.IGNORECASE),
        "layers_correlation": re.compile(r"(\d+) layers, each with correlation order: (\d+)", re.IGNORECASE),
        "spherical_harmonics": re.compile(r"spherical harmonics up to: l=(\d+)", re.IGNORECASE),
        "radial_basis": re.compile(r"(\d+) radial and (\d+) basis functions", re.IGNORECASE),
        "cutoff": re.compile(r"Radial cutoff:\s*([\d.]+)\s*A", re.IGNORECASE),
        "receptive_field": re.compile(r"total receptive field for each atom:\s*([\d.]+)\s*A", re.IGNORECASE),
        "hidden_irreps": re.compile(r"Hidden irreps:\s*(.+)", re.IGNORECASE),
        "total_parameters": re.compile(r"Total number of parameters:\s*(\d+)", re.IGNORECASE),
        
        # Optimizer Information
        "optimizer_type": re.compile(r"Using (\s*\w+\s*) as parameter optimizer", re.IGNORECASE),
        "batch_size": re.compile(r"Batch size:\s*(\d+)", re.IGNORECASE),
        "ema_decay": re.compile(r"Using Exponential Moving Average with decay:\s*([\d.]+)", re.IGNORECASE),
        "gradient_updates": re.compile(r"Number of gradient updates:\s*(\d+)", re.IGNORECASE),
        "lr_wd": re.compile(r"Learning rate:\s*([\d.]+),\s*weight decay:\s*([\d.e-]+)", re.IGNORECASE),
        "gradient_clipping": re.compile(r"Using gradient clipping with tolerance=([\d.]+)", re.IGNORECASE),
        
        # FIXED: Look specifically for lines ending with Loss(...)
        "loss_weights": re.compile(r"(\w+Loss)\((.+)\)", re.IGNORECASE)
    }

    try:
        with open(log_file_path, 'r') as file:
            log_content = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return

    # --- Parse Model Details ---
    m_chan = patterns["channels_max_l"].search(log_content)
    if m_chan:
        hyperparameters["model_details"]["channels"] = int(m_chan.group(1))
        hyperparameters["model_details"]["max_L"] = int(m_chan.group(2))
        
    m_lay = patterns["layers_correlation"].search(log_content)
    if m_lay:
        hyperparameters["model_details"]["layers"] = int(m_lay.group(1))
        hyperparameters["model_details"]["correlation_order"] = int(m_lay.group(2))
        
    m_sh = patterns["spherical_harmonics"].search(log_content)
    if m_sh:
        hyperparameters["model_details"]["spherical_harmonics_l"] = int(m_sh.group(1))
        
    m_rad = patterns["radial_basis"].search(log_content)
    if m_rad:
        hyperparameters["model_details"]["radial_functions"] = int(m_rad.group(1))
        hyperparameters["model_details"]["basis_functions"] = int(m_rad.group(2))
        
    m_cut = patterns["cutoff"].search(log_content)
    if m_cut:
        hyperparameters["model_details"]["radial_cutoff_A"] = float(m_cut.group(1))
        
    m_rf = patterns["receptive_field"].search(log_content)
    if m_rf:
        hyperparameters["model_details"]["total_receptive_field_A"] = float(m_rf.group(1))
        
    m_irr = patterns["hidden_irreps"].search(log_content)
    if m_irr:
        hyperparameters["model_details"]["hidden_irreps"] = m_irr.group(1).strip()
        
    m_param = patterns["total_parameters"].search(log_content)
    if m_param:
        hyperparameters["model_details"]["total_parameters"] = int(m_param.group(1))

    # --- Parse Optimizer Information ---
    m_opt = patterns["optimizer_type"].search(log_content)
    if m_opt:
        hyperparameters["optimizer_information"]["optimizer"] = m_opt.group(1).strip()
        
    m_bs = patterns["batch_size"].search(log_content)
    if m_bs:
        hyperparameters["optimizer_information"]["batch_size"] = int(m_bs.group(1))
        
    m_ema = patterns["ema_decay"].search(log_content)
    if m_ema:
        hyperparameters["optimizer_information"]["ema_decay"] = float(m_ema.group(1))
        
    m_grad = patterns["gradient_updates"].search(log_content)
    if m_grad:
        hyperparameters["optimizer_information"]["gradient_updates"] = int(m_grad.group(1))
        
    m_lr = patterns["lr_wd"].search(log_content)
    if m_lr:
        hyperparameters["optimizer_information"]["learning_rate"] = float(m_lr.group(1))
        hyperparameters["optimizer_information"]["weight_decay"] = float(m_lr.group(2))
        
    m_clip = patterns["gradient_clipping"].search(log_content)
    if m_clip:
        hyperparameters["optimizer_information"]["gradient_clipping_tolerance"] = float(m_clip.group(1))

    # --- Parse Loss Function ---
    m_loss = patterns["loss_weights"].search(log_content)
    if m_loss:
        loss_name = m_loss.group(1)
        hyperparameters["loss_function"]["type"] = loss_name
        
        weight_pairs = m_loss.group(2).split(",")
        weights_dict = {}
        for pair in weight_pairs:
            if "=" in pair:
                k, v = pair.split("=")
                weights_dict[k.strip()] = float(v.strip())
        hyperparameters["loss_function"]["weights"] = weights_dict

    # Write to YAML file
    with open(output_yaml_path, 'w') as yaml_file:
        yaml.dump(hyperparameters, yaml_file, default_flow_style=False, sort_keys=False)
    
    print(f"Success! Generated: {output_yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse MACE log files for model, loss, and optimizer configs.")
    parser.add_argument(
        "-i", "--inputfile", 
        required=True, 
        help="Path to the MACE log text file."
    )
    
    args = parser.parse_args()
    parse_mace_log(args.inputfile)
