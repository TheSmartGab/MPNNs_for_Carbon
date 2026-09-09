import os
import glob
import re
import torch
import yaml

def extract_radial_nn_from_model(model_path):
    """
    Safely inspects the architecture of conv_tp_weights (FullyConnectedNet)
    inside the model's interaction blocks by disabling weights_only tracking.
    """
    if not os.path.exists(model_path):
        return {}

    try:
        # Bypasses the strict PyTorch unpickling restriction for trusted local files
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model = checkpoint['model'] if (isinstance(checkpoint, dict) and 'model' in checkpoint) else checkpoint
        
        if hasattr(model, 'interactions') and len(model.interactions) > 0:
            first_interaction = model.interactions[0]
            if hasattr(first_interaction, 'conv_tp_weights'):
                net = first_interaction.conv_tp_weights
                
                if hasattr(net, 'layer_sizes'):
                    sizes = list(net.layer_sizes)
                    return {"radial_nn_architecture": f"FullyConnectedNet{sizes}"}
                
                layers = []
                for module in net.modules():
                    if isinstance(module, torch.nn.Linear):
                        if not layers:
                            layers.append(module.in_features)
                        layers.append(module.out_features)
                
                if layers:
                    return {"radial_nn_architecture": f"FullyConnectedNet{layers}"}
                
                net_str = str(net)
                return {"radial_nn_architecture": net_str.strip().replace("\n", " ")}
                
        return {"radial_nn_architecture": "Unknown / Parse Error"}
    except Exception as e:
        return {"radial_nn_architecture": f"Error reading file: {str(e)}"}

def parse_associated_log(log_path):
    """Parses scalar training logs and maps values safely into typing dict structures."""
    if not os.path.exists(log_path):
        return {}
        
    data = {"model_details": {}, "loss_function": {}, "optimizer_information": {}}
    
    patterns = {
        "channels_max_l": re.compile(r"Message passing with (\d+) channels and max_L=(\d+)", re.IGNORECASE),
        "layers_correlation": re.compile(r"(\d+) layers, each with correlation order: (\d+)", re.IGNORECASE),
        "spherical_harmonics": re.compile(r"spherical harmonics up to: l=(\d+)", re.IGNORECASE),
        "radial_basis": re.compile(r"(\d+) radial and (\d+) basis functions", re.IGNORECASE),
        "cutoff": re.compile(r"Radial cutoff:\s*([\d.]+)\s*A", re.IGNORECASE),
        "receptive_field": re.compile(r"total receptive field for each atom:\s*([\d.]+)\s*A", re.IGNORECASE),
        "hidden_irreps": re.compile(r"Hidden irreps:\s*(.+)", re.IGNORECASE),
        "total_parameters": re.compile(r"Total number of parameters:\s*(\d+)", re.IGNORECASE),
        "optimizer_type": re.compile(r"Using (\s*\w+\s*) as parameter optimizer", re.IGNORECASE),
        "batch_size": re.compile(r"Batch size:\s*(\d+)", re.IGNORECASE),
        "ema_decay": re.compile(r"Using Exponential Moving Average with decay:\s*([\d.]+)", re.IGNORECASE),
        "gradient_updates": re.compile(r"Number of gradient updates:\s*(\d+)", re.IGNORECASE),
        "lr_wd": re.compile(r"Learning rate:\s*([\d.]+),\s*weight decay:\s*([\d.e-]+)", re.IGNORECASE),
        "gradient_clipping": re.compile(r"Using gradient clipping with tolerance=([\d.]+)", re.IGNORECASE),
        "loss_weights": re.compile(r"(\w+Loss)\((.+)\)", re.IGNORECASE)
    }
    
    try:
        with open(log_path, 'r') as file:
            log_content = file.read()
            
        # Model
        m = patterns["channels_max_l"].search(log_content)
        if m: data["model_details"]["channels"], data["model_details"]["max_L"] = int(m.group(1)), int(m.group(2))
        m = patterns["layers_correlation"].search(log_content)
        if m: data["model_details"]["layers"], data["model_details"]["correlation_order"] = int(m.group(1)), int(m.group(2))
        m = patterns["spherical_harmonics"].search(log_content)
        if m: data["model_details"]["spherical_harmonics_l"] = int(m.group(1))
        m = patterns["radial_basis"].search(log_content)
        if m: data["model_details"]["radial_functions"], data["model_details"]["basis_functions"] = int(m.group(1)), int(m.group(2))
        m = patterns["cutoff"].search(log_content)
        if m: data["model_details"]["radial_cutoff_A"] = float(m.group(1))
        m = patterns["receptive_field"].search(log_content)
        if m: data["model_details"]["total_receptive_field_A"] = float(m.group(1))
        m = patterns["hidden_irreps"].search(log_content)
        if m: data["model_details"]["hidden_irreps"] = m.group(1).strip()
        m = patterns["total_parameters"].search(log_content)
        if m: data["model_details"]["total_parameters"] = int(m.group(1))
        
        # Optimizer
        m = patterns["optimizer_type"].search(log_content)
        if m: data["optimizer_information"]["optimizer"] = m.group(1).strip()
        m = patterns["batch_size"].search(log_content)
        if m: data["optimizer_information"]["batch_size"] = int(m.group(1))
        m = patterns["ema_decay"].search(log_content)
        if m: data["optimizer_information"]["ema_decay"] = float(m.group(1))
        m = patterns["gradient_updates"].search(log_content)
        if m: data["optimizer_information"]["gradient_updates"] = int(m.group(1))
        m = patterns["lr_wd"].search(log_content)
        if m: data["optimizer_information"]["learning_rate"], data["optimizer_information"]["weight_decay"] = float(m.group(1)), float(m.group(2))
        m = patterns["gradient_clipping"].search(log_content)
        if m: data["optimizer_information"]["gradient_clipping_tolerance"] = float(m.group(1))
        
        # Loss
        m = patterns["loss_weights"].search(log_content)
        if m:
            data["loss_function"]["type"] = m.group(1)
            weights = {}
            for pair in m.group(2).split(","):
                if "=" in pair:
                    k, v = pair.split("=")
                    weights[k.strip()] = float(v.strip())
            data["loss_function"]["weights"] = weights
    except Exception:
        pass
        
    return data

def main():
    master_records = {}
    
    # Locate all sub-directory models matching the target structure safely
    model_paths = glob.glob("*/MACE.model")
    
    if not model_paths:
        print("Error: Could not locate any '*/MACE.model' path configurations in this directory.")
        return

    for path in model_paths:
        run_folder = os.path.dirname(path)
        print(f"Interrogating run folder and binaries: '{run_folder}'...")
        
        # Find adjacent text log
        log_dir = os.path.join(run_folder, "logs")
        logs = glob.glob(os.path.join(log_dir, "*.log"))
        
        combined_data = parse_associated_log(logs[0]) if logs else {"model_details": {}, "loss_function": {}, "optimizer_information": {}}
        
        # Pull structural details from binary matrix
        radial_network = extract_radial_nn_from_model(path)
        combined_data["model_details"].update(radial_network)
        
        master_records[run_folder] = combined_data

    # Save out directly into your shared runtime workspace root
    output_filename = "all_models_combined_hyperparameters.yaml"
    with open(output_filename, 'w') as f:
        yaml.dump(master_records, f, default_flow_style=False, sort_keys=False)
        
    print(f"\nSuccess! Properties successfully saved to: {output_filename}")

if __name__ == "__main__":
    main()
