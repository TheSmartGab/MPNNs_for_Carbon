import torch

def compare_models(dict1, dict2, rtol=1e-5, atol=1e-8):
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    if keys1 != keys2:
        missing_in_1 = keys2 - keys1
        missing_in_2 = keys1 - keys2
        print("⚠️ Key mismatch!")
        if missing_in_1:
            print("Missing in first model:", missing_in_1)
        if missing_in_2:
            print("Missing in second model:", missing_in_2)
        return False
    
    all_same = True
    for key in dict1.keys():
        print("inspecting key", key)
        t1 = dict1[key]
        t2 = dict2[key]
        
        if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
            if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
                print(f"❌ Parameter differs: {key}")
                all_same = False
        else:
            if t1 != t2:
                print(f"❌ Non-tensor entry differs: {key}")
                all_same = False

    if all_same:
        print("✅ All parameters are identical!")
    return all_same
