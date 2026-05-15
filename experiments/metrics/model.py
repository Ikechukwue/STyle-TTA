from .content_metrics import (_get_lpips_model, batch_lpips_distance,
                             _get_hed_model, _get_ldc_model, batch_edge_similarity,
                             _get_depth_model,
                             batch_depth_analysis, batch_edge_analysis)
import torch 
import numpy as np
from tqdm import tqdm
def model_run(set_A, set_B, common_classes, model_name, groups_A, groups_B, device):
    if model_name=="lpips":
        model = _get_lpips_model().to(device).eval()
    
    elif model_name=="hed":
        model = _get_hed_model().to(device).eval()
    
    elif model_name=="ldc":
        model = _get_ldc_model().to(device)

    elif model_name in ["depthpro", "depthanything_v2_large", "dept_large"]:
        model = _get_depth_model(model_name)

    model_results_per_class = {}

    for cls in common_classes:
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]
        
        if not images_A or not images_B:
            continue
        
        batch_A = torch.stack(images_A).to(device)
        batch_B = torch.stack(images_B).to(device)
        # Call a targeted batched function for just this model
        if model_name == "lpips":
            result_grid = batch_lpips_distance(batch_A, batch_B, model)
        elif model_name in ["hed", "ldc"]:
            result_grid = batch_edge_similarity(batch_A, batch_B, model, model_name)
        #elif model_name in ['depthpro', 'deepthanything-v2_large', 'dept_large']:
        #    result_grid = batch_depth_similarity(batch_A, batch_B, model)
        model_results_per_class[cls] = result_grid.cpu().numpy().flatten()

    del model
    torch.cuda.empty_cache()
    return model_results_per_class

def calculate_depths_metrics(set_A, set_B, common_classes, model_name, groups_A, groups_B, device):
    model = _get_depth_model(model_name)
    model_results_per_class = {}

    for cls in common_classes:
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]
        
        if not images_A or not images_B:
            continue
        
        batch_A = torch.stack(images_A).to(device)
        batch_B = torch.stack(images_B).to(device)
        # Call a targeted batched function for just this model
        mae_grid, ssim_grid, spear_grid, dists_grid = batch_depth_analysis(batch_A, batch_B, model)
        class_resuls = {
            f"{model_name}_mae": mae_grid.flatten(),
            f"{model_name}_ssim": ssim_grid.flatten(),
            f"{model_name}_spear": spear_grid.flatten(),
            f"{model_name}_dists": dists_grid.flatten()
        }
        if cls not in model_results_per_class:
            model_results_per_class[cls] = {k: [] for k in class_resuls.keys()}
        for k, v in class_resuls.items():
            model_results_per_class[cls][k].append(v)

    del model
    torch.cuda.empty_cache()
    return model_results_per_class

def calculate_lpips_metrics(set_A, set_B, common_classes, model_name, groups_A, groups_B, device):
    model = _get_lpips_model().to(device).eval()
    model_results_per_class = {}

    for cls in common_classes:
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]
        
        if not images_A or not images_B:
            continue
        
        batch_A = torch.stack(images_A).to(device)
        batch_B = torch.stack(images_B).to(device)
        
        # result_grid is (N, M)
        result_grid = batch_lpips_distance(batch_A, batch_B, model)
        
        # Standardize the key name
        metric_key = f"{model_name}_score"
        
        if cls not in model_results_per_class:
            model_results_per_class[cls] = {metric_key: []}
        
        model_results_per_class[cls][metric_key].append(result_grid.cpu().numpy().flatten())

    del model
    torch.cuda.empty_cache()
    return model_results_per_class

def calculate_edge_metrics(set_A, set_B, common_classes, model_name, groups_A, groups_B, device):
    if model_name=="lpips":
        model = _get_lpips_model().to(device).eval()
    
    elif model_name=="hed":
        model = _get_hed_model().to(device).eval()
    
    elif model_name=="ldc":
        model = _get_ldc_model().to(device)

    model_results_per_class = {}

    for cls in common_classes:
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]
        
        if not images_A or not images_B:
            continue
        
        batch_A = torch.stack(images_A).to(device)
        batch_B = torch.stack(images_B).to(device)
        # Call a targeted batched function for just this model
        ssim_grid, dists_grid, fom_grid, haus_grid = batch_edge_analysis(batch_A, batch_B, model)
        class_resuls = {
            f"{model_name}_ssim": ssim_grid.flatten(),
            f"{model_name}_dists": dists_grid.flatten(),
            f"{model_name}_fom": fom_grid.flatten(),
            f"{model_name}_haus": haus_grid.flatten()
        }
        if cls not in model_results_per_class:
            model_results_per_class[cls] = {k: [] for k in class_resuls.keys()}
        for k, v in class_resuls.items():
            model_results_per_class[cls][k].append(v)

    del model
    torch.cuda.empty_cache()
    return model_results_per_class

def model_analysis(set_A, set_B, common_classes, groups_A, groups_B, models):
    metrics_report = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for model_name in tqdm(models, desc="Going through models:"):
            print(f"Processing all dataset classes using Model Phase: {model_name}")
            if model_name in ['depthpro', 'depthanything_v2_large', 'dpt_large']:
                model_results=calculate_depths_metrics(set_A, set_B, common_classes, model_name, groups_A, groups_B, device)

            elif model_name=="lpips":
                model_results=calculate_lpips_metrics(set_A, set_B, common_classes, model_name, groups_A, groups_B, device)
        
            else:
                model_results=calculate_edge_metrics(set_A, set_B, common_classes, model_name,groups_A, groups_B, device)

            if model_results:
                    for cls, results_dict in model_results.items():
                        # If this is the first model for this class, create the sub-dict
                        if cls not in metrics_report:
                            metrics_report[cls] = {}
                        
                        for metric_name, arrays_list in results_dict.items():
                            # Concatenate and store. 
                            # Since metric_name includes the model name (e.g. 'ldc_ssim'), 
                            # it won't overwrite 'depthpro_ssim'.
                            flat_data = np.concatenate(arrays_list)
                            metrics_report[cls][metric_name] = flat_data
        
    return metrics_report

    