from .content_metrics import (_get_lpips_model, batch_lpips_distance,
                             _get_hed_model, _get_ldc_model, batch_edge_similarity, _get_depth_model,
                             calculate_depths_metrics, calculate_edge_metrics, calculate_lpips_metrics,
                             clear_global_models,)
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

    elif model_name in ["depthpro", "depthanything_v2_large", "dpt_large"]:
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

def pixel_model_analysis(set_A, set_B, common_classes, groups_A, groups_B, models, intra):
    metrics_report = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in tqdm(models, desc="Going through models:"):
        print(f"Processing all dataset classes using Model Phase: {model_name}")

        torch.cuda.empty_cache()
        with torch.no_grad():
            #if model_name in ['depthpro', 'depthanything_v2_large', 'dpt_large']:
            #    model_results=calculate_depths_metrics(set_A, set_B, common_classes, model_name, groups_A, groups_B, device, intra)

            if model_name=="lpips":
                model_results=calculate_lpips_metrics(set_A, set_B, common_classes, model_name, groups_A, groups_B, device, intra)
        
            else:
                model_results=calculate_edge_metrics(set_A, set_B, common_classes, model_name,groups_A, groups_B, device, intra)

        for cls, metric_dict in model_results.items():
            if cls not in metrics_report:
                metrics_report[cls] = {}
            metrics_report[cls].update(metric_dict)
    clear_global_models()
    return metrics_report

    