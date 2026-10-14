import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_batch_prediction(model, val_loader, device, epoch, output_dir):
    model.eval()
    try:
        images, masks = next(iter(val_loader))
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            # Models with the boundary module return (logits, edge_map); only
            # the segmentation logits are visualised here.
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            predictions = torch.sigmoid(outputs)
            
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        
        batch_size = images.size(0)
        num_samples = min(4, batch_size)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i in range(num_samples):
            axes[i, 0].imshow(images_np[i].squeeze(), cmap='gray')
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(masks_np[i].squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(preds_np[i].squeeze(), cmap='gray')
            axes[i, 2].set_title(f'Pred (Epoch {epoch+1})')
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'viz_epoch_{epoch+1}.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Visualization failed: {e}")