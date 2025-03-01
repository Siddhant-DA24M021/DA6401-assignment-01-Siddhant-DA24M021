import wandb
import numpy as np

def log_predicted_images(model, X_test, X_test_std, y_test, class_names, num_samples = 10):

    # Select samples
    sample_indices = np.random.choice(X_test.shape[0], num_samples)
    X_samples = X_test[sample_indices]
    X_samples_std = X_test_std[sample_indices]
    y_samples = y_test[sample_indices]

    # Get predictions
    sample_preds = model.forward(X_samples_std)
    pred_classes = class_names[np.argmax(sample_preds, axis = 1)]
    true_classes = class_names[y_samples]


    columns = ["Image", "True Label", "Predicted Label"]
    prediction_table = wandb.Table(columns=columns)
    
    for i in range(num_samples):
        img = wandb.Image(X_samples[i])
        
        true_label = true_classes[i]
        pred_label = pred_classes[i]
        prediction_table.add_data(
            img, 
            true_label, 
            pred_label
        )
    
    # Log the table to wandb
    wandb.log({"Prediction Samples": prediction_table})