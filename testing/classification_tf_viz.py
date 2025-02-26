import matplotlib.pyplot as plt
import json
import os

def main():
    # Créer le dossier pour les plots s'il n'existe pas
    plots_dir = "data/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Charger les métriques
    model_time = "20250226_171713"
    metrics_dir = "data/metrics"
    metrics_files = [f for f in os.listdir(metrics_dir) if f.startswith(f"classification_model_{model_time}_epoch_")]
    metrics_files.sort(key=lambda x: int(x.split("_epoch_")[1].split("_")[0]))

    # Collecter toutes les métriques
    accuracy = []
    val_accuracy = []
    loss = []
    val_loss = []

    for metrics_file in metrics_files:
        with open(os.path.join(metrics_dir, metrics_file), 'r') as f:
            metrics = json.load(f)
            accuracy.append(metrics['accuracy'])
            val_accuracy.append(metrics['val_accuracy'])
            loss.append(metrics['loss'])
            val_loss.append(metrics['val_loss'])

    # Tracer et sauvegarder les courbes d'accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, label='Accuracy', marker='o')
    plt.plot(val_accuracy, label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    accuracy_plot_path = os.path.join(plots_dir, f"accuracy_plot_{model_time}.png")
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Tracer et sauvegarder les courbes de loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(plots_dir, f"loss_plot_{model_time}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Tracer les deux courbes côte à côte pour l'affichage
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(accuracy, label='Accuracy', marker='o')
    plt.plot(val_accuracy, label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    combined_plot_path = os.path.join(plots_dir, f"combined_metrics_{model_time}.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Afficher les métriques finales
    print("\nMétriques finales:")
    print(f"Accuracy: {accuracy[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracy[-1]:.4f}")
    print(f"Loss: {loss[-1]:.4f}")
    print(f"Validation Loss: {val_loss[-1]:.4f}")

    # Sauvegarder les métriques finales dans un fichier texte
    metrics_summary_path = os.path.join(plots_dir, f"metrics_summary_{model_time}.txt")
    with open(metrics_summary_path, 'w') as f:
        f.write("Métriques finales:\n")
        f.write(f"Accuracy: {accuracy[-1]:.4f}\n")
        f.write(f"Validation Accuracy: {val_accuracy[-1]:.4f}\n")
        f.write(f"Loss: {loss[-1]:.4f}\n")
        f.write(f"Validation Loss: {val_loss[-1]:.4f}\n")

    print(f"\nGraphiques et métriques sauvegardés dans {plots_dir}/")

if __name__ == "__main__":
    main() 