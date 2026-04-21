from dataclasses import dataclass

@dataclass
class Config:
    # General
    random_seed: int = 42
    device: str = "cuda"

    # Raw data paths
    raw_ton_iot_path: str = "data/raw/ton_raw/"
    raw_icu_path: str = "data/raw/sim_raw/"
    raw_cic_path: str = "data/raw/cic_raw/"

    # Processed data paths
    processed_ton_iot_path: str = "data/processed/ton_processed/"
    processed_icu_path: str = "data/processed/sim_processed/"
    processed_cic_path: str = "data/processed/cic_processed/"

    # Split directories
    ton_splits_path: str = "data/splits/ton_splits/"
    sim_splits_path: str = "data/splits/sim_splits/"
    cic_splits_path: str = "data/splits/cic_splits/"
    CIC_SPLIT_PATH = "data/splits/"
    CIC_DATASET = "cic"

    # Dataset columns
    target_column: str = "label"
    group_column: str = "device_id"   # may change for CIC
    input_dim: int = 50   # update after preprocessing CIC

    # Sequence settings
    seq_len: int = 20

    # Autoencoder
    latent_dim: int = 32
    ae_hidden_dim1: int = 128
    ae_hidden_dim2: int = 64
    ae_learning_rate: float = 0.001
    ae_batch_size: int = 64
    ae_epochs: int = 15

    # Mamba / LSTM shared
    transformer_input_dim: int = latent_dim + 1
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.2
    clf_learning_rate: float = 0.001
    clf_batch_size: int = 64
    clf_epochs: int = 15

    # Splits
    test_size: float = 0.2
    validation_size: float = 0.1

    # Model saving
    model_dir: str = "models"

    # Autoencoders
    ton_autoencoder_model_path: str = "models/autoencoder_ton.pt"
    sim_autoencoder_model_path: str = "models/autoencoder_sim.pt"
    cic_autoencoder_model_path: str = "models/autoencoder_cic.pt"

    # Classifiers (Mamba)
    ton_classifier_model_path: str = "models/mamba_classifier_ton.pt"
    sim_classifier_model_path: str = "models/mamba_classifier_sim.pt"
    cic_classifier_model_path: str = "models/mamba_classifier_cic.pt"

    # LSTM
    ton_lstm_model_path: str = "models/lstm_classifier_ton.pt"
    sim_lstm_model_path: str = "models/lstm_classifier_sim.pt"
    cic_lstm_model_path: str = "models/lstm_classifier_cic.pt"

    # Loss tracking
    loss_dir: str = "results/losses"

    # Simulated dataset classes
    sim_num_classes: int = 1

    # Evaluation
    threshold: float = 0.5