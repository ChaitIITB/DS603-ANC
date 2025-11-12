import numpy as np
import pandas as pd
import os
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class HARPreprocessor:
    """
    Preprocessor for UCI-HAR dataset

    Dataset Structure:
    - 10,299 samples
    - 561 features (time and frequency domain features)
    - 6 activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, 
                   SITTING, STANDING, LAYING
    - 30 subjects
    - Standard split: 70% train (7352 samples), 30% test (2947 samples) 
    """

    def __init__(self, data_dir='./data/UCI HAR Dataset'):
        """
        Initialize preprocessor

        Args:
            data_dir: Path to UCI HAR dataset directory
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')

        # Activity labels (1-6)
        self.activity_labels = {
            1: 'WALKING',
            2: 'WALKING_UPSTAIRS', 
            3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING',
            5: 'STANDING',
            6: 'LAYING'
        }

        self.feature_names = None
        self.scaler = StandardScaler()

    def load_raw_signals(self, data_type='train'):
        """
        Load raw inertial signals from the Inertial Signals folder

        Returns:
            X_signals: (n_samples, 128, 9) array
            y_labels: (n_samples,) array
            subjects: (n_samples,) array
        """
        signals_dir = os.path.join(self.train_dir if data_type == 'train' else self.test_dir, 
                                   'Inertial Signals')

        # Signal files (9 signals: 3 accel + 3 gyro, each with body and total components)
        signal_files = [
            'body_acc_x_{}.txt',
            'body_acc_y_{}.txt', 
            'body_acc_z_{}.txt',
            'body_gyro_x_{}.txt',
            'body_gyro_y_{}.txt',
            'body_gyro_z_{}.txt',
            'total_acc_x_{}.txt',
            'total_acc_y_{}.txt',
            'total_acc_z_{}.txt'
        ]

        signals = []
        for signal_file in signal_files:
            file_path = os.path.join(signals_dir, signal_file.format(data_type))
            signal_data = np.loadtxt(file_path)
            signals.append(signal_data)

        # Stack signals: (n_samples, 128, 9)
        X_signals = np.stack(signals, axis=2)

        # Load labels
        y_file = os.path.join(self.train_dir if data_type == 'train' else self.test_dir, 
                             f'y_{data_type}.txt')
        y_labels = np.loadtxt(y_file, dtype=int) - 1  # Convert to 0-indexed

        # Load subjects
        subject_file = os.path.join(self.train_dir if data_type == 'train' else self.test_dir,
                                   f'subject_{data_type}.txt')
        subjects = np.loadtxt(subject_file, dtype=int)

        print(f"Loaded {data_type} raw signals: {X_signals.shape}")
        return X_signals, y_labels, subjects

    def load_features(self, data_type='train'):
        """
        Load preprocessed features (561 features)

        Returns:
            X_features: (n_samples, 561) array
            y_labels: (n_samples,) array  
            subjects: (n_samples,) array
        """
        # Load features
        X_file = os.path.join(self.train_dir if data_type == 'train' else self.test_dir,
                             f'X_{data_type}.txt')
        X_features = np.loadtxt(X_file)

        # Load labels
        y_file = os.path.join(self.train_dir if data_type == 'train' else self.test_dir,
                             f'y_{data_type}.txt')
        y_labels = np.loadtxt(y_file, dtype=int) - 1  # Convert to 0-indexed

        # Load subjects
        subject_file = os.path.join(self.train_dir if data_type == 'train' else self.test_dir,
                                   f'subject_{data_type}.txt')
        subjects = np.loadtxt(subject_file, dtype=int)

        # Load feature names
        if self.feature_names is None:
            features_file = os.path.join(self.data_dir, 'features.txt')
            self.feature_names = pd.read_csv(features_file, sep=' ', header=None)[1].values

        print(f"Loaded {data_type} features: {X_features.shape}")
        return X_features, y_labels, subjects

    def normalize_data(self, X_train, X_test):
        """
        Normalize features using StandardScaler fitted on training data

        Args:
            X_train: Training data
            X_test: Test data

        Returns:
            X_train_norm, X_test_norm: Normalized data
        """
        # Reshape if needed (for raw signals)
        original_shape_train = X_train.shape
        original_shape_test = X_test.shape

        if len(X_train.shape) == 3:
            # Raw signals: (n_samples, 128, 9) -> (n_samples, 128*9)
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        # Fit scaler on training data only
        X_train_norm = self.scaler.fit_transform(X_train)
        X_test_norm = self.scaler.transform(X_test)

        # Reshape back if needed
        if len(original_shape_train) == 3:
            X_train_norm = X_train_norm.reshape(original_shape_train)
            X_test_norm = X_test_norm.reshape(original_shape_test)

        print(f"Data normalized using StandardScaler")
        return X_train_norm, X_test_norm

    def compute_statistics(self, X_train, y_train, X_test, y_test, subjects_train, subjects_test):
        """
        Compute and display dataset statistics for paper

        Returns:
            stats_dict: Dictionary with statistics
        """
        stats_dict = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X_train) + len(X_test),
            'n_features': X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[1:],
            'n_classes': len(np.unique(y_train)),
            'n_subjects': len(np.unique(np.concatenate([subjects_train, subjects_test]))),
            'train_subjects': len(np.unique(subjects_train)),
            'test_subjects': len(np.unique(subjects_test))
        }

        # Class distribution
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {stats_dict['total_samples']}")
        print(f"Train samples: {stats_dict['train_samples']} (70%)")
        print(f"Test samples: {stats_dict['test_samples']} (30%)")
        print(f"Features: {stats_dict['n_features']}")
        print(f"Classes: {stats_dict['n_classes']}")
        print(f"Subjects: {stats_dict['n_subjects']}")

        print("\n=== Class Distribution (Training Set) ===")
        for label in range(6):
            count = np.sum(y_train == label)
            percentage = 100 * count / len(y_train)
            print(f"{self.activity_labels[label+1]:20s}: {count:5d} samples ({percentage:.2f}%)")

        print("\n=== Class Distribution (Test Set) ===")
        for label in range(6):
            count = np.sum(y_test == label)
            percentage = 100 * count / len(y_test)
            print(f"{self.activity_labels[label+1]:20s}: {count:5d} samples ({percentage:.2f}%)")

        return stats_dict

    def plot_sample_data(self, X_signals, y_labels, save_dir='./plots'):
        """
        Plot sample time-series for each activity class

        Args:
            X_signals: Raw signal data (n_samples, 128, 9)
            y_labels: Labels
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(6, 3, figsize=(15, 18))
        fig.suptitle('Sample Time-Series for Each Activity', fontsize=16)

        signal_names = ['Acc-X', 'Acc-Y', 'Acc-Z']

        for activity_idx in range(6):
            # Get one sample for this activity
            sample_idx = np.where(y_labels == activity_idx)[0][0]
            sample_data = X_signals[sample_idx]

            for signal_idx in range(3):
                ax = axes[activity_idx, signal_idx]
                ax.plot(sample_data[:, signal_idx])
                ax.set_ylabel(self.activity_labels[activity_idx+1], fontsize=10)
                if activity_idx == 0:
                    ax.set_title(signal_names[signal_idx])
                if activity_idx == 5:
                    ax.set_xlabel('Time Step')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_timeseries.png'), dpi=300, bbox_inches='tight')
        print(f"Saved sample time-series plot to {save_dir}/sample_timeseries.png")
        plt.close()

    def plot_class_distribution(self, y_train, y_test, save_dir='./plots'):
        """
        Plot class distribution for train and test sets
        """
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Training set
        train_counts = [np.sum(y_train == i) for i in range(6)]
        axes[0].bar(range(6), train_counts, color='steelblue', alpha=0.8)
        axes[0].set_xticks(range(6))
        axes[0].set_xticklabels([self.activity_labels[i+1] for i in range(6)], 
                               rotation=45, ha='right')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Training Set Class Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Test set
        test_counts = [np.sum(y_test == i) for i in range(6)]
        axes[1].bar(range(6), test_counts, color='coral', alpha=0.8)
        axes[1].set_xticks(range(6))
        axes[1].set_xticklabels([self.activity_labels[i+1] for i in range(6)],
                               rotation=45, ha='right')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Test Set Class Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight') 
        print(f"Saved class distribution plot to {save_dir}/class_distribution.png")
        plt.close()

    def save_processed_data(self, X_train, y_train, subjects_train,
                           X_test, y_test, subjects_test,
                           save_dir='./processed_data',
                           data_type='features'):
        """
        Save processed data to disk

        Args:
            data_type: 'features' or 'signals'
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save as numpy arrays
        np.save(os.path.join(save_dir, f'X_train_{data_type}.npy'), X_train)
        np.save(os.path.join(save_dir, f'y_train_{data_type}.npy'), y_train)
        np.save(os.path.join(save_dir, f'subjects_train_{data_type}.npy'), subjects_train)

        np.save(os.path.join(save_dir, f'X_test_{data_type}.npy'), X_test)
        np.save(os.path.join(save_dir, f'y_test_{data_type}.npy'), y_test)
        np.save(os.path.join(save_dir, f'subjects_test_{data_type}.npy'), subjects_test)

        # Save scaler
        with open(os.path.join(save_dir, f'scaler_{data_type}.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            'activity_labels': self.activity_labels,
            'feature_names': self.feature_names,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'n_classes': 6,
            'data_type': data_type
        }
        with open(os.path.join(save_dir, f'metadata_{data_type}.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"\nSaved processed data to {save_dir}/")
        print(f"  - X_train_{data_type}.npy: {X_train.shape}")
        print(f"  - X_test_{data_type}.npy: {X_test.shape}")
        print(f"  - Labels, subjects, scaler, and metadata")


if __name__ == "main":
    """
    Main preprocessing pipeline
    """
    print("="*60)
    print("UCI-HAR Dataset Preprocessing")
    print("="*60)

    # Initialize preprocessor
    preprocessor = HARPreprocessor(data_dir='./data/UCI HAR Dataset')

    # OPTION 1: Load preprocessed features (561 features)
    # Use this for faster training with engineered features
    print("\n[Option 1] Loading preprocessed features...")
    X_train_feat, y_train, subjects_train = preprocessor.load_features('train')
    X_test_feat, y_test, subjects_test = preprocessor.load_features('test')

    # Normalize
    X_train_feat_norm, X_test_feat_norm = preprocessor.normalize_data(X_train_feat, X_test_feat)

    # Compute statistics
    stats = preprocessor.compute_statistics(X_train_feat_norm, y_train, 
                                           X_test_feat_norm, y_test,
                                           subjects_train, subjects_test)

    # Save processed features
    preprocessor.save_processed_data(X_train_feat_norm, y_train, subjects_train,
                                     X_test_feat_norm, y_test, subjects_test,
                                     save_dir='./processed_data',
                                     data_type='features')

    # OPTION 2: Load raw signals (128 timesteps, 9 channels)
    # Use this for LSTM/GRU/Transformer models that process sequences
    print("\n[Option 2] Loading raw inertial signals...")
    X_train_sig, y_train, subjects_train = preprocessor.load_raw_signals('train')
    X_test_sig, y_test, subjects_test = preprocessor.load_raw_signals('test')

    # Normalize
    X_train_sig_norm, X_test_sig_norm = preprocessor.normalize_data(X_train_sig, X_test_sig)

    # Save processed signals
    preprocessor.save_processed_data(X_train_sig_norm, y_train, subjects_train,
                                     X_test_sig_norm, y_test, subjects_test,
                                     save_dir='./processed_data',
                                     data_type='signals')

    # Create visualizations
    print("\nGenerating visualizations...")
    preprocessor.plot_class_distribution(y_train, y_test)
    preprocessor.plot_sample_data(X_train_sig_norm, y_train)

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use 'signals' data for LSTM/GRU/Transformer models")
    print("2. Use 'features' data for faster baseline experiments")
    print("3. Check ./plots/ for data visualizations")
    print("4. Proceed to training baseline models")