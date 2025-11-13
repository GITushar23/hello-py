"""
Dataset Corruption Detection Task

Contains:
- PROMPT: Challenge prompt for the agent
- TOOLS: Tool definitions for Anthropic API
- TOOL_HANDLERS: Tool handler functions
- grading_func: Validates agent's answer
"""

import json
import signal
from collections.abc import Callable
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from anthropic.types import ToolUnionParam
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Global state
# ============================================================================
_CORRUPTED_DATASET = None
_GROUND_TRUTH_NOISY_INDICES = None
_PYTHON_NAMESPACE = None
_LAST_SUBMITTED_PREDICTIONS = []


# ============================================================================
# Dataset corruption
# ============================================================================
def create_corrupted_dataset(noise_rate: float = 0.20, seed: int = 42) -> tuple:
    """Creates Fashion-MNIST dataset with instance-dependent + asymmetric noise."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    X_train = train_dataset.data.numpy().astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
    y_test = test_dataset.targets.numpy()

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # Semantic confusion matrix for Fashion-MNIST
    confusion_groups = {
        0: [6, 2, 4], 1: [3], 2: [0, 6, 4], 3: [1], 4: [0, 2, 6],
        5: [7, 9], 6: [0, 2, 4], 7: [5, 9], 8: [], 9: [7, 5],
    }

    # Train small model to identify hard samples
    print("Training small model to identify hard samples...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=256,
        shuffle=True
    )

    model.train()
    for _ in range(3):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # Get prediction confidence
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_train).to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()

    confidence = probs[np.arange(len(y_train)), y_train]

    # Create instance-dependent corruption
    n_samples = len(y_train)
    n_corrupt = int(noise_rate * n_samples)
    n_hard = int(0.4 * n_corrupt)
    n_medium = n_corrupt - n_hard

    sorted_indices = np.argsort(confidence)
    hard_pool = sorted_indices[:int(0.3 * n_samples)]
    hard_corrupted = np.random.choice(hard_pool, size=n_hard, replace=False)
    medium_pool = sorted_indices[int(0.3 * n_samples):int(0.7 * n_samples)]
    medium_corrupted = np.random.choice(medium_pool, size=n_medium, replace=False)
    noisy_indices = np.concatenate([hard_corrupted, medium_corrupted])

    # Apply asymmetric noise
    y_noisy = y_train.copy()
    for idx in noisy_indices:
        true_class = y_train[idx]
        similar_classes = confusion_groups.get(true_class, [])

        if similar_classes and np.random.random() < 0.7:
            y_noisy[idx] = np.random.choice(similar_classes)
        else:
            sample_probs = probs[idx]
            top_indices = np.argsort(sample_probs)[::-1]
            candidates = [c for c in top_indices[:3] if c != true_class]
            if candidates:
                candidate_probs = sample_probs[candidates]
                candidate_probs = candidate_probs / candidate_probs.sum()
                y_noisy[idx] = np.random.choice(candidates, p=candidate_probs)
            else:
                other_classes = [c for c in range(10) if c != true_class]
                y_noisy[idx] = np.random.choice(other_classes)

    print(f"Created dataset with {len(noisy_indices)} corrupted samples ({noise_rate*100:.1f}% noise rate)")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, y_noisy, X_test, y_test, noisy_indices


# ============================================================================
# Tool implementations
# ============================================================================
def load_corrupted_dataset_tool() -> dict[str, Any]:
    """Loads the Fashion-MNIST dataset with corrupted labels."""
    global _CORRUPTED_DATASET, _GROUND_TRUTH_NOISY_INDICES, _PYTHON_NAMESPACE

    if _CORRUPTED_DATASET is None:
        X_train, y_train, X_test, y_test, noisy_indices = create_corrupted_dataset()
        _CORRUPTED_DATASET = (X_train, y_train, X_test, y_test)
        _GROUND_TRUTH_NOISY_INDICES = set(noisy_indices)
        _PYTHON_NAMESPACE = None

    X_train, y_train, X_test, y_test = _CORRUPTED_DATASET

    return {
        "message": "Dataset loaded successfully. Access via X_train, y_train, X_test, y_test in python_expression tool.",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_classes": 10,
        "noise_rate": len(_GROUND_TRUTH_NOISY_INDICES) / len(X_train),
    }


def submit_predictions_tool(predictions_file: str) -> dict[str, Any]:
    """Submit predictions for which training samples are corrupted."""
    global _LAST_SUBMITTED_PREDICTIONS

    try:
        noisy_indices = np.load(predictions_file)

        if not isinstance(noisy_indices, np.ndarray):
            return {"submitted": False, "error": "File must contain a numpy array", "num_predicted_noisy": 0}

        noisy_indices_list = noisy_indices.flatten().astype(int).tolist()
        _LAST_SUBMITTED_PREDICTIONS = noisy_indices_list

        return {
            "submitted": True,
            "num_predicted_noisy": len(noisy_indices_list),
            "predictions_file": predictions_file,
        }

    except FileNotFoundError:
        return {
            "submitted": False,
            "error": f"File '{predictions_file}' not found. Make sure you saved it first.",
            "num_predicted_noisy": 0,
        }
    except Exception as e:
        return {"submitted": False, "error": f"Error loading predictions: {str(e)}", "num_predicted_noisy": 0}


class TimeoutError(Exception):
    """Raised when code execution exceeds timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Code execution exceeded 5 minute timeout")


def python_expression_tool(expression: str) -> dict[str, Any]:
    """Evaluates Python expressions using exec. Variables persist across calls."""
    global _CORRUPTED_DATASET, _PYTHON_NAMESPACE

    try:
        if _PYTHON_NAMESPACE is None:
            _PYTHON_NAMESPACE = {'np': np, 'torch': torch}
            import sklearn
            _PYTHON_NAMESPACE['sklearn'] = sklearn

        if _CORRUPTED_DATASET is not None and 'X_train' not in _PYTHON_NAMESPACE:
            X_train, y_train, X_test, y_test = _CORRUPTED_DATASET
            _PYTHON_NAMESPACE.update({
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
            })

        # Set timeout (Unix only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)
        except (AttributeError, ValueError):
            pass

        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, _PYTHON_NAMESPACE, _PYTHON_NAMESPACE)

        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass

        return {"result": stdout.getvalue(), "error": None}

    except TimeoutError as e:
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass
        return {"result": None, "error": str(e)}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass
        return {"result": None, "error": str(e)}


# ============================================================================
# Tool definitions
# ============================================================================
TOOLS: list[ToolUnionParam] = [
    {
        "name": "load_corrupted_dataset",
        "description": "Loads Fashion-MNIST dataset with ~20% label corruption. Returns dataset info and makes X_train, y_train, X_test, y_test available in python_expression tool. Call this first before analyzing the data.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "python_expression",
        "description": "Evaluates Python code with a 5-minute timeout. Dataset arrays (X_train, y_train, X_test, y_test) are available after loading. Has numpy (np), torch, and sklearn available. IMPORTANT: Variables persist across calls - you can define variables in one call and use them in later calls. If code takes >5 minutes, it will be terminated with a timeout error. Use print() to output results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Python code to execute. Has 5-minute timeout. Variables defined here will be available in future calls. Use print() to see output.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "submit_predictions",
        "description": "Submit your final predictions for which training samples have corrupted labels. IMPORTANT: Save your predictions to a file first using np.save('predictions.npy', noisy_indices_array) in python_expression tool, then submit the filename here. Do NOT try to pass the full list of indices directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "predictions_file": {
                    "type": "string",
                    "description": "Path to .npy file containing your predictions (e.g., 'predictions.npy'). File should contain a numpy array of integer indices (0-59999).",
                }
            },
            "required": ["predictions_file"],
        },
    },
]

TOOL_HANDLERS: dict[str, Callable[..., Any]] = {
    "load_corrupted_dataset": load_corrupted_dataset_tool,
    "python_expression": python_expression_tool,
    "submit_predictions": submit_predictions_tool,
}


# ============================================================================
# Task prompt
# ============================================================================
PROMPT = """You are given a Fashion-MNIST dataset with corrupted labels. Approximately 15-20% of the training labels are incorrect due to annotation errors.

Your task: Identify which training samples have corrupted labels.

Available tools:
1. load_corrupted_dataset: Load the dataset (call this first)
2. python_expression: Run Python code to analyze data and implement detection methods
   - Libraries available: numpy (np), torch, sklearn
   - After loading, X_train, y_train, X_test, y_test are accessible
   - Variables persist across calls
3. submit_predictions: Submit a file containing your predictions
   - IMPORTANT: First save predictions using np.save('predictions.npy', noisy_indices_array)
   - Then call submit_predictions with predictions_file='predictions.npy'

Constraints:
- You CANNOT use the cleanlab library
- You must implement your own detection logic
- Maximum 20 steps to solve this (one tool call per step)
- If you use torch consider using CUDA for faster execution.

Success criteria:
- Your predictions will be evaluated using F1 score (balance of precision and recall)
- Pass threshold: F1 ≥ 0.65 AND (Precision ≥ 0.55 OR Recall ≥ 0.55)
- Precision = (correctly identified noisy) / (total predicted noisy)
- Recall = (correctly identified noisy) / (actual noisy)
- This is a HARD threshold - simple approaches may not work!

Fashion-MNIST classes (0-9):
0: T-shirt, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

Workflow:
1. Load dataset with load_corrupted_dataset
2. Implement detection method using python_expression
3. Save predictions: np.save('predictions.npy', your_noisy_indices_array)
4. Submit: submit_predictions(predictions_file='predictions.npy')"""


# ============================================================================
# Grading function
# ============================================================================
def grading_func(result: Any) -> tuple[bool, dict[str, Any]]:
    """Validates the agent's corruption detection predictions. Returns (success, metrics)."""
    global _GROUND_TRUTH_NOISY_INDICES, _CORRUPTED_DATASET

    if _GROUND_TRUTH_NOISY_INDICES is None or _CORRUPTED_DATASET is None:
        print("ERROR: Dataset not loaded")
        return False, {}

    predicted_indices = set(_LAST_SUBMITTED_PREDICTIONS)
    true_indices = _GROUND_TRUTH_NOISY_INDICES

    if len(predicted_indices) == 0:
        print("ERROR: No predictions submitted")
        return False, {}

    # Calculate metrics
    true_positives = len(predicted_indices & true_indices)
    false_positives = len(predicted_indices - true_indices)
    false_negatives = len(true_indices - predicted_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print(f"\n=== Evaluation Metrics ===")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"=========================")

    passes = f1 >= 0.65 and (precision >= 0.55 or recall >= 0.55)

    if passes:
        print(f"✓ PASS: F1={f1:.4f} >= 0.65, and Precision={precision:.4f} or Recall={recall:.4f} >= 0.55")
    else:
        print(f"✗ FAIL: F1={f1:.4f} (need >= 0.65), Precision={precision:.4f}, Recall={recall:.4f} (need one >= 0.55)")

    return passes, metrics


def reset_task_state():
    """Reset all global state for a new run."""
    global _CORRUPTED_DATASET, _GROUND_TRUTH_NOISY_INDICES, _PYTHON_NAMESPACE, _LAST_SUBMITTED_PREDICTIONS
    _CORRUPTED_DATASET = None
    _GROUND_TRUTH_NOISY_INDICES = None
    _PYTHON_NAMESPACE = None
    _LAST_SUBMITTED_PREDICTIONS = []


# receipts@aichamp.com