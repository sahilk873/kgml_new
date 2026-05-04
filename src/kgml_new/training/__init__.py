from kgml_new.training.eval import (
    link_prediction_dot_product,
    link_prediction_mlp_torch,
    link_prediction_sklearn,
    node_classification_scores,
)
from kgml_new.training.history import TrainingHistory
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    create_train_val_split,
    train_unsupervised,
    train_unsupervised_batched,
    train_unsupervised_fullgraph,
)
from kgml_new.training.node2vec_train import (
    train_node2vec_embeddings,
    train_node2vec_embeddings_with_validation,
)

try:
    from kgml_new.training.node_classification import (
        NODE_CLASSIFICATION_METHODS,
        NodeClassificationHistory,
        train_embedding_node_classifier,
        train_native_node_classifier,
        train_txgnn_node_classifier,
    )
except ImportError:  # optional / not shipped in all checkouts
    NODE_CLASSIFICATION_METHODS = ()
    NodeClassificationHistory = None  # type: ignore[misc,assignment]
    train_embedding_node_classifier = None  # type: ignore[misc,assignment]
    train_native_node_classifier = None  # type: ignore[misc,assignment]
    train_txgnn_node_classifier = None  # type: ignore[misc,assignment]

from kgml_new.training.train_link_mlp import (
    train_link_mlp,
    train_link_mlp_with_validation,
)
from kgml_new.training.hgt_train import HGTTrainResult, evaluate_hgt_relation, train_hgt
from kgml_new.training.txgnn_train import TxGNNTrainResult, evaluate_relation, train_txgnn

__all__ = [
    "compute_node_embeddings",
    "create_train_val_split",
    "link_prediction_dot_product",
    "link_prediction_mlp_torch",
    "link_prediction_sklearn",
    "node_classification_scores",
    "NODE_CLASSIFICATION_METHODS",
    "NodeClassificationHistory",
    "TrainingHistory",
    "train_embedding_node_classifier",
    "train_link_mlp",
    "train_link_mlp_with_validation",
    "train_native_node_classifier",
    "train_txgnn_node_classifier",
    "train_node2vec_embeddings",
    "train_node2vec_embeddings_with_validation",
    "train_unsupervised",
    "train_unsupervised_batched",
    "train_unsupervised_fullgraph",
    "train_txgnn",
    "evaluate_relation",
    "TxGNNTrainResult",
    "train_hgt",
    "evaluate_hgt_relation",
    "HGTTrainResult",
]

if NodeClassificationHistory is None:
    __all__ = [x for x in __all__ if x not in frozenset({
        "NODE_CLASSIFICATION_METHODS",
        "NodeClassificationHistory",
        "train_embedding_node_classifier",
        "train_native_node_classifier",
        "train_txgnn_node_classifier",
    })]
