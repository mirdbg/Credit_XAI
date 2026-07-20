"""Utilidades compartidas para el MLP Keras baseline y heterogéneo.

El notebook 02B es el único que entrena y selecciona. Los notebooks 03 y 04
solo cargan el manifiesto persistido, verifican su huella y predicen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError:  # Permite inspeccionar el módulo sin TensorFlow instalado.
    tf = None


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="CreditXAI")
    class Log1pNonNegative(tf.keras.layers.Layer):
        """Aplica log(1 + max(x, 0)) de forma segura y serializable."""

        def call(self, inputs):
            return tf.math.log1p(tf.maximum(inputs, tf.cast(0.0, inputs.dtype)))

        def get_config(self):
            return super().get_config()
else:
    class Log1pNonNegative:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("Se necesita TensorFlow/Keras para usar Log1pNonNegative.")


def _rebuild_compact_heterogeneous_model(spec: Dict[str, Any]):
    """Reconstruye 02B para recuperar pesos de artefactos Lambda antiguos."""
    if tf is None:
        raise ImportError("Se necesita TensorFlow/Keras para reconstruir el modelo.")
    layers = tf.keras.layers
    regularizers = tf.keras.regularizers
    branches = spec["vector_branches"]
    embedding_cfg = spec["embedding_features"]
    delinquency = spec["shared_embedding_group"]
    independent = [c for c in embedding_cfg if c not in delinquency]
    reps, inputs = [], []

    for name, units in (("continuous_input", 16), ("integer_input", 4), ("binary_input", 8)):
        columns = branches[name]
        inp = tf.keras.Input((len(columns),), name=name)
        inputs.append(inp)
        if name == "continuous_input":
            x = Log1pNonNegative(name="continuous_log1p")(inp)
            x = layers.Normalization(name=name + "_norm")(x)
        elif name == "binary_input":
            x = inp
        else:
            x = layers.Normalization(name=name + "_norm")(inp)
        reps.append(layers.Dense(units, activation="relu", name=name + "_dense")(x))

    count_columns = branches["count_numeric_input"]
    count_input = tf.keras.Input((len(count_columns),), name="count_numeric_input")
    inputs.append(count_input)
    count_x = Log1pNonNegative(name="counts_log1p")(count_input)
    count_x = layers.Normalization(name="counts_log_norm")(count_x)
    reps.append(layers.Dense(12, activation="relu", name="count_numeric_dense")(count_x))

    shared_dim = max(embedding_cfg[c]["input_dim"] for c in delinquency)
    shared_embedding = layers.Embedding(shared_dim, 3, name="shared_delinquency_embedding")
    for feature in delinquency:
        inp = tf.keras.Input((1,), dtype="int32", name="emb_" + feature)
        inputs.append(inp)
        reps.append(layers.Flatten(name="flat_" + feature)(shared_embedding(inp)))
    for feature in independent:
        cfg = embedding_cfg[feature]
        inp = tf.keras.Input((1,), dtype="int32", name="emb_" + feature)
        inputs.append(inp)
        emb = layers.Embedding(
            cfg["input_dim"], cfg["embedding_dim"], name="embedding_" + feature
        )(inp)
        reps.append(layers.Flatten(name="flat_" + feature)(emb))

    x = layers.Concatenate(name="concatenate_branches")(reps)
    x = layers.GaussianNoise(0.03, name="representation_noise")(x)
    for units, dropout in ((64, 0.35), (32, 0.25)):
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(5e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout)(x)
    output = layers.Dense(1, activation="sigmoid", name="probability")(x)
    return tf.keras.Model(inputs, output, name="heterogeneous_mlp")


def stable_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_selection_manifest(path: Path = Path("outputs/models/mlp_selection_manifest.json")) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. Ejecuta primero 02B_MLP_Heterogeneo_Keras.ipynb."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    spec = manifest["preprocessing_spec"]
    observed = stable_hash(spec)
    if observed != manifest["preprocessing_hash"]:
        raise ValueError("La especificación de preprocesamiento fue modificada: huella inválida.")
    return manifest


def dataframe_to_keras_inputs(df: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convierte un DataFrame preprocesado en el contrato exacto del modelo."""
    missing = sorted(set(spec["features"]) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan variables requeridas: {missing}")
    inputs: Dict[str, np.ndarray] = {}
    for branch, columns in spec["vector_branches"].items():
        if columns:
            inputs[branch] = df[columns].astype("float32").to_numpy()
    for feature, cfg in spec["embedding_features"].items():
        values = np.rint(df[feature].to_numpy()).astype("int32")
        values = np.clip(values, 0, int(cfg["max_token"]))
        inputs[f"emb_{feature}"] = values.reshape(-1, 1)
    return inputs


def load_keras_model_from_manifest(
    manifest: Dict[str, Any], models_dir: Path = Path("outputs/models")
):
    if tf is None:
        raise ImportError("Se necesita TensorFlow/Keras para cargar el MLP seleccionado.")
    model_path = models_dir / manifest["selected_model"]["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Falta el modelo seleccionado: {model_path}")

    # El heterogéneo se reconstruye siempre desde código conocido y solo se
    # leen sus pesos. Así la ejecución de 03/04 no depende del registro global
    # de clases personalizadas ni de la caché del kernel de Jupyter.
    if manifest["selected_model"]["name"] == "heterogeneous_mlp":
        model = _rebuild_compact_heterogeneous_model(manifest["preprocessing_spec"])
        model.load_weights(model_path)
        return model

    custom_objects = {
        "Log1pNonNegative": Log1pNonNegative,
        "CreditXAI>Log1pNonNegative": Log1pNonNegative,
    }
    try:
        return tf.keras.models.load_model(
            model_path, custom_objects=custom_objects, compile=False
        )
    except ValueError as exc:
        # Compatibilidad con artefactos locales anteriores creados con Lambda:
        # reconstruimos código conocido y recuperamos solo los pesos.
        if "Lambda" not in str(exc) and "lambda" not in str(exc):
            raise
        model = _rebuild_compact_heterogeneous_model(manifest["preprocessing_spec"])
        model.load_weights(model_path)
        return model


def predict_selected_proba(model, df: pd.DataFrame, manifest: Dict[str, Any], batch_size: int = 4096) -> np.ndarray:
    spec = manifest["preprocessing_spec"]
    if manifest["selected_model"]["name"] == "baseline_keras":
        inputs = df[spec["features"]].astype("float32").to_numpy()
    else:
        inputs = dataframe_to_keras_inputs(df, spec)
    return np.asarray(model.predict(inputs, batch_size=batch_size, verbose=0)).reshape(-1)


def verify_runtime_contract(model, df: pd.DataFrame, manifest: Dict[str, Any]) -> None:
    """Falla pronto si columnas, nombres de entradas o huella no coinciden."""
    spec = manifest["preprocessing_spec"]
    assert stable_hash(spec) == manifest["preprocessing_hash"]
    if manifest["selected_model"]["name"] == "baseline_keras":
        inputs = df[spec["features"]].head(2).astype("float32").to_numpy()
        if len(model.inputs) != 1 or int(model.inputs[0].shape[-1]) != len(spec["features"]):
            raise ValueError("El baseline no coincide con el número de variables del manifiesto.")
    else:
        inputs = dataframe_to_keras_inputs(df.head(2), spec)
        expected = sorted(t.name.split(":")[0] for t in model.inputs)
        observed = sorted(inputs)
        if expected != observed:
            raise ValueError(f"Contrato de entradas distinto. Modelo={expected}; datos={observed}")
    _ = model.predict(inputs, verbose=0)
