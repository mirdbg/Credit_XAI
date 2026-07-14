"""
Utilidades de preprocesado compartidas entre los notebooks 01, 02, 03 y 04.

Este fichero existe por un motivo muy concreto: `ScaledKNNImputer` se guarda
con joblib dentro de `preprocessing_objects.joblib` (notebook 01) y se vuelve
a cargar en los notebooks 02, 03 y 04. Si la clase estuviera definida
directamente dentro de una celda del notebook 01 (como estaba originalmente),
Python no podría reconstruir el objeto en los demás notebooks, porque cada
notebook corre en su propio kernel/proceso y la clase solo existiría en
`__main__` del notebook 01.

Al definirla aquí, en un módulo .py normal, cualquier notebook que haga
`from preprocessing_utils import ScaledKNNImputer` antes de cargar el
joblib podrá reconstruir el objeto sin problema, esté donde esté.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import StandardScaler


@dataclass
class ScaledKNNImputer:
    """
    Imputador KNN con escalado interno y método explicativo.

    Además de imputar valores perdidos, permite explicar una imputación concreta
    mostrando qué filas similares se han usado como referencia.

    Importante:
    - La explicación no replica exactamente el cálculo interno de KNNImputer.
    - Para explicar una variable concreta, busca vecinos eliminando esa variable
      del cálculo de distancia.
    - Esto evita comparar usando justo la columna que estaba vacía.
    """
    columns: List[str]
    n_neighbors: int = 7
    weights: str = "distance"
    fit_sample_size: int = 10000
    random_state: int = 42

    def __post_init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
        )
        self.fitted_ = False
        self.fit_indices_ = None
        self.fit_df_ = None
        self.train_imputed_ = None

    def fit(self, df: pd.DataFrame):
        x_all = df[self.columns].astype(float).copy()

        self.fit_df_ = df.copy()

        self.scaler.fit(x_all)

        rng = np.random.default_rng(self.random_state)

        missing_mask = x_all.isna().any(axis=1)
        missing_indices = x_all.index[missing_mask].to_numpy()
        complete_indices = x_all.index[~missing_mask].to_numpy()

        n_missing_keep = min(len(missing_indices), self.fit_sample_size // 2)
        n_complete_keep = max(self.fit_sample_size - n_missing_keep, 0)

        sampled_missing = (
            rng.choice(missing_indices, size=n_missing_keep, replace=False)
            if n_missing_keep > 0
            else np.array([], dtype=int)
        )

        sampled_complete = (
            rng.choice(
                complete_indices,
                size=min(n_complete_keep, len(complete_indices)),
                replace=False,
            )
            if len(complete_indices) > 0 and n_complete_keep > 0
            else np.array([], dtype=int)
        )

        fit_indices = np.concatenate([sampled_missing, sampled_complete])

        if len(fit_indices) == 0:
            fit_indices = x_all.index.to_numpy()

        self.fit_indices_ = fit_indices

        x_fit = x_all.loc[fit_indices]
        x_fit_scaled = self.scaler.transform(x_fit)

        self.imputer.fit(x_fit_scaled)

        self.fitted_ = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("El imputador debe ajustarse con fit antes de usar transform.")

        result = df.copy()
        result[self.columns] = result[self.columns].astype(float)

        x = result[self.columns].astype(float).copy()

        rows_with_missing = x.isna().any(axis=1)

        if rows_with_missing.any():
            x_missing = x.loc[rows_with_missing]

            x_scaled = self.scaler.transform(x_missing)
            x_imputed_scaled = self.imputer.transform(x_scaled)
            x_imputed = self.scaler.inverse_transform(x_imputed_scaled)

            result.loc[rows_with_missing, self.columns] = x_imputed

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.fit(df).transform(df)
        self.train_imputed_ = result.copy()
        return result

    def explain_imputation(
        self,
        original_df: pd.DataFrame,
        imputed_df: pd.DataFrame,
        column: str,
        row_index: Optional[int] = None,
        n_neighbors: int = 5,
        display_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:

        if not self.fitted_:
            raise RuntimeError("El imputador debe ajustarse antes de explicar imputaciones.")

        if column not in self.columns:
            raise ValueError(f"La columna {column} no está dentro de las columnas imputables.")

        if row_index is None:
            missing_rows = original_df.index[original_df[column].isna()]
            if len(missing_rows) == 0:
                raise ValueError(f"No hay valores missing originales en la columna {column}.")
            row_index = missing_rows[0]

        if not pd.isna(original_df.loc[row_index, column]):
            raise ValueError(
                f"La fila {row_index} no tenía missing en {column}. "
                "Elige una fila que realmente haya sido imputada."
            )

        col_pos = self.columns.index(column)

        # Dataset real usado por el KNNImputer durante fit
        fit_df = original_df.loc[self.fit_indices_, self.columns].astype(float).copy()

        # Fila que queremos explicar
        query_df = original_df.loc[[row_index], self.columns].astype(float).copy()

        # Escalamos igual que en el imputador
        fit_scaled = self.scaler.transform(fit_df)
        query_scaled = self.scaler.transform(query_df)

        # Distancia exacta que usa KNNImputer
        distances = nan_euclidean_distances(query_scaled, fit_scaled)[0]

        # Solo pueden imputar esta variable los vecinos que tienen valor no missing en esa columna
        valid_mask = ~np.isnan(fit_scaled[:, col_pos])

        valid_distances = distances[valid_mask]
        valid_positions = np.where(valid_mask)[0]

        # Ordenamos vecinos por distancia
        order = np.argsort(valid_distances)

        neighbor_positions = valid_positions[order[:n_neighbors]]
        neighbor_distances = distances[neighbor_positions]

        neighbors_original = fit_df.iloc[neighbor_positions].copy()

        # Valores de la columna en escala original
        neighbor_values = neighbors_original[column].to_numpy(dtype=float)

        # Valores de la columna en escala escalada
        neighbor_values_scaled = fit_scaled[neighbor_positions, col_pos]

        if self.weights == "distance":
            eps = 1e-12

            # Si hay distancia 0, KNNImputer usa solo los vecinos con distancia 0
            if np.any(neighbor_distances == 0):
                zero_mask = neighbor_distances == 0
                estimated_scaled = np.mean(neighbor_values_scaled[zero_mask])
            else:
                weights = 1 / (neighbor_distances + eps)
                estimated_scaled = np.average(neighbor_values_scaled, weights=weights)
        else:
            estimated_scaled = np.mean(neighbor_values_scaled)

        # Deshacemos el escalado SOLO de esa columna
        estimated_original = (
            estimated_scaled * self.scaler.scale_[col_pos]
            + self.scaler.mean_[col_pos]
        )

        imputed_value = imputed_df.loc[row_index, column]

        output = pd.DataFrame({
            "neighbor_rank": np.arange(1, len(neighbor_positions) + 1),
            "knn_distance": neighbor_distances,
            f"{column}_neighbor_value": neighbor_values,
        }, index=neighbors_original.index)

        if display_columns is None:
            display_columns = [c for c in self.columns if c != column][:6]

        for c in display_columns:
            if c in neighbors_original.columns and c != column:
                output[c] = neighbors_original[c]

        print("=" * 100)
        print("Explicación de imputación KNN")
        print(f"Fila imputada: {row_index}")
        print(f"Variable imputada: {column}")
        print(f"Valor original: {original_df.loc[row_index, column]}")
        print(f"Valor imputado por KNNImputer: {imputed_value:.4f}")
        print(f"Valor reconstruido desde vecinos: {estimated_original:.4f}")
        print("=" * 100)

        return output
