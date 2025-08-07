'''
@Author: Luis Miguel Ramirez 
@Author: Luisa María Zapata Saldarriaga
'''

import numpy as np
import math
import pandas as pd
from matplotlib import  pyplot as plt
from typing import Tuple 
from sklearn.cluster import KMeans 
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import norm

class BayesianLinearRegression:
    def __init__(self, kernel_params, sigma2=0.1, alpha2=1.0):
        self.kernel_params = kernel_params
        self.sigma2 = sigma2
        self.alpha2 = alpha2
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.bases = None
        self.phi = None
        self.m = None
        self.V_n = None
        self.y = None
        self.w_d = None  # Pesos de covariables para modelar el ruido
        self._log_context = {"feature":None, "file_key":None, "tag":None}
        self.heteroscedastic = False  # por defecto, modelo homocedástico

        
    def set_log_context(self, feature=None, file_key=None, tag=None, logger=None):
        self._log_context.update({"feature": feature, "file_key": file_key, "tag": tag})
        self._logger = logger
    def random_bases(self, x: np.ndarray) -> np.ndarray:
        """
        modelNormative copy.ipynb
        """
        n = x.shape[0]
        if self.kernel_params["number_of_bases"] >= n:
            raise ValueError("number_of_bases should be less than number of training points")
        idx = np.random.choice(n, size=self.kernel_params["number_of_bases"], replace=False)
        return x[idx, :]

    def k_means_bases(self, x: np.ndarray) -> np.ndarray:
        kmeans = KMeans(n_clusters=self.kernel_params["number_of_bases"], random_state = 42).fit(x)
        return kmeans.cluster_centers_

    def create_bases(self, x: np.ndarray) -> np.ndarray:
        method = self.kernel_params.get("bases_sampling_method", "Random")
        if method == "KMeans":
            return self.k_means_bases(x)
        elif method == "Random":
            return self.random_bases(x)
        else:
            raise KeyError("Unknown bases sampling method: " + str(method))


    def _parse_hyps(self, hyp, Xv=None):
        """
        Parsea los hiperparámetros para obtener la varianza (heterocedástica) y la precisión de los coeficientes.
        Se espera que `Xv` contenga las covariables (e.g., sexo).
        """
        if Xv is not None:
            if Xv.ndim == 1:
                Xv = Xv[:, np.newaxis]
            Dv = Xv.shape[1]
            w_d = np.asarray(hyp[:Dv])
            beta = np.exp(Xv @ w_d)
            alpha = np.exp(hyp[Dv:])
        else:
            # En caso de que no haya covariables, usar solo hiperparámetro escalar
            beta = np.exp(hyp[0])
            alpha = np.exp(hyp[1:])

        return beta, alpha

    
    def create_design_matrix(self, x: np.ndarray) -> np.ndarray:
        num_datapoints, num_bases = x.shape[0], self.bases.shape[0]
        phi = np.zeros((num_datapoints, num_bases))
        for j in range(num_bases):
            # Compute the squared Euclidean distance between the datapoints and the j-th base
            diff = x - self.bases[j, :]  # (num_datapoints, num_features) - (num_features,)
            sq_dist = np.sum(diff ** 2, axis=1)  # Squared Euclidean distance, result shape: (num_datapoints,)
            
            # Apply the exponential kernel (RBF kernel)
            phi[:, j] = np.exp(-sq_dist / (2 * self.kernel_params["length_scale"] ** 2))
        return np.hstack([np.ones((num_datapoints, 1)), phi])

  
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        # Guardar valores originales de x y y para la transformación
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

        # Estandarizar x y normalizar y
        x_standardized = (x - self.x_mean) / self.x_std
        y_normalized = (y - self.y_mean) / self.y_std

        # Crear las bases y la matriz de diseño
        self.bases = self.create_bases(x_standardized)
        self.phi = self.create_design_matrix(x_standardized)
        num_bases = self.bases.shape[0]

        # Asume homocedasticidad
        self.lambda_n_vec = np.ones_like(y_normalized) * self.sigma2

        # Calcular la covarianza posterior y la media
        W_inv = 1.0 / self.lambda_n_vec
        phi_scaled = self.phi * W_inv[:, np.newaxis]
        self.V_n = np.linalg.inv((1 / self.alpha2) * np.eye(num_bases + 1) + self.phi.T @ phi_scaled)
        self.m = self.V_n @ self.phi.T @ (W_inv * y_normalized)
        self.y = y_normalized
        self.heteroscedastic = False  # Asumir homocedasticidad por defecto


    def compute_log_marginal_likelihood(self, sigma2, alpha2):  
        from scipy.linalg import cho_solve, cho_factor
        #Usa la descomposición de Cholesky o métodos iterativos en lugar de calcular C directamente:Esto evita el cálculo explícito de C_inv_y con np.linalg.solve(), reduciendo el uso de memoria.
        n = self.phi.shape[0]  # Number of data points
        # Compute the marginal covariance matrix C
        C = sigma2 * np.eye(n) + (1.0 / alpha2) * (self.phi @ self.phi.T)
        
        # Añadir regularización para evitar problemas con matrices mal condicionadas
        epsilon = 1e-6  # Valor pequeño para regularización
        C += np.eye(C.shape[0]) * epsilon  # Regularizamos la matriz C
        
        try:
            sign, logdet = np.linalg.slogdet(C)
            L = cho_factor(C)  # Cholesky factorization
            C_inv_y = cho_solve(L, self.y)  # Solve using Cholesky
        except np.linalg.LinAlgError:
            print(np.inf)  # In case Cholesky fails, return a large loss
            # Si Cholesky falla, usamos SVD como alternativa
            print("Warning: Cholesky decomposition failed, using SVD instead.")
            U, s, Vh = np.linalg.svd(C)
            # Usamos SVD para calcular la inversa de C y resolver el sistema
            C_inv_y = np.dot(Vh.T, np.dot(np.diag(1/s), np.dot(U.T, self.y)))
            logdet = 2 * np.sum(np.log(s))  # Log-determinante desde SVD
            sign = 1  # El signo de la matriz es 1 ya que es positiva definida
        

        #C_inv_y = np.linalg.solve(C, self.y)
        quad_term = -0.5 * self.y.T @ C_inv_y
        log_marginal = -0.5 * logdet + quad_term - (n / 2) * np.log(2 * np.pi)
        # Asegurarse de que el resultado sea finito para evitar problemas numéricos
        if not np.isfinite(log_marginal):
            log_marginal = 1 / np.finfo(float).eps
        
        return log_marginal
    def fit_with_heteroscedasticity_and_optimization(self, x, y, Xv):
        self.x_mean = np.mean(x, axis=0)
        self.x_std = np.std(x, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

        x_std = (x - self.x_mean) / self.x_std
        y_std = (y - self.y_mean) / self.y_std
        Xv = np.array(Xv)

        # 🔧 Escalado seguro para Xv (covariables)
        self.Xv_mean = np.mean(Xv, axis=0)
        self.Xv_std = np.std(Xv, axis=0)
        Xv_scaled = (Xv - self.Xv_mean) / self.Xv_std

        self.bases = self.create_bases(x_std)
        self.phi = self.create_design_matrix(x_std)
        Phi = self.phi
        N, M = Phi.shape
        alpha = self.alpha2
        self.heteroscedastic = True  # Asumir heterocedasticidad

        def nlZ(w_d):
            z = np.clip(Xv_scaled @ w_d, -10, 10)  # 🔥 Clip para evitar overflow
            lambda_n = np.exp(z)
            W_inv = 1.0 / lambda_n
            Phi_scaled = Phi * W_inv[:, None]

            try:
                A = Phi.T @ Phi_scaled + alpha * np.eye(M)
                V_n = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                return 1e10

            m = V_n @ Phi.T @ (W_inv * y_std)
            residual = y_std - Phi @ m
            try:
                logdet = np.linalg.slogdet(A)[1]
            except np.linalg.LinAlgError:
                logdet = 1e10

            nlZ_val = 0.5 * (
                np.sum(np.log(1 / lambda_n)) +
                residual.T @ (W_inv * residual) +
                logdet +
                M * np.log(alpha)
            )
            lambda_reg = 0.1  # o menor, prueba también con 0.01
            nlZ_val += lambda_reg * np.sum(w_d**2)

            return nlZ_val

        Dv = Xv_scaled.shape[1]
        w_d0 = np.zeros(Dv)

        print("Xv_scaled shape:", Xv_scaled.shape)
        print("First rows:\n", Xv_scaled[:5])
        print("Initial w_d:", w_d0)
        print("y shape:", y.shape)

        res = minimize(nlZ, w_d0, method='L-BFGS-B',
                    options={'disp': True, 'maxiter': 200, 'gtol': 1e-4})

        if not res.success:
            raise RuntimeError(f"Optimización de w_d falló: {res.message}")

        self.w_d = res.x
        z_final = np.clip(Xv_scaled @ self.w_d, -10, 10)
        self.lambda_n_vec = np.exp(z_final)

        W_inv = 1.0 / self.lambda_n_vec
        Phi_scaled = Phi * W_inv[:, None]
        A = Phi.T @ Phi_scaled + alpha * np.eye(M)
        self.V_n = np.linalg.inv(A)
        self.m = self.V_n @ Phi.T @ (W_inv * y_std)
        self.y = y_std

    def _make_psd(self, matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Fuerza la matriz a ser simétrica positiva semidefinida (PSD).
        Clipping de valores propios negativos y simetrización.
        """
        matrix = (matrix + matrix.T) / 2  # Forzar simetría
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals[eigvals < eps] = eps  # Clipping de valores propios negativos
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


    def optimize_hyperparameters(self) -> None:
        def neg_log_marginal_likelihood(params):
            log_sigma2, log_alpha2 = params  # Optimizing in log-space
            sigma2, alpha2 = np.exp(log_sigma2), np.exp(log_alpha2)  # Convert back
            val = -self.compute_log_marginal_likelihood(sigma2, alpha2)
            print(f"[Hyperopt] sigma2={sigma2:.4f}, alpha2={alpha2:.4f}, -logML={val:.2f}")
            return val

        # Set initial values in log-space (avoid poor initialization)
        init_params = np.log([max(self.sigma2, 1e-5), max(self.alpha2, 1e-5)])
        bounds =[(np.log(0.1), np.log(5)), (np.log(1), np.log(1000))]

        result = minimize(neg_log_marginal_likelihood, init_params, bounds=bounds, method="L-BFGS-B") # "powell"
        if not result.success:
            raise RuntimeError("Optimización de hiperparámetros falló: " + result.message)
        # Convert back to original scale safely
        self.sigma2, self.alpha2 = np.exp(result.x)

    def predict_and_adjust(self, hyp, X, y, Xs, Xv_adapt=None, Xv_test=None):
        """
        Ajusta las predicciones usando los residuos de un conjunto de adaptación.
        Se asume que las covariables (como sexo) están en `Xv_adapt` y `Xv_test`.
        """
        ys_out = np.zeros(Xs.shape[0])
        s2_out = np.zeros(Xs.shape[0])

        # Obtener predicciones en conjunto de adaptación
        ys_ref, _ = self.predict_with_samples(X[Xv_adapt], n_samples=500, Xv_test=Xv_adapt)
        residuals = ys_ref - y[Xv_adapt]
        residuals_mu = np.mean(residuals)
        residuals_sd = np.std(residuals)

        # Predecir para test
        ys_out, s2_dummy, _, _, _, _, _ = self.predict_with_samples(Xs, n_samples=500, Xv_test=Xv_test)
        ys_out -= residuals_mu
        s2_out[:] = residuals_sd**2

        return ys_out, s2_out

    
    def predict_with_samples(self, x_star: np.ndarray, n_samples: int =500,  Xv_test=None ) -> tuple:
        # Standardize x_star
        x_star_standardized = (x_star - self.x_mean) / self.x_std

        # Create design matrix for x_star
        phi_star = self.create_design_matrix(x_star_standardized)
        def is_positive_semidefinite(matrix):
            min_eig = np.min(np.linalg.eigvalsh(matrix))
            if min_eig < -1e-8:
                # Llamar logger si está disponible
                if self._logger:
                    self._logger(self._log_context["feature"],
                                self._log_context["file_key"],
                                self._log_context["tag"],
                                min_eig)
                return False
            return True
        
        if not is_positive_semidefinite(self.V_n):
            print("⚠️ V_n no es semi-definida positiva. Añadiendo jitter.")
            jitter = 1e-6 * np.eye(self.V_n.shape[0])
            Vn_to_use = self.V_n + jitter
        else:
            Vn_to_use = self.V_n
        posterior_cov_psd = self._make_psd(Vn_to_use)
        posterior_weights = np.random.multivariate_normal(
            mean=np.asarray(self.m).flatten(),
            cov=posterior_cov_psd,
            size=n_samples
        )

        

        # Generate predictive outputs for each sample
        y_samples = np.dot(phi_star, posterior_weights.T)  # shape: (n_test, n_samples)

        # Compute mean and standard deviation across samples (epistemic)
        pred_mean  = y_samples.mean(axis=1)
        pred_std  = y_samples.std(axis=1)

        # Compute aleatoric uncertainty from heteroscedastic model (if Xv_test is provided)
        if self.heteroscedastic:
            if self.w_d is None or Xv_test is None:
                raise ValueError("Missing covariates (Xv_test) or weights (w_d) in heteroscedastic mode.")
            Xv_test = np.array(Xv_test)
            if Xv_test.ndim == 1:
                Xv_test = Xv_test[:, np.newaxis]
            Xv_test_scaled = (Xv_test - self.Xv_mean) / self.Xv_std
            z = np.clip(Xv_test_scaled @ self.w_d, -10, 10)
            sigma2_test = 1.0 / np.exp(z)
        else:
            sigma2_test = self.sigma2 * np.ones_like(pred_std)
            #sigma2_test = self.sigma2 * self.y_std ** 2 * np.ones_like(pred_std)



        # Varianza total (epistemic + aleatoric)
        pred_std_total = np.sqrt(pred_std**2 + sigma2_test)

        # Revert normalization to original scale
        y_samples_original_scale = y_samples * self.y_std + self.y_mean
        y_mean_original_scale = pred_mean * self.y_std + self.y_mean
        y_std_total_original_scale = pred_std_total * self.y_std

        # Intervalos de credibilidad al 95%
        ci_bounds = norm.ppf([0.025, 0.975])
        ci_lower = pred_mean + ci_bounds[0] * pred_std
        ci_upper = pred_mean + ci_bounds[1] * pred_std

        # Intervalos de predicción (incluyen ruido)
        pi_lower = pred_mean + ci_bounds[0] * pred_std_total
        pi_upper = pred_mean + ci_bounds[1] * pred_std_total

        ci_lower_orig = ci_lower * self.y_std + self.y_mean
        ci_upper_orig = ci_upper * self.y_std + self.y_mean
        pi_lower_orig = pi_lower * self.y_std + self.y_mean
        pi_upper_orig = pi_upper * self.y_std + self.y_mean

        return y_mean_original_scale, y_std_total_original_scale, y_samples_original_scale, ci_lower_orig, ci_upper_orig, pi_lower_orig, pi_upper_orig

    def get_variance_decomposition(self, x_star: np.ndarray, Xv_test: np.ndarray = None):
        """
        Devuelve la varianza total, aleatoric y epistemic para cada punto de test.
        Si se proporciona Xv_test y w_d, calcula varianza aleatoric heterocedástica.
        """
        # Estandarizar x_star
        x_star_standardized = (x_star - self.x_mean) / self.x_std
        phi_star = self.create_design_matrix(x_star_standardized)

        # Predicción puntual
        mean_pred = phi_star @ self.m

        # Epistemic: incertidumbre del modelo
        epistemic_var = np.sum(phi_star @ self.V_n * phi_star, axis=1)

        # Aleatoric: si hay covariables, usar w_d para estimarla
        if self.heteroscedastic:
            if self.w_d is None or Xv_test is None:
                raise ValueError("Missing covariates (Xv_test) or weights (w_d) in heteroscedastic mode.")
            Xv_test = np.array(Xv_test)
            if Xv_test.ndim == 1:
                Xv_test = Xv_test[:, np.newaxis]
            Xv_test_scaled = (Xv_test - self.Xv_mean) / self.Xv_std
            z = np.clip(Xv_test_scaled @ self.w_d, -10, 10)
            aleatoric_var = 1.0 / np.exp(z)
        else:
            aleatoric_var = self.sigma2 * np.ones_like(epistemic_var)


        total_var = epistemic_var + aleatoric_var

        return total_var, aleatoric_var, epistemic_var

    
    def predict_centile_bands(self, x_star: np.ndarray, Xv_test: np.ndarray = None,
                          percentiles=[1, 5, 25, 75, 95, 99], factor=0.5):
        total_var, aleatoric_var, epistemic_var = self.get_variance_decomposition(x_star, Xv_test=Xv_test)

        # Escalar a la varianza original
        total_var *= self.y_std ** 2
        epistemic_var *= self.y_std ** 2

        s2_lower = total_var - factor * epistemic_var
        s2_upper = total_var + factor * epistemic_var

        # Escalar la predicción mediana también
        median = self.median_prediction(x_star) * self.y_std + self.y_mean

        bounds = {}
        for p in percentiles:
            z = norm.ppf(p / 100)
            lower = median + z * np.sqrt(total_var)
            upper = median + z * np.sqrt(total_var)
            bounds[p] = (lower, upper)

        return bounds


    def median_prediction(self, x_star: np.ndarray):
        x_star_standardized = (x_star - self.x_mean) / self.x_std
        phi_star = self.create_design_matrix(x_star_standardized)
        return (phi_star @ self.m).flatten()