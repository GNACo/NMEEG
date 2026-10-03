"""
@Author: Luis Miguel Ramirez  email: 
@Author: Luisa Maria Zapata Saldarriga email:luisazapatasaldarriaga@gmail.com 
"""
import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm

class GaussianBayesianRegression:
    """
    Implementación de regresión bayesiana usando procesos gaussianos.
    Se asume un kernel RBF (squared exponential) con hiperparámetros:
        - length_scale: escala de la distancia.
        - variance: varianza (amplitud) del kernel.
    sigma2 es la varianza del ruido (likelihood noise).
    Se incluye estandarización de datos para mejorar la estabilidad.
    """

    def __init__(self, kernel_params: dict, sigma2: float = 0.1):
        """
        kernel_params: dict con 'length_scale' y 'variance'
        sigma2: varianza del ruido
        """
        self.kernel_params = kernel_params
        self.sigma2 = sigma2
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.x_train = None
        self.y_train = None
        self.L = None  # Factorización de Cholesky de la matriz de covarianza

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray, length_scale: float, variance: float) -> np.ndarray:
        """
        Calcula el kernel RBF (squared exponential) entre dos conjuntos de puntos.
        """
        # Calcular las distancias al cuadrado de forma vectorizada
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return variance * np.exp(-0.5 * sqdist / (length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo a los datos de entrenamiento.
        Se estandarizan X e y para mejorar la estabilidad numérica.
        """
        # Guardar medias y desviaciones para revertir la escala
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        # Estandarizar
        X_std = (X - self.x_mean) / self.x_std
        y_std = (y - self.y_mean) / self.y_std
        
        self.x_train = X_std
        self.y_train = y_std.flatten()  # Vector columna
        
        # Calcular la matriz de covarianza para los datos de entrenamiento
        l = self.kernel_params["length_scale"]
        var = self.kernel_params["variance"]
        K = self._rbf_kernel(self.x_train, self.x_train, l, var) + self.sigma2 * np.eye(len(self.x_train))
        K += 1e-6 * np.eye(len(self.x_train))  # jitter para estabilidad
        self.L = np.linalg.cholesky(K)

    def compute_log_marginal_likelihood(self, sigma2: float, length_scale: float, variance: float) -> float:
        """
        Calcula la log-verosimilitud marginal (log marginal likelihood) del modelo.
        """
        n = len(self.x_train)
        K = self._rbf_kernel(self.x_train, self.x_train, length_scale, variance) + sigma2 * np.eye(n)
        K += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        logdet = 2 * np.sum(np.log(np.diag(L)))
        log_marginal = -0.5 * self.y_train.T.dot(alpha) - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
        return log_marginal

    def optimize_hyperparameters(self) -> None:
        """
        Optimiza los hiperparámetros (length_scale, variance y sigma2) maximizando la log-verosimilitud marginal.
        Se hace la optimización en espacio logarítmico para garantizar que los parámetros sean positivos.
        """
        def neg_log_marginal(log_params):
            log_length_scale, log_variance, log_sigma2 = log_params
            l = np.exp(log_length_scale)
            var = np.exp(log_variance)
            noise = np.exp(log_sigma2)
            return -self.compute_log_marginal_likelihood(noise, l, var)
        
        # Valores iniciales en log-espacio
        init = np.log([self.kernel_params["length_scale"], self.kernel_params["variance"], self.sigma2])
        bounds = [(np.log(1e-5), np.log(1e5)), (np.log(1e-5), np.log(1e5)), (np.log(1e-5), np.log(1e5))]
        res = minimize(neg_log_marginal, init, bounds=bounds, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError("Optimización falló: " + res.message)
        # Actualizar hiperparámetros con los valores optimizados
        l_opt, var_opt, sigma2_opt = np.exp(res.x)
        self.kernel_params["length_scale"] = l_opt
        self.kernel_params["variance"] = var_opt
        self.sigma2 = sigma2_opt
        # Recalcular la factorización con los nuevos hiperparámetros
        self.fit(self.x_train * self.x_std + self.x_mean, self.y_train * self.y_std + self.y_mean)

    def predict_with_samples(self, X_star: np.ndarray, n_samples: int = 50) -> tuple:
        """
        Realiza predicciones para nuevos datos.
        Retorna:
            - media de las predicciones (escala original)
            - desviación estándar (escala original)
            - muestras de la distribución predictiva (escala original)
            - intervalo de credibilidad al 95% (inferior y superior, escala original)
            - intervalo de predicción al 95% (incluyendo ruido, escala original)
        """
        # Estandarizar los datos de prueba
        X_star_std = (X_star - self.x_mean) / self.x_std
        n_train = len(self.x_train)
        n_star = len(X_star_std)
        l = self.kernel_params["length_scale"]
        var = self.kernel_params["variance"]
        
        # Calcular las matrices kernel cruzadas
        K_star = self._rbf_kernel(self.x_train, X_star_std, l, var)
        K_ss = self._rbf_kernel(X_star_std, X_star_std, l, var) + 1e-6 * np.eye(n_star)
        
        # Resolver para alpha utilizando la factorización de Cholesky
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        pred_mean = K_star.T.dot(alpha)
        
        # Varianza predictiva
        v = np.linalg.solve(self.L, K_star)
        pred_var = np.diag(K_ss) - np.sum(v**2, axis=0)
        pred_var = np.maximum(pred_var, 0)  # evitar valores negativos por errores numéricos
        
        # Para el intervalo de predicción se añade la varianza del ruido
        pred_var_total = pred_var + self.sigma2
        
        # Generar muestras de la distribución predictiva
        y_samples = np.array([
            np.random.multivariate_normal(pred_mean, np.diag(pred_var))
            for _ in range(n_samples)
        ]).T  # forma: (n_star, n_samples)
        
        # Revertir la estandarización
        pred_mean_orig = pred_mean * self.y_std + self.y_mean
        pred_std_orig = np.sqrt(pred_var) * self.y_std
        y_samples_orig = y_samples * self.y_std + self.y_mean
        
        # Intervalos de credibilidad (para la función latente)
        ci_bounds = norm.ppf([0.025, 0.975])
        ci_lower = pred_mean + ci_bounds[0] * np.sqrt(pred_var)
        ci_upper = pred_mean + ci_bounds[1] * np.sqrt(pred_var)
        ci_lower_orig = ci_lower * self.y_std + self.y_mean
        ci_upper_orig = ci_upper * self.y_std + self.y_mean
        
        # Intervalos de predicción (incluyen ruido)
        pi_lower = pred_mean + ci_bounds[0] * np.sqrt(pred_var_total)
        pi_upper = pred_mean + ci_bounds[1] * np.sqrt(pred_var_total)
        pi_lower_orig = pi_lower * self.y_std + self.y_mean
        pi_upper_orig = pi_upper * self.y_std + self.y_mean
        
        return (pred_mean_orig, pred_std_orig, y_samples_orig,
                ci_lower_orig, ci_upper_orig,
                pi_lower_orig, pi_upper_orig)
