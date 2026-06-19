import numpy as np
import util

import logging
logger = logging.getLogger(__name__)

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions
    """
    logger.info("Starting training process.")
    
    X_train, y_train = util.load_dataset(train_path, 'y', True)
    X_eval, y_eval_actual = util.load_dataset(eval_path, 'y', True)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_eval_predicted = clf.predict(X_eval)
    np.savetxt(pred_path, y_eval_predicted, delimiter=',')

    error_rate = np.mean(np.absolute(np.round(y_eval_predicted) - y_eval_actual))
    logger.info(f"Error rate for evaluation: {error_rate}")

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.
    
    Example usage:
        > clf = LogisticRegression()
        > clf.fit(X_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__():
        super().__init__()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Run Newton's Method to minimize J(theta) for logistic regression
        
        Args:
            x: Training example inputs. Shape(m, n).
            y: Training example labels. Shape (m,).
        """

        logger.info("Computing the log likelihood function given the current value of theta.")

        def compute_loss(theta: float) -> float:
            """Compute the log likelihood function given the current value of theta
            
            Args:
                theta: Theta to compute the loss function
            
            Returns:
                The computed value of the loss function.
            """

            m = x.shape[0]
            sum = 0
            for i in range(m):
                h_i = 1 / (1 + np.exp(-np.dot(theta, x[i,:])))
                sum += y[i]*h_i + (1-y[i])*(1-h_i)
            loss = -sum/m
            return loss

        for i in range(self.max_iter):

            theta_new = compute_loss(self.theta) / util.derivative(compute_loss, self.theta) * self.theta
            
            logger.info(f"Theta value at interation {i+1}: {theta_new}.")
            if self.verbose:
                print(f"Theta value at interation {i+1}: {theta_new}.")

            dist = abs(theta_new - self.theta)
            self.theta = theta_new

            if dist <= self.eps:
                logger.info(f"Theta value converged after {i} iterations at {self.theta}.")
                if self.verbose:
                    print(f"Theta value converged after {i} iterations at {self.theta}.")
                break

            if dist > self.eps and i == self.max_iter:
                logger.info(f"Theta value does not converge after {i} iterations at which its value: {self.theta}.")
                if self.verbose:
                    print(f"Theta value does not converge after {i} iterations at which its value: {self.theta}.")
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions given new inputs x.
        
        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        m = x.shape[0]
        predictions = np.array([1/(1 + np.exp(-np.dot(self.theta, x[i,:]))) for i in range(m)]).reshape(m,)

        logger.info(f"Predictions of y given x: {predictions}")
        if self.verbose:
            print(f"Predictions of y given x: {predictions}")

        return predictions
