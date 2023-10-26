import os
import sys
from enum import Enum

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from utils.eda.eda import (
    impute_with_iterative,
    impute_with_knn,
    impute_with_mean,
    impute_with_median,
)


class ImputationStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    ITERATIVE = "iterative"
    KNN = "knn"


IMPUTATION_FUNCTIONS = {
    ImputationStrategy.MEAN: impute_with_mean,
    ImputationStrategy.MEDIAN: impute_with_median,
    ImputationStrategy.ITERATIVE: impute_with_iterative,
    ImputationStrategy.KNN: impute_with_knn,
}
