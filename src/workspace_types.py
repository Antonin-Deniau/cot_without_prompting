from typing import Callable

import numpy as np
from numpy.typing import NDArray

type TokenProbPair = tuple[int, float]
type TokenLogits = list[TokenProbPair]
type PromptOutput = list[TokenLogits]

type GenerateLogits = Callable[[str, int | None, int], PromptOutput]
type Decode = Callable[[NDArray[np.int_]], str]
