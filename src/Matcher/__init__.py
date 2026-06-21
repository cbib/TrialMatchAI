from __future__ import annotations

import warnings

from trialmatchai import __version__

warnings.warn(
    "The 'Matcher' namespace is deprecated. Import from 'trialmatchai' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["__version__"]
