"""Allow ``python -m citesage.evaluation`` to launch run_eval."""

import sys

from .run_eval import main

sys.exit(main())
