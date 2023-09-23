import os
import runpy
globals().update(runpy.run_path(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'embed/embed.py')))