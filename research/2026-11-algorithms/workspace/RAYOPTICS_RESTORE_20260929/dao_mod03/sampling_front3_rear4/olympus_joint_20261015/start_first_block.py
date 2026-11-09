import faulthandler, runpy, sys
faulthandler.dump_traceback_later(90, repeat=True)
sys.argv=['data.py','--workers','8','--limit','1']
runpy.run_path('data.py',run_name='__main__')
