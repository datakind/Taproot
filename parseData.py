import time, os
import pandas as pd

PROJECT_ROOT = os.path.dirname(__file__)

if __name__ == '__main__':
	startTime = time.time()

	localPath = os.path.join('..', 'Data')

	projApp2projEval_link_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Application-Project Evaluation links.csv'))
	projectsApps_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Applications.csv'))
	projectsAwarded_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Awarded.csv'))
	projectsEvals_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Evaluations (2010 - Q2).csv'))
	proj2vol_link_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects-Volunteers Links.csv'))
	volunteersApp_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Volunteers Applications.csv'))
	volunteersEvals_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Volunteers Evaluation (2010 - Q2).csv'))

	endTime = time.time()
	print "%.1f seconds to parse data" % (endTime - startTime)