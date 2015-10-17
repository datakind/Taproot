import time, os
from collections import defaultdict
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(__file__)

if __name__ == '__main__':
	startTime = time.time()

	localPath = os.path.join('..', 'Data')

	# read in data
	projApp2projEval_link_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Application-Project Evaluation links.csv'))
	projectsApps_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Applications.csv'))
	projectsAwarded_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects Awarded.csv'))
	projectsEvals_df = pd.read_csv(os.path.join(PROJECT_ROOT, '..', 'Codes', 'Output', 'NPO_Eval_2010Q2.csv'))
	proj2vol_link_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Projects-Volunteers Links.csv'))
	volunteersApp_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Volunteers Applications.csv'))
	volunteersEvals_df = pd.read_csv(os.path.join(PROJECT_ROOT, localPath, 'Volunteers Evaluation (2010 - Q2).csv'))

	# there are 60 duplicate emails in projectsEvals
	duplicateEmails = []
	for email, c in zip(projectsEvals_df.groupby(['Email Address'])['Email Address'].count().index,
		projectsEvals_df.groupby(['Email Address'])['Email Address'].count()):
		if c > 1:
			duplicateEmails.append([email, c])
	print duplicateEmails

	# remove junk row
	projectsEvals_df.drop(projectsEvals_df.index[1], inplace=True)

	# do any emails show up with multiple org_names? (NO: every email is associated with a single org name!)
	emails = defaultdict(list)
	for row in np.array(projApp2projEval_link_df):
		org_name = row[1]
		email = row[-1]
		if email not in emails:
			emails[email].append(org_name)
	for email, org_names in emails.items():
		if len(set(org_names)) > 1:
			print email, org_names

	# remove duplicate emails from projApp2projEval_link_df
	# unique key = 'app contact email'
	tmp_array = []
	emails = {}
	for row in np.array(projApp2projEval_link_df):
		email = row[-1]
		if email not in emails:
			emails[email] = None
			tmp_array.append(row)
	projApp2projEval_link_df = pd.DataFrame(np.array(tmp_array), columns=projApp2projEval_link_df.columns)

	# combine dataframes
	projectsComb_df = pd.merge(projectsApps_df, projectsAwarded_df, left_on=['sgapp_id', 'org_id'], 
		right_on=['sgapp_id', 'org_id'], how='left')
	# 1200 evals
	projectsAppEvalComb_df = pd.merge(projApp2projEval_link_df, projectsEvals_df,
		left_on='app contact email', right_on='Email Address', how='inner')
	# NOTE: some emails in the evaluation file don't match with application-evaluation-links file

	# use this file for feature building
	projects_df = pd.merge(projectsComb_df, projectsAppEvalComb_df, left_on='sgapp_id', right_on='sgaid', how='inner')

	# add outcome label
	projects_df['outcome'] = None
	projects_df.ix[projects_df['How satisfied with the overall experience? '] < 6, 'outcome'] = 0
	projects_df.ix[projects_df['How satisfied with the overall experience? '] == 6, 'outcome'] = 1

	# for each ord_name, choose the first entry. This way, we don't bias the model.
	
	# find the earliest timestamp for each org_id
	org_ids = {}
	for row in np.array(projects_df):
		org_id = row[1]
		date_app_completed = row[4]
		if org_id not in org_ids:
			org_ids[org_id] = pd.to_datetime(date_app_completed)
		elif org_ids[org_id] > pd.to_datetime(date_app_completed):
			org_ids[org_id] = pd.to_datetime(date_app_completed)
	
	# if timestamp matches, and org_id not in array
	tmp_ids = {}
	tmp_array = []
	for row in np.array(projects_df):
		org_id = row[1]
		date_app_completed = row[4]
		if org_id not in tmp_ids and pd.to_datetime(date_app_completed) == org_ids[org_id]:
			tmp_ids[org_id] = None
			tmp_array.append(row)
	projects_df = pd.DataFrame(np.array(tmp_array), columns=projects_df.columns)

	projects_df.to_csv(os.path.join(PROJECT_ROOT, localPath, 'projects.csv'), index=False)

	##################
	# TODO:
	# 1) If an organization repeats, what is their satisfaction level?

	##################

	# start building features


	endTime = time.time()
	print "%.1f seconds to parse data" % (endTime - startTime)